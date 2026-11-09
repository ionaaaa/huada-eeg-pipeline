"""Native Huada LaBraM/REVE runtimes backed by portable age checkpoint v1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch
import torch.nn as nn

from scripts.checkpoints.portable_age_checkpoint import PortableAgeCheckpoint

from .contracts import HUADA_CHANNEL_AXIS, position_bank_fingerprint, validate_huada_target_contract


class NativeBackendError(RuntimeError):
    pass


class _HuadaLabramWrapper(nn.Module):
    """Huada channel-name adapter; positions are validity markers, not geometry."""

    def __init__(self, model: nn.Module, mapped_names: list[str]) -> None:
        super().__init__()
        self.model = model
        self._labram_names = list(mapped_names)
        self.register_buffer("_positionless_mask", torch.zeros(64, dtype=torch.bool))
        self.register_buffer("_known_mask", torch.ones(64, dtype=torch.bool))

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        valid = (channel_positions != -0.1).any(dim=-1)
        valid = valid | self._positionless_mask
        valid = valid & self._known_mask
        common = valid.all(dim=0)
        if int(common.sum().item()) != 64:
            raise NativeBackendError("LaBraM Huada contract requires all 64 mapped channels")
        names = [self._labram_names[index] for index in common.nonzero(as_tuple=True)[0].tolist()]
        return self.model(x[:, common, :], ch_names=names, return_all_tokens=True)


class _HuadaReveWrapper(nn.Module):
    def __init__(self, model: nn.Module, position_bank: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("huada_position_bank", position_bank, persistent=False)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        pos = self.huada_position_bank.unsqueeze(0).expand(eeg.shape[0], -1, -1)
        output = self.model(eeg, pos=pos, return_output=True)
        if type(output) not in (list, tuple) or not output:
            raise NativeBackendError("REVE encoder did not return its layer outputs")
        return output[-1]


class _BaseNativeAgeBackend(nn.Module):
    family: str
    feature_dim: int
    token_count: int

    def __init__(
        self,
        checkpoint: PortableAgeCheckpoint,
        contract: Mapping[str, Any],
        wrapped_model: nn.Module,
    ) -> None:
        super().__init__()
        self.contract = validate_huada_target_contract(
            dict(contract), expected_family=self.family
        )
        if checkpoint.model_family != self.family:
            raise NativeBackendError("portable checkpoint family mismatch")
        self.wrapped_model = wrapped_model
        self.probe = nn.Linear(self.feature_dim, 1)
        probe_result = self.probe.load_state_dict(
            dict(checkpoint.age_probe_state_dict), strict=True
        )
        if probe_result.missing_keys or probe_result.unexpected_keys:
            raise NativeBackendError("portable age probe strict load failed")
        transform = checkpoint.payload["target_transform"]
        self.register_buffer(
            "target_mean", torch.tensor(float(transform["mean"]), dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "target_std", torch.tensor(float(transform["std"]), dtype=torch.float64), persistent=False
        )

    def _validate_window(self, value: torch.Tensor) -> None:
        if value.ndim != 3 or tuple(value.shape[1:]) != (64, 400):
            raise NativeBackendError(f"native backend expects [B,64,400], got {tuple(value.shape)}")
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise NativeBackendError("native backend input must be finite float32")

    def _backbone_forward(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_window(value)
        tokens = self._backbone_forward(value)
        expected = (value.shape[0], self.token_count, self.feature_dim)
        if tuple(tokens.shape) != expected:
            raise NativeBackendError(
                f"backbone output shape mismatch: expected={expected}, actual={tuple(tokens.shape)}"
            )
        representation = tokens.mean(dim=1)
        z = self.probe(representation)
        # Preserve the source runtime's float32 output semantics while using
        # the full-precision Python scalar stored in the portable checkpoint.
        age = z * float(self.target_std.item()) + float(self.target_mean.item())
        return {
            "backbone_tokens": tokens,
            "representation": representation,
            "predicted_z": z,
            "predicted_age_years": age,
        }


class HuadaLaBraMAgeBackend(_BaseNativeAgeBackend):
    family = "labram"
    feature_dim = 200
    token_count = 129

    @classmethod
    def build(
        cls,
        checkpoint: PortableAgeCheckpoint,
        contract: Mapping[str, Any],
        *,
        model_factory: Callable[[], nn.Module] | None = None,
    ) -> "HuadaLaBraMAgeBackend":
        validated = validate_huada_target_contract(dict(contract), expected_family="labram")
        if model_factory is None:
            from braindecode.models import Labram

            model_factory = lambda: Labram(n_times=400, n_chans=64, n_outputs=0)
        inner = model_factory()
        state = dict(checkpoint.backbone_state_dict)
        source_adapter_keys = {
            "channel_wrapper._positionless_mask",
            "channel_wrapper._known_mask",
        }
        missing_adapter = source_adapter_keys.difference(state)
        if missing_adapter:
            raise NativeBackendError(
                f"portable LaBraM source adapter is incomplete: {sorted(missing_adapter)}"
            )
        inner_state = {key: value for key, value in state.items() if key not in source_adapter_keys}
        result = inner.load_state_dict(inner_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise NativeBackendError("portable LaBraM strict backbone load failed")
        mapped = list(validated["channel_mapping"]["mapped_channel_names"])
        wrapper = _HuadaLabramWrapper(inner, mapped)
        return cls(checkpoint, validated, wrapper)

    def _backbone_forward(self, value: torch.Tensor) -> torch.Tensor:
        markers = torch.zeros(
            value.shape[0], 64, 3, dtype=value.dtype, device=value.device
        )
        return self.wrapped_model(value, channel_positions=markers)


class HuadaREVEAgeBackend(_BaseNativeAgeBackend):
    family = "reve"
    feature_dim = 512
    token_count = 128

    @classmethod
    def build(
        cls,
        checkpoint: PortableAgeCheckpoint,
        contract: Mapping[str, Any],
        *,
        model_factory: Callable[[], nn.Module] | None = None,
    ) -> "HuadaREVEAgeBackend":
        validated = validate_huada_target_contract(dict(contract), expected_family="reve")
        if model_factory is None:
            from braindecode.models import REVE

            model_factory = lambda: REVE(n_times=400, n_chans=64, n_outputs=2)
        inner = model_factory()
        result = inner.load_state_dict(dict(checkpoint.backbone_state_dict), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise NativeBackendError("portable REVE strict backbone load failed")
        bank = validated["position_bank"]
        positions = torch.tensor(bank["positions"], dtype=torch.float32)
        if position_bank_fingerprint(HUADA_CHANNEL_AXIS, positions.numpy()) != bank["fingerprint"]:
            raise NativeBackendError("REVE target position bank changed before construction")
        wrapper = _HuadaReveWrapper(inner, positions)
        return cls(checkpoint, validated, wrapper)

    def _backbone_forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.wrapped_model(value)


def build_native_age_backend(
    checkpoint: PortableAgeCheckpoint,
    contract: Mapping[str, Any],
    *,
    model_factory: Callable[[], nn.Module] | None = None,
) -> _BaseNativeAgeBackend:
    if checkpoint.model_family == "labram":
        return HuadaLaBraMAgeBackend.build(
            checkpoint, contract, model_factory=model_factory
        )
    if checkpoint.model_family == "reve":
        return HuadaREVEAgeBackend.build(
            checkpoint, contract, model_factory=model_factory
        )
    raise NativeBackendError(
        f"native age backend does not support {checkpoint.model_family!r}"
    )
