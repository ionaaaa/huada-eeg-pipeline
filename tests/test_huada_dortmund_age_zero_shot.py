from __future__ import annotations

import copy
import csv
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.downstream_50m.dortmund_huada_zero_shot import (
    CHECKPOINT_KIND, MASK_SEMANTICS_VERSION, SOURCE_AGE_MEAN,
    SOURCE_AGE_STD, CheckpointContract, ZeroShotValidationError,
    build_formal_model, compute_metric_bundle, finalize_prediction_rows, infer_label_blind,
    label_blind_collate, load_strict_checkpoint, recompute_from_csv,
    validate_checkpoint_payload, validate_coverage, validate_input_compatibility,
)
from scripts.downstream_50m.run_huada_dortmund_age_zero_shot import (
    WINDOW_FIELDS, main as zero_shot_main,
)


VOCAB_SHA = "a" * 64
CANONICAL_SHA = "b" * 64


def test_formal_model_contract_counts_without_loading_checkpoint():
    model = build_formal_model()
    assert len(model.backbone.state_dict()) == 154
    assert sum(t.numel() for t in model.backbone.state_dict().values()) == 38_194_713
    assert list(model.head.weight.shape) == [1, 512]
    assert list(model.head.bias.shape) == [1]


class TinyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        self.head = nn.Linear(4, 1)

    def forward(self, token_inputs, token_channel_indices, token_time_indices, token_valid_mask):
        del token_channel_indices, token_time_indices
        pooled = (token_inputs[:, :, :3] * token_valid_mask.unsqueeze(-1)).sum(1)
        pooled = pooled / token_valid_mask.sum(1, keepdim=True)
        return self.head(self.backbone(pooled))


def tiny_contract(model: TinyWrapper | None = None) -> CheckpointContract:
    model = model or TinyWrapper()
    return CheckpointContract(
        backbone_tensor_count=len(model.backbone.state_dict()),
        backbone_parameter_count=sum(x.numel() for x in model.backbone.state_dict().values()),
        head_tensor_count=2, head_parameter_count=5,
        head_weight_shape=(1, 4), head_bias_shape=(1,), feature_dim=4,
    )


def fixture_payload(model: TinyWrapper | None = None) -> dict:
    model = model or TinyWrapper()
    return {
        "checkpoint_kind": CHECKPOINT_KIND,
        "backbone_state_dict": copy.deepcopy(model.backbone.state_dict()),
        "age_probe_state_dict": copy.deepcopy(model.head.state_dict()),
        "best_epoch": 20,
        "model_architecture_config": {
            "class": "EEGPretrainModel", "d_model": 4, "depth": 12,
            "n_heads": 8, "mlp_ratio": 4.0, "dropout": 0.1,
            "patch_len": 100, "num_tokens": 640, "num_time_patches": 10,
            "n_channels": 64, "probe_layer_idx": 8,
            "pooling": "masked_mean", "feature_dim": 4,
        },
        "electrode_vocabulary": {"sha256": "c" * 64, "content": {"sha256": VOCAB_SHA}, "version": "fixture-v1"},
        "channel_canonicalization": {"sha256": CANONICAL_SHA, "version": "fixture-v1"},
        "token_mask_semantics_version": MASK_SEMANTICS_VERSION,
        "target_transform": {
            "target_transform": "zscore", "source_age_mean": 27.5,
            "source_age_std": 4.218591189948011, "ddof": 0,
            "target_unit": "year", "equal_subject_weight": True,
            "window_weighted": False,
        },
    }


def save_fixture_checkpoint(tmp_path: Path, payload: dict | None = None, name: str = "best_full_checkpoint.pt"):
    path = tmp_path / name
    torch.save(payload or fixture_payload(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_strict_fixture_checkpoint_loads_frozen_with_no_training_objects(tmp_path):
    path, digest = save_fixture_checkpoint(tmp_path)
    model, report, _ = load_strict_checkpoint(
        path, digest, torch.device("cpu"), model_factory=TinyWrapper,
        contract=tiny_contract(),
    )
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert report["missing_keys"] == report["unexpected_keys"] == 0
    assert report["shape_mismatch"] == report["dtype_mismatch"] == 0
    assert report["optimizer_created"] is report["scheduler_created"] is report["backward_called"] is False


def test_sha_mismatch_and_last_checkpoint_fail_closed(tmp_path):
    path, _ = save_fixture_checkpoint(tmp_path)
    with pytest.raises(ZeroShotValidationError, match="SHA256"):
        load_strict_checkpoint(path, "0" * 64, torch.device("cpu"), model_factory=TinyWrapper, contract=tiny_contract())
    last, digest = save_fixture_checkpoint(tmp_path, name="last_training_checkpoint.pt")
    with pytest.raises(ZeroShotValidationError, match="forbidden"):
        load_strict_checkpoint(last, digest, torch.device("cpu"), model_factory=TinyWrapper, contract=tiny_contract())


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(checkpoint_kind="wrong"), "checkpoint_kind"),
        (lambda p: p.update(best_epoch=19), "best_epoch"),
        (lambda p: p["model_architecture_config"].update(probe_layer_idx=7), "architecture"),
        (lambda p: p["model_architecture_config"].update(pooling="flatten"), "architecture"),
        (lambda p: p["target_transform"].update(source_age_mean=26.0), "source_age_mean"),
        (lambda p: p["target_transform"].update(source_age_std=3.0), "source_age_std"),
        (lambda p: p.update(token_mask_semantics_version="wrong"), "mask semantics"),
        (lambda p: p.update(optimizer_state_dict={}), "last/resume"),
    ],
)
def test_checkpoint_metadata_fail_closed(mutation, match):
    model = TinyWrapper(); payload = fixture_payload(model)
    mutation(payload)
    with pytest.raises(ZeroShotValidationError, match=match):
        validate_checkpoint_payload(payload, model=model, contract=tiny_contract(model))


def test_checkpoint_key_shape_and_dtype_fail_closed():
    for mode in ("missing", "shape", "dtype"):
        model = TinyWrapper(); payload = fixture_payload(model)
        if mode == "missing":
            payload["backbone_state_dict"].pop(next(iter(payload["backbone_state_dict"])))
        elif mode == "shape":
            payload["age_probe_state_dict"]["weight"] = torch.zeros(1, 3)
        else:
            key = next(iter(payload["backbone_state_dict"]))
            payload["backbone_state_dict"][key] = payload["backbone_state_dict"][key].double()
        with pytest.raises(ZeroShotValidationError, match="state_dict mismatch"):
            validate_checkpoint_payload(payload, model=model, contract=tiny_contract(model))


def test_vocabulary_and_mask_compatibility_fail_closed():
    report = {
        "electrode_vocabulary_sha256": VOCAB_SHA,
        "channel_canonicalization_sha256": CANONICAL_SHA,
        "token_mask_semantics_version": MASK_SEMANTICS_VERSION,
    }
    valid = {
        "electrode_vocabulary_sha256": VOCAB_SHA,
        "channel_canonicalization_sha256": CANONICAL_SHA,
        "token_mask_semantics_version": MASK_SEMANTICS_VERSION,
    }
    assert validate_input_compatibility(checkpoint_report=report, input_contract=valid)["dortmund_source_channel_order_used"] is False
    for key in valid:
        bad = dict(valid); bad[key] = "wrong"
        with pytest.raises(ZeroShotValidationError):
            validate_input_compatibility(checkpoint_report=report, input_contract=bad)


class LabelBlindSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_count = 0
        self.seen_keys = None

    def forward(self, token_inputs, token_channel_indices, token_time_indices, token_valid_mask):
        self.forward_count += 1
        self.seen_keys = {"token_inputs", "token_channel_indices", "token_time_indices", "token_valid_mask"}
        return token_inputs[:, 0, 0:1] * 0 + torch.arange(token_inputs.shape[0], device=token_inputs.device).float().unsqueeze(1) / 10


def artificial_samples():
    samples = []
    ages = {"SUB1": 22.0, "SUB2": 23.0, "SUB3": 24.0, "SUB4": 25.0, "SUB5": None, "SUB6": 27.0, "SUB7": 29.0, "SUB8": 31.0}
    for subject, age in ages.items():
        number = int(subject[3:])
        sample = {
            "sample_id": f"{subject}_w0", "subject_id": subject,
            "recording_id": f"{subject}_rec", "window_index": 0,
            "true_age_years": age,
            "token_inputs": torch.full((640, 100), float(number), dtype=torch.float32),
            "token_channel_indices": torch.arange(64).repeat_interleave(10).long(),
            "token_time_indices": torch.arange(10).repeat(64).long(),
            "token_valid_mask": torch.ones(640, dtype=torch.float32),
        }
        samples.append(sample)
    return samples


def test_label_blind_all_subject_inference_sub5_exclusion_and_transform():
    batch = label_blind_collate(artificial_samples())
    spy = LabelBlindSpy()
    rows = infer_label_blind(spy, [batch], device=torch.device("cpu"), source_mean=SOURCE_AGE_MEAN, source_std=SOURCE_AGE_STD, checkpoint_sha256="x" * 64)
    assert spy.forward_count == 1
    assert spy.seen_keys == {"token_inputs", "token_channel_indices", "token_time_indices", "token_valid_mask"}
    assert len(rows) == 8
    sub5 = next(row for row in rows if row["subject_id"] == "SUB5")
    assert sub5["predicted_age_years"] is not None
    assert sub5["metrics_eligible"] is False
    assert sub5["true_age_years"] is sub5["signed_error_years"] is None
    assert rows[3]["predicted_age_years"] == pytest.approx(rows[3]["predicted_z"] * SOURCE_AGE_STD + SOURCE_AGE_MEAN)
    bundle = compute_metric_bundle(rows)
    assert len(bundle["subject_rows_all"]) == 8
    assert len(bundle["subject_rows_labeled"]) == 7
    assert all(row["subject_id"] != "SUB5" for row in bundle["subject_rows_labeled"])
    assert bundle["source_mean_baseline_metrics"]["baseline_prediction"] == 27.5
    assert bundle["source_mean_baseline_metrics"]["source"] == "dortmund_train_subject_mean"


def test_subject_mean_median_no_clip_or_calibration():
    rows = []
    for subject, age, values in (("SUB1", 22.0, [-3.0, 2.0]), ("SUB2", 31.0, [0.0, 1.0])):
        for index, z in enumerate(values):
            rows.append({
                "sample_id": f"{subject}_{index}", "subject_id": subject,
                "recording_id": "r", "window_index": index,
                "true_age_years": age, "predicted_z": z,
                "predicted_age_years": z * SOURCE_AGE_STD + SOURCE_AGE_MEAN,
                "num_valid_tokens": 640, "checkpoint_sha256": "x",
            })
    rows = finalize_prediction_rows(rows, source_mean=SOURCE_AGE_MEAN, source_std=SOURCE_AGE_STD)
    bundle = compute_metric_bundle(rows)
    subject = bundle["subject_rows_all"][0]
    assert subject["mean_predicted_age"] == pytest.approx(sum(row["predicted_age_years"] for row in rows[:2]) / 2)
    assert min(row["predicted_age_years"] for row in rows) < 22.0  # proves no Huada-range clipping


def test_coverage_and_nonfinite_fail_closed():
    samples = artificial_samples(); batch = label_blind_collate(samples)
    rows = infer_label_blind(LabelBlindSpy(), [batch], device=torch.device("cpu"), source_mean=SOURCE_AGE_MEAN, source_std=SOURCE_AGE_STD, checkpoint_sha256="x")
    assert validate_coverage(rows, [sample["sample_id"] for sample in samples])["subject_count"] == 8
    with pytest.raises(ZeroShotValidationError, match="Coverage"):
        validate_coverage(rows[:-1], [sample["sample_id"] for sample in samples])
    bad = artificial_samples(); bad[0]["token_inputs"][0, 0] = float("nan")
    with pytest.raises(ZeroShotValidationError, match="Non-finite"):
        infer_label_blind(LabelBlindSpy(), [label_blind_collate(bad)], device=torch.device("cpu"), source_mean=SOURCE_AGE_MEAN, source_std=SOURCE_AGE_STD, checkpoint_sha256="x")


def test_metric_independent_csv_recomputation(tmp_path):
    samples = artificial_samples(); rows = infer_label_blind(LabelBlindSpy(), [label_blind_collate(samples)], device=torch.device("cpu"), source_mean=SOURCE_AGE_MEAN, source_std=SOURCE_AGE_STD, checkpoint_sha256="x")
    path = tmp_path / "window_predictions.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDS, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in WINDOW_FIELDS})
    recomputed = recompute_from_csv(path, SOURCE_AGE_MEAN, SOURCE_AGE_STD, [sample["sample_id"] for sample in samples])
    assert recomputed["row_count"] == 8
    assert recomputed["z_to_years_recomputed"] is True
    assert recomputed["sub5_excluded_from_metrics"] is True
    assert recomputed["coverage"]["subject_count"] == 8


def test_dry_run_does_not_touch_paths_checkpoint_model_or_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / "must_not_exist"
    import scripts.downstream_50m.run_huada_dortmund_age_zero_shot as entry
    monkeypatch.setattr(entry, "load_strict_checkpoint", lambda *a, **k: (_ for _ in ()).throw(AssertionError("checkpoint touched")))
    monkeypatch.setattr(entry, "HuadaAllSubjectDataset", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dataset touched")))
    result = zero_shot_main([
        "--input-lance", "NONEXISTENT_REAL_LANCE", "--checkpoint", "NONEXISTENT_REAL_CHECKPOINT",
        "--expected-checkpoint-sha256", "3e44d9e7942d20d3d004e191a077ac636eeea770a6d185149591b683c5b3f13e",
        "--output-dir", str(output), "--dry-run",
    ])
    assert result == 0 and not output.exists()
    printed = capsys.readouterr().out
    assert '"will_open_lance": false' in printed
    assert '"will_read_checkpoint": false' in printed
    assert '"will_execute_forward": false' in printed
