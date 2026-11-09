"""Native full-recording preprocessing/cache for Huada LaBraM and REVE."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .contracts import HUADA_CHANNEL_AXIS, canonical_json, contract_sha256


class NativePreprocessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessedRecording:
    data: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NativeWindow:
    window_index: int
    start_sample: int
    stop_sample: int
    data: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_notch_harmonics(
    base_hz: float, *, source_sfreq: float, bandpass_high_hz: float
) -> list[float]:
    if base_hz <= 0 or source_sfreq <= 0 or bandpass_high_hz <= 0:
        raise ValueError("frequencies must be positive")
    ceiling = min(float(bandpass_high_hz), float(source_sfreq) / 2.0)
    values: list[float] = []
    harmonic = float(base_hz)
    while harmonic < ceiling:
        values.append(harmonic)
        harmonic += float(base_hz)
    return values


def _filter_and_resample(
    data: np.ndarray,
    *,
    source_sfreq: float,
    low_hz: float,
    high_hz: float,
    notch_hz: list[float] | None,
    target_sfreq: float,
) -> np.ndarray:
    """MNE implementation; filtering order is part of the frozen contract."""

    from mne.filter import filter_data, notch_filter, resample

    filtered = filter_data(
        data.astype(np.float64, copy=True),
        sfreq=float(source_sfreq),
        l_freq=float(low_hz),
        h_freq=float(high_hz),
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    if notch_hz:
        filtered = notch_filter(
            filtered,
            Fs=float(source_sfreq),
            freqs=np.asarray(notch_hz, dtype=float),
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose="ERROR",
        )
    return resample(
        filtered,
        down=float(source_sfreq),
        up=float(target_sfreq),
        axis=-1,
        method="polyphase",
        pad="reflect",
        verbose="ERROR",
    )


def _validate_input(data_v: np.ndarray, source_sfreq: float) -> np.ndarray:
    data = np.asarray(data_v)
    if data.ndim != 2 or data.shape[0] != len(HUADA_CHANNEL_AXIS):
        raise NativePreprocessingError(
            f"expected Huada recording [64,T], got {data.shape}"
        )
    if float(source_sfreq) != 500.0:
        raise NativePreprocessingError(
            f"expected audited Huada source sampling rate 500 Hz, got {source_sfreq}"
        )
    if not np.issubdtype(data.dtype, np.floating) or not np.isfinite(data).all():
        raise NativePreprocessingError("raw MNE-volts recording must be finite floating data")
    return np.asarray(data, dtype=np.float64)


def _standard_scaler_per_channel(data: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """scikit-learn StandardScaler semantics, independently per channel."""

    means = data.mean(axis=1, dtype=np.float64)
    variances = data.var(axis=1, dtype=np.float64, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(variances).all():
        raise NativePreprocessingError("non-finite REVE scaler statistics")
    scales = np.sqrt(variances)
    constant = scales == 0.0
    scales[constant] = 1.0
    standardized = (data - means[:, None]) / scales[:, None]
    return standardized, {
        "mean": means.tolist(),
        "variance": variances.tolist(),
        "scale": scales.tolist(),
        "constant_channel_indices": np.flatnonzero(constant).astype(int).tolist(),
        "ddof": 0,
        "statistics_sample_count": int(data.shape[1]),
    }


def preprocess_native_recording(
    data_v: np.ndarray,
    *,
    family: str,
    source_sfreq: float = 500.0,
    filter_resample=_filter_and_resample,
) -> ProcessedRecording:
    """Process one complete Huada recording before any 2-second slicing."""

    data = _validate_input(data_v, source_sfreq)
    if family == "labram":
        notches = valid_notch_harmonics(
            50.0, source_sfreq=source_sfreq, bandpass_high_hz=75.0
        )
        processed = filter_resample(
            data,
            source_sfreq=source_sfreq,
            low_hz=0.1,
            high_hz=75.0,
            notch_hz=notches,
            target_sfreq=200.0,
        )
        processed = processed * 10000.0
        metadata = {
            "family": family,
            "processing_order": ["bandpass", "notch", "resample", "extractor_scale"],
            "bandpass_hz": [0.1, 75.0],
            "notch_hz": notches,
            "extractor_scale": 10000.0,
            "scaler": None,
        }
    elif family == "reve":
        processed = filter_resample(
            data,
            source_sfreq=source_sfreq,
            low_hz=0.5,
            high_hz=99.5,
            notch_hz=None,
            target_sfreq=200.0,
        )
        processed, scaler = _standard_scaler_per_channel(processed)
        processed = np.clip(processed, -15.0, 15.0)
        metadata = {
            "family": family,
            "processing_order": ["bandpass", "resample", "standardize", "clip"],
            "bandpass_hz": [0.5, 99.5],
            "notch_hz": None,
            "extractor_scale": None,
            "clip_range": [-15.0, 15.0],
            "scaler": scaler,
        }
    else:
        raise NativePreprocessingError(f"unsupported native family: {family!r}")
    output = np.ascontiguousarray(processed, dtype=np.float32)
    if output.ndim != 2 or output.shape[0] != 64 or not np.isfinite(output).all():
        raise NativePreprocessingError("native preprocessing produced invalid output")
    metadata.update(
        {
            "source_sampling_rate_hz": 500.0,
            "target_sampling_rate_hz": 200.0,
            "source_data_unit": "MNE_volts",
            "cache_scope": "full_recording",
            "shape": list(output.shape),
            "dtype": "float32",
            "channel_axis": list(HUADA_CHANNEL_AXIS),
        }
    )
    return ProcessedRecording(output, metadata)


def iter_native_windows(
    recording: np.ndarray, *, window_samples: int = 400, hop_samples: int = 400
) -> Iterator[NativeWindow]:
    data = np.asarray(recording)
    if data.ndim != 2 or data.shape[0] != 64 or data.dtype != np.float32:
        raise NativePreprocessingError("cached recording must be float32 [64,T]")
    if window_samples != 400 or hop_samples != 400:
        raise NativePreprocessingError("v1 native backend requires non-overlap 2 s windows")
    for index, start in enumerate(range(0, data.shape[1] - window_samples + 1, hop_samples)):
        stop = start + window_samples
        yield NativeWindow(index, start, stop, np.ascontiguousarray(data[:, start:stop]))


class NativeRecordingCache:
    """Safe mmap-friendly full-recording cache: .npy + JSON + _SUCCESS."""

    DATA_FILE = "recording.float32.npy"
    METADATA_FILE = "metadata.json"

    @classmethod
    def write(
        cls,
        path: str | Path,
        recording: ProcessedRecording,
        *,
        contract: dict[str, Any],
        recording_id: str,
    ) -> Path:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite native cache: {target}")
        if not target.parent.is_dir():
            raise FileNotFoundError(f"cache parent directory does not exist: {target.parent}")
        data = np.asarray(recording.data)
        if data.dtype != np.float32 or data.ndim != 2 or data.shape[0] != 64:
            raise NativePreprocessingError("cache data must be float32 [64,T]")
        temporary = target.with_name(f".{target.name}.tmp.{uuid.uuid4().hex}")
        temporary.mkdir()
        try:
            data_path = temporary / cls.DATA_FILE
            np.save(data_path, data, allow_pickle=False)
            metadata = {
                **recording.metadata,
                "recording_id": str(recording_id),
                "target_contract_sha256": contract_sha256(contract),
                "data_file": cls.DATA_FILE,
                "data_file_sha256": _sha256_file(data_path),
                "fbmae_token_lance_reused": False,
            }
            (temporary / cls.METADATA_FILE).write_text(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            # Read back without pickle before publication.
            check = np.load(data_path, mmap_mode="r", allow_pickle=False)
            if check.dtype != np.float32 or check.shape != data.shape or not np.isfinite(check).all():
                raise NativePreprocessingError("native cache round-trip validation failed")
            # Windows does not allow renaming a directory while its mmap is open.
            del check
            (temporary / "_SUCCESS").write_text(
                canonical_json({"recording_id": str(recording_id), "data_file_sha256": metadata["data_file_sha256"]})
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_family: str,
        expected_contract_sha256: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        root = Path(path)
        if root.is_symlink() or not root.is_dir() or not (root / "_SUCCESS").is_file():
            raise NativePreprocessingError(f"native cache is incomplete: {root}")
        metadata = json.loads((root / cls.METADATA_FILE).read_text(encoding="utf-8"))
        if metadata.get("family") != expected_family:
            raise NativePreprocessingError("native cache family mismatch")
        if metadata.get("target_contract_sha256") != expected_contract_sha256:
            raise NativePreprocessingError("native cache target contract mismatch")
        if metadata.get("fbmae_token_lance_reused") is not False:
            raise NativePreprocessingError("native cache claims FBMAE token reuse")
        data_path = root / cls.DATA_FILE
        if _sha256_file(data_path) != metadata.get("data_file_sha256"):
            raise NativePreprocessingError("native cache data SHA mismatch")
        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if data.dtype != np.float32 or data.ndim != 2 or data.shape[0] != 64:
            raise NativePreprocessingError("native cache data contract mismatch")
        return data, metadata


def preprocess_bdf_to_cache(
    source_bdf: str | Path,
    output_dir: str | Path,
    *,
    family: str,
    contract: dict[str, Any],
    recording_id: str,
) -> Path:
    """Read one BDF as MNE volts and publish one full-recording native cache."""

    source = Path(source_bdf)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"BDF is not a regular non-symlink file: {source}")
    import mne

    raw = mne.io.read_raw_bdf(source, preload=True, verbose="ERROR")
    if list(raw.ch_names) != list(HUADA_CHANNEL_AXIS):
        raise NativePreprocessingError("BDF channel names/order differ from Huada contract")
    if float(raw.info["sfreq"]) != 500.0:
        raise NativePreprocessingError("BDF sampling rate differs from Huada contract")
    processed = preprocess_native_recording(raw.get_data(), family=family, source_sfreq=500.0)
    return NativeRecordingCache.write(
        output_dir,
        processed,
        contract=contract,
        recording_id=recording_id,
    )
