"""Lance dataset port with a thin Huada task-label adapter."""

from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path

import lance
import numpy as np
import torch
from torch.utils.data import Dataset


SEX_LABEL_MAPPING = {"female": 0, "male": 1}
SUPPORTED_TASKS = {"sex_classification", "age_regression"}


def load_storage_options(storage_options_path):
    if not storage_options_path:
        return None
    with open(storage_options_path, "r", encoding="utf-8") as handle:
        return {key: str(value) for key, value in json.load(handle).items()}


def load_lance_part_paths(list_file):
    with open(list_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_subject_manifest(path):
    if not path:
        return {}
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            subject_id = str(row.get("subject_id", "")).strip()
            if not subject_id:
                raise ValueError(f"Manifest row has no subject_id: {row}")
            if subject_id in rows:
                raise ValueError(f"Duplicate subject_id in manifest: {subject_id}")
            rows[subject_id] = row
    return rows


class LanceEEGPretrainDataset(Dataset):
    """Read the shared token Lance without copying token tensors.

    This keeps the upstream tensor conversion and batched ``take`` behavior,
    while deriving ``label`` from the existing ``sex`` or ``age`` column.
    """

    TENSOR_COLUMNS = [
        "num_tokens", "patch_len", "n_bands", "frames_per_patch", "n_channels",
        "token_inputs", "targets", "token_channel_indices", "token_time_indices",
        "token_valid_mask", "channel_valid_mask", "sample_id", "subject_id",
        "recording_id", "age", "sex", "source_sfreq", "model_sfreq",
    ]
    FULL_COLUMNS = TENSOR_COLUMNS
    COLUMNS = FULL_COLUMNS
    CLASSIFICATION_COLUMNS = TENSOR_COLUMNS

    def __init__(
        self,
        lance_path,
        storage_options=None,
        columns=None,
        task="sex_classification",
        subject_split_manifest=None,
        split=None,
        max_samples=None,
    ):
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task={task!r}; expected {sorted(SUPPORTED_TASKS)}")
        self.lance_path = str(lance_path)
        self.storage_options = storage_options
        self.columns = list(columns) if columns is not None else list(self.FULL_COLUMNS)
        self.task = task
        self.split = split
        self.manifest = load_subject_manifest(subject_split_manifest)
        self.ds = None
        self._metadata = self._read_metadata()
        self.indices = self._select_indices(max_samples=max_samples)
        self.length = len(self.indices)

    def _open(self):
        if self.ds is None:
            kwargs = {} if self.storage_options is None else {"storage_options": self.storage_options}
            self.ds = lance.dataset(self.lance_path, **kwargs)
        return self.ds

    def _read_metadata(self):
        ds = self._open()
        required = set(self.TENSOR_COLUMNS)
        missing = sorted(required.difference(ds.schema.names))
        if missing:
            raise ValueError(f"Input Lance is missing required fields: {missing}")
        return ds.to_table(columns=["sample_id", "subject_id", "age", "sex"]).to_pylist()

    @staticmethod
    def _normalize_subject_id(value):
        if value is None:
            return ""
        if hasattr(value, "as_py"):
            value = value.as_py()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)

    def _label_from_row(self, row):
        if self.task == "sex_classification":
            sex = str(row.get("sex", "")).strip().lower()
            if sex not in SEX_LABEL_MAPPING:
                raise ValueError(f"Unknown sex label {sex!r} for {row.get('sample_id')}")
            return SEX_LABEL_MAPPING[sex], sex
        age = row.get("age")
        if age is None:
            return None, ""
        return float(age), ""

    def _select_indices(self, max_samples=None):
        selected = []
        subject_labels = {}
        manifest_subjects_seen = set()
        for index, row in enumerate(self._metadata):
            subject_id = self._normalize_subject_id(row["subject_id"])
            label, label_name = self._label_from_row(row)
            manifest_row = self.manifest.get(subject_id)
            if self.manifest:
                if manifest_row is None:
                    continue
                manifest_subjects_seen.add(subject_id)
                manifest_task = str(manifest_row.get("task", "")).strip()
                if manifest_task and manifest_task != self.task:
                    raise ValueError(
                        f"Manifest task mismatch for {subject_id}: {manifest_task} != {self.task}"
                    )
                if str(manifest_row.get("exclusion_reason", "")).strip():
                    continue
                if self.split is not None and str(manifest_row.get("split", "")).strip() != self.split:
                    continue
                expected = str(manifest_row.get("label", "")).strip()
                actual = "" if label is None else str(int(label) if self.task == "sex_classification" else float(label))
                if expected and expected != actual:
                    raise ValueError(
                        f"Manifest/cache label mismatch for {subject_id}: {expected!r} != {actual!r}"
                    )
                expected_name = str(manifest_row.get("label_name", "")).strip().lower()
                if expected_name and expected_name != label_name:
                    raise ValueError(
                        f"Manifest/cache label_name mismatch for {subject_id}: {expected_name!r} != {label_name!r}"
                    )
            if label is None:  # Missing age is excluded; it is never imputed.
                continue
            previous = subject_labels.setdefault(subject_id, label)
            if previous != label:
                raise ValueError(f"Inconsistent labels within subject {subject_id}")
            selected.append(index)
            if max_samples is not None and len(selected) >= int(max_samples):
                break
        if self.manifest:
            missing_subjects = {
                subject for subject, row in self.manifest.items()
                if not str(row.get("exclusion_reason", "")).strip()
                and (self.split is None or str(row.get("split", "")).strip() == self.split)
            }.difference(manifest_subjects_seen)
            if missing_subjects:
                raise ValueError(f"Manifest subjects absent from input Lance: {sorted(missing_subjects)}")
        return selected

    def __len__(self):
        return self.length

    def _row_to_sample(self, row):
        num_tokens = int(row["num_tokens"])
        patch_len = int(row["patch_len"])
        n_bands = int(row["n_bands"])
        frames_per_patch = int(row["frames_per_patch"])
        n_channels = int(row["n_channels"])
        label, label_name = self._label_from_row(row)
        if label is None:
            raise AssertionError("Missing-age row passed dataset filtering")
        sample = {
            "sample_id": str(row["sample_id"]),
            "subject_id": self._normalize_subject_id(row["subject_id"]),
            "recording_id": str(row["recording_id"]),
            "token_inputs": torch.from_numpy(np.asarray(row["token_inputs"], dtype=np.float32).reshape(num_tokens, patch_len)),
            "targets": torch.from_numpy(np.asarray(row["targets"], dtype=np.float32).reshape(num_tokens, n_bands, frames_per_patch)),
            "token_channel_indices": torch.from_numpy(np.asarray(row["token_channel_indices"], dtype=np.int64)),
            "token_time_indices": torch.from_numpy(np.asarray(row["token_time_indices"], dtype=np.int64)),
            "token_valid_mask": torch.from_numpy(np.asarray(row["token_valid_mask"], dtype=np.float32)),
            "channel_valid_mask": torch.from_numpy(np.asarray(row["channel_valid_mask"], dtype=np.float32)),
            "source_sfreq": torch.tensor(float(row["source_sfreq"]), dtype=torch.float32),
            "model_sfreq": torch.tensor(float(row["model_sfreq"]), dtype=torch.float32),
            "label_name": label_name,
        }
        if sample["channel_valid_mask"].shape != (n_channels,):
            raise ValueError(f"channel_valid_mask shape mismatch for {sample['sample_id']}")
        if sample["token_valid_mask"].shape != (num_tokens,):
            raise ValueError(f"token_valid_mask shape mismatch for {sample['sample_id']}")
        if self.task == "sex_classification":
            sample["label"] = torch.tensor(int(label), dtype=torch.int64)
        else:
            sample["label"] = torch.tensor(float(label), dtype=torch.float32)
        return sample

    def __getitem__(self, idx):
        global_index = self.indices[int(idx)]
        row = self._open().take(indices=[global_index], columns=self.columns).to_pylist()[0]
        return self._row_to_sample(row)

    def __getitems__(self, indices):
        if not indices:
            return []
        global_indices = [self.indices[int(index)] for index in indices]
        rows = self._open().take(indices=global_indices, columns=self.columns).to_pylist()
        return [self._row_to_sample(row) for row in rows]

    def get_subject_label_rows(self, indices=None, chunk_size=50000):
        del chunk_size
        local_indices = range(len(self)) if indices is None else [int(index) for index in indices]
        result = []
        for local_index in local_indices:
            row = self._metadata[self.indices[local_index]]
            label, _ = self._label_from_row(row)
            result.append({
                "index": local_index,
                "sample_id": str(row["sample_id"]),
                "subject_id": self._normalize_subject_id(row["subject_id"]),
                "label": label,
            })
        return result

    def get_label_rows(self, indices=None, chunk_size=50000):
        return [
            {"index": row["index"], "label": row["label"]}
            for row in self.get_subject_label_rows(indices, chunk_size)
        ]


class MultiLanceEEGPretrainDataset(Dataset):
    """Upstream-compatible multiple-Lance wrapper."""

    def __init__(self, lance_paths, storage_options=None, columns=None, **kwargs):
        if not lance_paths:
            raise ValueError("lance_paths is empty")
        self.datasets = [
            LanceEEGPretrainDataset(path, storage_options, columns, **kwargs)
            for path in lance_paths
        ]
        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _locate_index(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_index = bisect.bisect_right(self.cumulative_sizes, idx)
        previous = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return dataset_index, idx - previous

    def __getitem__(self, idx):
        dataset_index, sample_index = self._locate_index(int(idx))
        return self.datasets[dataset_index][sample_index]
