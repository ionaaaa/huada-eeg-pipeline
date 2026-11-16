from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = ROOT / "scripts" / "downstream_50m"
SCRIPTS = ROOT / "scripts"
for value in (str(DOWNSTREAM), str(SCRIPTS)):
    if value not in sys.path:
        sys.path.insert(0, value)

from age_joint_training import (  # noqa: E402
    MultiSourceAgeDataset,
    joint_metric_bundle,
    validate_frozen_joint_contract,
)
from dortmund_age_finetune import SubjectBalancedDistributedSampler  # noqa: E402
import dortmund_age_finetune as age_core  # noqa: E402
from fbmae_training_profiles import (  # noqa: E402
    ProfiledTokenDataset,
    load_training_profiles,
    resolve_training_profile,
)
from run_dortmund_mipdb_age_joint_finetune import validate_joint_cli  # noqa: E402


class SyntheticTokens(Dataset):
    def __init__(self, dataset_id: str, subjects: list[str], windows: int = 2):
        self.dataset_id = dataset_id
        self.rows = []
        for subject_index, subject in enumerate(subjects):
            for window in range(windows):
                self.rows.append({
                    "index": len(self.rows),
                    "sample_id": f"{subject}_w{window}",
                    "subject_id": subject,
                    "recording_id": f"{subject}_recording",
                    "eyes_condition": "eyes_open" if window % 2 == 0 else "eyes_closed",
                    "block_id": "block01" if dataset_id == "mipdb" else "",
                    "age": 20.0 + subject_index,
                    "sex": "",
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "subject_id": row["subject_id"],
            "recording_id": row["recording_id"],
            "block_id": row["block_id"],
            "token_inputs": torch.ones(640, 100),
            "targets": torch.ones(640, 8, 4),
            "token_channel_indices": torch.arange(64).repeat_interleave(10),
            "token_time_indices": torch.arange(10).repeat(64),
            "token_valid_mask": torch.ones(640),
            "channel_valid_mask": torch.ones(64),
            "label": torch.tensor(row["age"]),
            "sex": "",
            "eyes_condition": row["eyes_condition"],
        }

    def get_sampling_rows(self):
        return [dict(row) for row in self.rows]


@pytest.fixture(scope="module")
def catalog():
    return load_training_profiles(ROOT / "configs" / "mipdb_dortmund_fbmae_training_profiles_v1.json")


def build_pair(catalog):
    dortmund = SyntheticTokens("dortmund", ["D1", "D2"], windows=2)
    mipdb = SyntheticTokens("mipdb", ["M1"], windows=2)
    b = MultiSourceAgeDataset({
        "dortmund": ProfiledTokenDataset(
            dortmund, "dortmund", resolve_training_profile(catalog, "joint_matched18", "dortmund")
        ),
        "mipdb": ProfiledTokenDataset(
            mipdb, "mipdb", resolve_training_profile(catalog, "joint_matched18", "mipdb")
        ),
    })
    c = MultiSourceAgeDataset({
        "dortmund": ProfiledTokenDataset(
            dortmund, "dortmund", resolve_training_profile(catalog, "joint_native64_vs_sparse18", "dortmund")
        ),
        "mipdb": ProfiledTokenDataset(
            mipdb, "mipdb", resolve_training_profile(catalog, "joint_native64_vs_sparse18", "mipdb")
        ),
    })
    return b, c


def test_global_sample_ids_are_namespaced_and_unique(catalog):
    b, _ = build_pair(catalog)
    ids = [row["sample_id"] for row in b.get_sampling_rows()]
    assert len(ids) == len(set(ids))
    assert all(value.startswith(("dortmund::", "mipdb::")) for value in ids)
    assert resolve_training_profile(catalog, "joint_matched18", "mipdb")["dataset_filter"]["block_id"] == "block01"


def test_b_c_have_identical_preprofile_sampling_schedule(catalog):
    b, c = build_pair(catalog)
    left = SubjectBalancedDistributedSampler(
        b.get_sampling_rows(), samples_per_epoch=60, seed=42, per_device_batch_size=2
    )
    right = SubjectBalancedDistributedSampler(
        c.get_sampling_rows(), samples_per_epoch=60, seed=42, per_device_batch_size=2
    )
    assert left.global_indices() == right.global_indices()
    assert left.statistics()["sample_id_sequence_sha256"] == right.statistics()["sample_id_sequence_sha256"]
    assert left.statistics()["sampling_schedule_sha256"] == right.statistics()["sampling_schedule_sha256"]
    assert left.statistics()["dataset_count"] == right.statistics()["dataset_count"]


def test_mipdb_is_bitwise_identical_b_c(catalog):
    b, c = build_pair(catalog)
    mipdb_index = next(row["index"] for row in b.get_sampling_rows() if row["dataset_id"] == "mipdb")
    for field in ("token_inputs", "token_valid_mask", "targets", "channel_valid_mask"):
        assert torch.equal(b[mipdb_index][field], c[mipdb_index][field])


def test_only_dortmund_profile_differs_and_invalid_values_are_zero(catalog):
    b, c = build_pair(catalog)
    b_item, c_item = b[0], c[0]
    assert int(b_item["token_valid_mask"].sum()) == 180
    assert int(c_item["token_valid_mask"].sum()) == 640
    invalid = ~b_item["token_valid_mask"].bool()
    assert torch.all(b_item["token_inputs"][invalid] == 0)
    assert torch.all(b_item["targets"][invalid] == 0)
    assert torch.all(c_item["token_inputs"] == 1)


def test_subject_balanced_sampler_has_natural_mipdb_share():
    rows = []
    for dataset_id, count in (("dortmund", 172), ("mipdb", 15)):
        for subject in range(count):
            rows.append({
                "index": len(rows), "dataset_id": dataset_id,
                "subject_id": str(subject), "sampling_subject_id": f"{dataset_id}::{subject}",
                "sample_id": f"{dataset_id}::{subject}", "recording_id": "r",
                "eyes_condition": "unknown",
            })
    sampler = SubjectBalancedDistributedSampler(
        rows, samples_per_epoch=6200, seed=42, per_device_batch_size=4
    )
    counts = sampler.statistics()["dataset_count"]
    assert counts["mipdb"] / sum(counts.values()) == pytest.approx(15 / 187, abs=0.002)
    assert sampler.statistics()["subject_count"] == 187


def test_joint_metrics_are_dataset_macro_not_window_pooled():
    rows = []
    for dataset_id, prediction, copies in (("dortmund", 21.0, 20), ("mipdb", 30.0, 1)):
        for index in range(copies):
            rows.append({
                "dataset_id": dataset_id, "sample_id": f"{dataset_id}-{index}",
                "subject_id": f"{dataset_id}-subject", "sex": "", "eyes_condition": "",
                "true_age": 20.0, "predicted_age": prediction,
            })
    metrics = joint_metric_bundle(rows)
    assert metrics["datasets"]["dortmund"]["subject_mean"]["mae"] == 1.0
    assert metrics["datasets"]["mipdb"]["subject_mean"]["mae"] == 10.0
    assert metrics["joint_macro"]["subject_mean_mae"] == 5.5


def test_frozen_joint_contract_and_subject_equal_scaler():
    rows, transform, _ = validate_frozen_joint_contract(
        joint_manifest_path=ROOT / "configs" / "joint_subject_split_manifest_v1.csv",
        target_transform_path=ROOT / "configs" / "joint_target_transform_v1.json",
        contract_path=ROOT / "configs" / "joint_age_training_contract_v1.json",
    )
    assert len(rows) == 233
    assert all(str(row["subject_id"]).startswith("sub-A") for row in rows if row["dataset_id"] == "mipdb")
    assert transform.train_subject_count == 187
    assert transform.equal_subject_weight and not transform.window_weighted and transform.ddof == 0
    assert transform.source_age_mean == 27.459893048128343
    assert transform.source_age_std == 4.20621465212978


def test_frozen_contract_fails_closed_on_tamper(tmp_path):
    for name in (
        "joint_subject_split_manifest_v1.csv", "joint_target_transform_v1.json",
        "joint_age_training_contract_v1.json",
    ):
        shutil.copyfile(ROOT / "configs" / name, tmp_path / name)
    path = tmp_path / "joint_target_transform_v1.json"
    payload = json.loads(path.read_text())
    payload["source_age_mean"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA mismatch"):
        validate_frozen_joint_contract(
            joint_manifest_path=tmp_path / "joint_subject_split_manifest_v1.csv",
            target_transform_path=path,
            contract_path=tmp_path / "joint_age_training_contract_v1.json",
        )


def test_frozen_contract_accepts_lf_crlf_transport_change(tmp_path):
    for name in (
        "joint_subject_split_manifest_v1.csv", "joint_target_transform_v1.json",
        "joint_age_training_contract_v1.json",
    ):
        source = ROOT / "configs" / name
        text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
        with (tmp_path / name).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    rows, transform, contract = validate_frozen_joint_contract(
        joint_manifest_path=tmp_path / "joint_subject_split_manifest_v1.csv",
        target_transform_path=tmp_path / "joint_target_transform_v1.json",
        contract_path=tmp_path / "joint_age_training_contract_v1.json",
    )
    assert len(rows) == 233 and transform.train_subject_count == 187
    assert contract["validation"]["fingerprint_policy"] == "canonical_content_v1"


def test_distributed_cuda_topology_fails_before_set_device(monkeypatch):
    monkeypatch.setattr(age_core, "distributed_context", lambda: {
        "distributed_enabled": True, "rank": 1, "local_rank": 1, "world_size": 4,
    })
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="Distributed CUDA topology mismatch"):
        age_core.initialize_distributed("cuda")


def test_cloud_entry_preflights_visible_gpu_count():
    text = (ROOT / "commands" / "run_dortmund_mipdb_age_joint_finetune_full.sh").read_text()
    assert "torch.cuda.device_count()" in text
    assert "NPROC_PER_NODE > VISIBLE_GPU_COUNT" in text


@pytest.mark.parametrize("profile", ["joint_matched18", "joint_native64_vs_sparse18"])
def test_joint_entry_accepts_only_b_c(profile):
    validate_joint_cli(["--training-profile", profile])


def test_joint_entry_rejects_single_dataset_profile():
    with pytest.raises(ValueError, match="accepts only"):
        validate_joint_cli(["--training-profile", "dortmund_native64"])
