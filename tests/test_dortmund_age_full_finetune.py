from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
MODULES = ROOT / "scripts" / "downstream_50m"
if str(MODULES) not in sys.path:
    sys.path.insert(0, str(MODULES))

from dortmund_age_split import (  # noqa: E402
    SubjectMetadata,
    allocate_split_counts,
    build_subject_split,
    resolve_strata,
    validate_ratios,
    write_manifest,
)
from dortmund_age_finetune import (  # noqa: E402
    NoPaddingDistributedEvalSampler,
    StrictDortmundAgeModel,
    SubjectBalancedDistributedSampler,
    TestAccessGuard as AccessGuard,
    aggregate_subject_predictions,
    atomic_torch_save,
    audit_gradient_coverage,
    build_linear_warmup_cosine_scheduler,
    build_optimizer,
    build_target_transform,
    configure_probe_gradient_scope,
    ensure_finite,
    regression_metrics,
    train_mean_baseline,
    validate_prediction_coverage,
    validate_resume_payload,
    validation_is_better,
    zero_initialize_age_head,
)
from model import EEGPretrainModel  # noqa: E402
import run_dortmund_age_full_finetune as full_entry  # noqa: E402

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from fbmae_training_profiles import (  # noqa: E402
    apply_training_profile,
    load_training_profiles,
    resolve_training_profile,
)


def subjects(count=214):
    return [
        SubjectMetadata(
            subject_id=f"sub-{index:03d}",
            age=float(20 + index % 16),
            sex="female" if index % 2 == 0 else "male",
            conditions=("eyes-open", "eyes-closed"),
            recording_ids=(f"r{index}-eo", f"r{index}-ec"),
            window_count=4,
        )
        for index in range(count)
    ]


def tiny_model():
    backbone = EEGPretrainModel(
        input_dim=4, d_model=8, n_heads=2, depth=12, mlp_ratio=2,
        dropout=0, n_channels=4, n_time_patches=2, n_bands=2,
        frames_per_patch=2,
    )
    return StrictDortmundAgeModel(
        backbone=backbone, d_model=8, num_classes=1, probe_layer_idx=8,
        freeze_backbone=False, pooling="mean", num_tokens=8,
    )


def test_default_split_counts_are_172_21_21():
    assert allocate_split_counts(214, 0.8, 0.1, 0.1) == {
        "train": 172, "validation": 21, "test": 21,
    }


@pytest.mark.parametrize("ratios", [(0.0, 0.5, 0.5), (0.8, 0.1, 0.11)])
def test_ratio_validation_rejects_invalid(ratios):
    with pytest.raises(ValueError):
        validate_ratios(*ratios)


def test_configurable_split_is_reproducible_and_isolated():
    first, summary = build_subject_split(subjects(), split_seed=42)
    second, _ = build_subject_split(subjects(), split_seed=42)
    assert first == second
    assert [summary["splits"][name]["subject_count"] for name in ("train", "validation", "test")] == [172, 21, 21]
    assert len({row["subject_id"] for row in first}) == 214
    assert all(set(row["conditions"].split(";")) == {"eyes-open", "eyes-closed"} for row in first)


def test_manifest_is_byte_reproducible(tmp_path):
    rows, _ = build_subject_split(subjects(), split_seed=42)
    left, right = tmp_path / "a.csv", tmp_path / "b.csv"
    assert write_manifest(left, rows) == write_manifest(right, rows)
    assert left.read_bytes() == right.read_bytes()


def test_sparse_strata_fallback_is_explicit():
    sparse = [
        SubjectMetadata("a", 20, "female"),
        SubjectMetadata("b", 20, "male"),
        SubjectMetadata("c", 21, "female"),
        SubjectMetadata("d", 21, "male"),
    ]
    result = resolve_strata(sparse)
    assert {value["stratification_level"] for value in result.values()} == {"two_year_age_bin"}
    assert all("sparse" in value["stratification_reason"] for value in result.values())


def test_train_only_subject_equal_target_transform_and_roundtrip():
    rows = [
        {"subject_id": "a", "age": "20", "split": "train"},
        {"subject_id": "b", "age": "30", "split": "train"},
        {"subject_id": "c", "age": "99", "split": "validation"},
    ]
    transform = build_target_transform(rows, "manifest")
    assert transform.source_age_mean == 25
    assert transform.source_age_std == 5
    assert transform.equal_subject_weight and not transform.window_weighted and transform.ddof == 0
    ages = np.array([20.0, 25.0, 30.0])
    assert np.max(np.abs(transform.inverse_numpy(transform.transform_numpy(ages)) - ages)) < 1e-12


def test_profile_age_filter_is_stable_and_channel_choice_does_not_change_scaler():
    rows = [
        {"subject_id": f"s{age}-{split}", "age": str(age), "sex": "female", "split": split}
        for split in ("train", "validation", "test") for age in (20, 30, 35, 36, 44)
    ]
    filtered = full_entry._filter_manifest_rows(rows, 20, 35)
    assert [row["subject_id"] for row in filtered] == [
        row["subject_id"] for row in rows if float(row["age"]) <= 35
    ]
    assert all(20 <= float(row["age"]) <= 35 for row in filtered)
    fingerprint = full_entry._manifest_csv_sha256(filtered)
    native = build_target_transform(filtered, fingerprint)
    matched = build_target_transform(filtered, fingerprint)
    assert native == matched


def test_dortmund_profiles_keep_fixed_geometry_and_authoritative_invalid_tokens():
    catalog = load_training_profiles(
        ROOT / "configs" / "mipdb_dortmund_fbmae_training_profiles_v1.json"
    )
    native = resolve_training_profile(catalog, "dortmund_native64", "dortmund")
    matched = resolve_training_profile(catalog, "dortmund_matched18", "dortmund")
    inputs = np.ones((640, 100), dtype=np.float32)
    source_valid = np.ones(640, dtype=bool)
    source_valid[0] = False
    native_result = apply_training_profile(inputs, source_valid, "dortmund", native)
    matched_result = apply_training_profile(inputs, source_valid, "dortmund", matched)
    assert native_result["token_inputs"].shape == matched_result["token_inputs"].shape == (640, 100)
    assert native_result["token_valid_mask"].sum() == 639
    assert not native_result["token_valid_mask"][0]
    assert matched_result["token_valid_mask"].sum() == 179
    assert matched_result["channel_valid_mask"].sum() == 18
    assert np.all(matched_result["token_inputs"][matched_result["token_valid_mask"] == 0] == 0)


def sampler_rows():
    result = []
    index = 0
    for subject in ("a", "b", "c"):
        for condition in ("eyes-open", "eyes-closed"):
            for window in range(3):
                result.append({
                    "index": index, "sample_id": f"{subject}-{condition}-{window}",
                    "subject_id": subject, "eyes_condition": condition,
                })
                index += 1
    return result


def test_subject_balanced_sampler_is_deterministic_balanced_and_set_epoch():
    sampler = SubjectBalancedDistributedSampler(
        sampler_rows(), samples_per_epoch=18, seed=7, per_device_batch_size=2,
    )
    first = list(sampler)
    stats = sampler.statistics()
    assert max(stats["subject_counts"].values()) - min(stats["subject_counts"].values()) <= 1
    for subject in ("a", "b", "c"):
        eo = stats["subject_condition_counts"][f"{subject}|eyes_open"]
        ec = stats["subject_condition_counts"][f"{subject}|eyes_closed"]
        assert abs(eo - ec) <= 1
    same = SubjectBalancedDistributedSampler(sampler_rows(), samples_per_epoch=18, seed=7, per_device_batch_size=2)
    assert first == list(same)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_distributed_sampler_uses_one_global_sequence_and_equal_rank_lengths():
    samplers = [
        SubjectBalancedDistributedSampler(
            sampler_rows(), samples_per_epoch=17, seed=9, rank=rank,
            world_size=2, per_device_batch_size=2,
        )
        for rank in range(2)
    ]
    global_sequence = samplers[0].global_indices()
    assert list(samplers[0]) == global_sequence[0::2]
    assert list(samplers[1]) == global_sequence[1::2]
    assert len(samplers[0]) == len(samplers[1]) == 10
    assert samplers[0].statistics()["ddp_padding_duplicates"] == 0


def test_no_padding_evaluation_sampler_and_gather_coverage():
    shards = [list(NoPaddingDistributedEvalSampler(7, rank, 3)) for rank in range(3)]
    assert sorted(value for shard in shards for value in shard) == list(range(7))
    assert sum(map(len, shards)) == 7
    rows = [{"sample_id": str(index)} for index in range(7)]
    validate_prediction_coverage(rows, [str(index) for index in range(7)])
    with pytest.raises(ValueError):
        validate_prediction_coverage(rows + [rows[0]], [str(index) for index in range(7)])


def test_subject_metrics_aggregate_windows_not_rank_metrics():
    rows = [
        {"sample_id": "a1", "subject_id": "a", "sex": "female", "eyes_condition": "eyes-open", "true_age": 20, "predicted_age": 19},
        {"sample_id": "a2", "subject_id": "a", "sex": "female", "eyes_condition": "eyes-closed", "true_age": 20, "predicted_age": 21},
        {"sample_id": "b1", "subject_id": "b", "sex": "male", "eyes_condition": "eyes-open", "true_age": 30, "predicted_age": 28},
    ]
    aggregated = aggregate_subject_predictions(rows)
    assert [row["mean_predicted_age"] for row in aggregated] == [20, 28]
    assert regression_metrics([20, 30], [20, 28])["mae"] == 1
    baseline = train_mean_baseline(aggregated, 25)
    assert baseline["mae"] == 5 and baseline["baseline_prediction"] == 25


def test_undefined_correlations_are_json_safe():
    metrics = regression_metrics([20, 20], [19, 21])
    assert metrics["r2"] is None and not metrics["r2_defined"]
    assert metrics["pearson_r"]["value"] is None
    json.dumps(metrics, allow_nan=False)


def test_selection_tolerance_and_earlier_epoch():
    incumbent = {"subject_mae": 1, "subject_rmse": 2, "validation_loss": 3, "epoch": 2}
    almost = {"subject_mae": 1 + 1e-8, "subject_rmse": 2 + 1e-8, "validation_loss": 3 + 1e-8, "epoch": 3}
    assert not validation_is_better(almost, incumbent, tolerance=1e-6)
    rmse_win = {"subject_mae": 1 + 1e-8, "subject_rmse": 1.5, "validation_loss": 99, "epoch": 3}
    assert validation_is_better(rmse_win, incumbent, tolerance=1e-6)


def test_test_access_guard_fails_closed_and_allows_one_post_lock_test():
    guard = AccessGuard()
    with pytest.raises(RuntimeError):
        guard.mark_dataset_created()
    assert not guard.state()["test_dataset_created"]
    guard.lock_selection()
    guard.mark_dataset_created(); guard.mark_dataloader_created(); guard.mark_rows(2); guard.mark_forward(); guard.mark_metrics()
    with pytest.raises(RuntimeError):
        guard.mark_metrics()


def test_masked_mean_fails_on_all_invalid():
    model = tiny_model()
    with pytest.raises(ValueError, match="All-invalid"):
        model.pool_valid_tokens(torch.ones(1, 2, 8), torch.zeros(1, 2))


def test_intermediate_gradient_scope_freezes_layers_9_11_and_upstream_head():
    model = tiny_model()
    zero_initialize_age_head(model)
    report = configure_probe_gradient_scope(model, 8)
    assert report["executed_encoder_layers"] == list(range(9))
    assert report["explicitly_frozen_encoder_layers"] == [9, 10, 11]
    assert report["upstream_head_frozen"]
    assert all(not parameter.requires_grad for parameter in model.backbone.encoder.encoder.layers[9].parameters())
    assert all(not parameter.requires_grad for parameter in model.backbone.head.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.encoder.encoder.layers[8].parameters())


def test_gradient_coverage_optimizer_differential_lr_weight_decay_and_clipping():
    model = tiny_model()
    configure_probe_gradient_scope(model, 8)
    with torch.no_grad():
        model.head.weight.fill_(0.1)
    optimizer, report = build_optimizer(model, backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)
    assert {row["lr"] for row in report if row["owner"] == "backbone"} == {1e-5}
    assert {row["lr"] for row in report if row["owner"] == "age_head"} == {1e-4}
    assert any(row["weight_decay"] == 0.05 for row in report)
    assert any(row["weight_decay"] == 0 for row in report)
    output = model(
        torch.randn(2, 8, 4), torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]]).repeat(2, 1),
        torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]]).repeat(2, 1), torch.ones(2, 8),
    ).sum()
    output.backward()
    audit = audit_gradient_coverage(model)
    assert audit["unexplained_trainable_without_gradient"] == []
    assert not audit["nonfinite_gradient_parameters"]
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=True)
    assert torch.isfinite(norm)
    optimizer.step()


def test_nonfinite_fail_closed():
    with pytest.raises(FloatingPointError):
        ensure_finite("loss", torch.tensor(float("nan")))


def test_scheduler_and_atomic_best_last_checkpoint(tmp_path):
    model = tiny_model(); configure_probe_gradient_scope(model, 8)
    optimizer, _ = build_optimizer(model, backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)
    scheduler, config = build_linear_warmup_cosine_scheduler(optimizer, total_steps=10, warmup_ratio=0.1)
    assert config["warmup_steps"] == 1
    optimizer.step(); scheduler.step()
    for name in ("best_full_checkpoint.pt", "last_training_checkpoint.pt"):
        path = tmp_path / name
        atomic_torch_save({"kind": name, "state": model.state_dict()}, path)
        assert path.is_file() and not path.with_name(path.name + ".tmp").exists()


def test_resume_validation_checks_hashes_and_world_size():
    payload = {
        "source_token_lance_fingerprint": "token", "split_manifest_sha256": "manifest",
        "world_size": 2, "resolved_config_sha256": "config", "resume_granularity": "epoch_boundary",
    }
    validate_resume_payload(payload, token_fingerprint="token", manifest_sha256="manifest", world_size=2, resolved_config_sha256="config")
    with pytest.raises(ValueError):
        validate_resume_payload(payload, token_fingerprint="changed", manifest_sha256="manifest", world_size=2, resolved_config_sha256="config")


def test_command_contracts_and_formal_command_is_not_executed_by_tests():
    commands = ROOT / "commands"
    full = (commands / "run_dortmund_age_full_finetune_full.sh").read_text(encoding="utf-8")
    ddp = (commands / "run_dortmund_age_full_finetune_ddp_smoke.sh").read_text(encoding="utf-8")
    assert "torchrun --standalone" in full and '--nproc_per_node="${NPROC_PER_NODE}"' in full
    assert "--probe-layer-idx 8" in full and "--pooling mean" in full
    assert '--training-profile "${TRAINING_PROFILE}"' in full
    assert "--n-channels" not in full
    assert "--amp-mode bf16" in full and "--skip-test" not in full
    assert "--smoke --skip-test" in ddp
    smoke = (commands / "run_dortmund_age_full_finetune_smoke.sh").read_text(encoding="utf-8")
    assert "--audit-policy record_only" in smoke and "--smoke" in smoke
    assert "torchrun --standalone" in smoke
    assert "--audit-policy record_only" in full and "--smoke" not in full
