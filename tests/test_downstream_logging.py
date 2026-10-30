from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from run_logging import log_event, log_exception, setup_run_logging
import feature_cache as cache
import largeScale_fineTune as training


def test_logging_writes_stdout_text_and_structured_events(tmp_path, capsys):
    logger = setup_run_logging(tmp_path, "INFO", append=False)
    log_event(
        logger, "INFO", "candidate.start", "candidate ready",
        fold=2, candidate_id="candidate-a", backbone_forward_count=0,
    )
    stdout = capsys.readouterr().out
    assert "INFO | candidate.start" in stdout
    assert "fold=2" in stdout and "candidate_id=candidate-a" in stdout
    assert "candidate ready" in (tmp_path / "run.log").read_text(encoding="utf-8")
    event = json.loads((tmp_path / "run_events.jsonl").read_text(encoding="utf-8"))
    assert event["stage"] == "candidate.start"
    assert event["fold"] == 2
    assert event["candidate_id"] == "candidate-a"


def test_exception_log_contains_context_and_traceback(tmp_path):
    logger = setup_run_logging(tmp_path, "INFO", append=False)
    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("fixture failure")
        except RuntimeError:
            log_exception(
                logger, "cache.failed", "cache build failed",
                fold=1, candidate_id="candidate-b", cache_path="fixture-cache",
            )
            raise
    text = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "cache.failed" in text
    assert "fold=1" in text
    assert "candidate_id=candidate-b" in text
    assert "cache_path=fixture-cache" in text
    assert "Traceback" in text and "fixture failure" in text


def test_cache_hit_explicitly_logs_extraction_skipped(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"fixture")
    hit_path = tmp_path / "cache" / ("a" * 64)
    manifest = {
        "mean_row_count": 3, "mean_feature_dim": 512,
        "flatten_feature_dim": 327680,
    }
    monkeypatch.setattr(
        cache, "lance_fingerprint",
        lambda *args, **kwargs: ("input-fingerprint", {
            "row_count": 3, "sample_id_fingerprint": "samples",
        }),
    )
    monkeypatch.setattr(cache, "get_git_state", lambda root: ("commit", False))
    monkeypatch.setattr(cache, "model_implementation_fingerprint", lambda root: "code")
    monkeypatch.setattr(
        cache, "resolve_cache_policy",
        lambda *args, **kwargs: (hit_path, manifest, True),
    )
    logger = setup_run_logging(tmp_path / "logs", "INFO", append=False)
    options = SimpleNamespace(
        storage_options=None, input_lance="fixture.lance", checkpoint=checkpoint,
        device="cpu", feature_cache_root=tmp_path / "cache",
        feature_cache_policy="auto", logger=logger,
    )
    result = cache.ensure_feature_cache(options)
    output = capsys.readouterr().out
    assert result["feature_cache_reused"] is True
    assert "cache hit: feature extraction skipped" in output
    assert "row_count=3" in output


def test_cache_miss_explicitly_logs_build_reason(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(
        cache, "lance_fingerprint",
        lambda *args, **kwargs: ("input-fingerprint", {
            "row_count": 1, "sample_id_fingerprint": "samples",
        }),
    )
    monkeypatch.setattr(cache, "get_git_state", lambda root: ("commit", False))
    monkeypatch.setattr(cache, "model_implementation_fingerprint", lambda root: "code")
    source = SimpleNamespace(metadata=[{
        "sample_id": "sample-1", "subject_id": "SUB1", "recording_id": None,
    }])
    monkeypatch.setattr(cache, "TokenFeatureSourceDataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        training, "build_backbone",
        lambda *args, **kwargs: (object(), {
            "matched_key_count": 10, "matched_parameter_fraction": 1.0,
        }),
    )
    built_manifest = {
        "backbone_processed_window_count": 1, "backbone_forward_count": 1,
        "mean_row_count": 1, "flatten_row_count": 1,
        "mean_feature_dim": 512, "flatten_feature_dim": 327680,
    }

    def fake_build(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        (output / cache.SUCCESS_NAME).write_text("fixture\n", encoding="utf-8")
        return built_manifest

    monkeypatch.setattr(cache, "build_feature_cache", fake_build)
    monkeypatch.setattr(cache, "validate_feature_cache", lambda *args, **kwargs: built_manifest)
    logger = setup_run_logging(tmp_path / "logs", "INFO", append=False)
    options = SimpleNamespace(
        storage_options=None, input_lance="fixture.lance", checkpoint=checkpoint,
        device="cpu", feature_cache_root=tmp_path / "cache",
        feature_cache_policy="auto", feature_cache_batch_size=2,
        cache_log_every_batches=1, logger=logger,
    )
    result = cache.ensure_feature_cache(options)
    output = capsys.readouterr().out
    assert result["feature_cache_reused"] is False
    assert "cache miss: building feature cache" in output
    assert "reason=matching complete cache not found" in output
