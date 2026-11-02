"""Small standard-library logging helpers for downstream runs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


LOGGER_NAME = "huada.downstream_50m"
STAGE_ALIASES = {
    # Public log naming can evolve without changing feature-producing modules,
    # whose file hashes intentionally participate in the frozen cache key.
    "candidate.epoch": "train.epoch",
}


class _StageFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "stage"):
            record.stage = "general"
        return True


class _JsonlHandler(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.stream = self.path.open("a", encoding="utf-8")

    def emit(self, record):
        try:
            event = dict(getattr(record, "event", {}))
            event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            event.setdefault("level", record.levelname)
            event.setdefault("stage", getattr(record, "stage", "general"))
            event.setdefault("message", record.getMessage())
            if record.exc_info:
                event["traceback"] = self.formatException(record.exc_info)
            self.stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self.flush()
        except Exception:
            self.handleError(record)

    @staticmethod
    def formatException(exc_info):
        return logging.Formatter().formatException(exc_info)

    def flush(self):
        if self.stream:
            self.stream.flush()

    def close(self):
        try:
            if self.stream:
                self.stream.close()
        finally:
            super().close()


def setup_run_logging(output_dir, level="INFO", append=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False
    logger.setLevel(getattr(logging, str(level).upper()))
    stage_filter = _StageFilter()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(stage)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(stage_filter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(
        output_dir / "run.log", mode="a" if append else "w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(stage_filter)
    logger.addHandler(file_handler)
    json_handler = _JsonlHandler(output_dir / "run_events.jsonl")
    json_handler.addFilter(stage_filter)
    logger.addHandler(json_handler)
    return logger


def get_run_logger():
    return logging.getLogger(LOGGER_NAME)


def log_event(logger, level, stage, message, **fields):
    if logger is None:
        return
    stage = STAGE_ALIASES.get(stage, stage)
    event = {
        "stage": stage,
        "fold": fields.get("fold"),
        "candidate_id": fields.get("candidate_id"),
        **fields,
    }
    suffix = " ".join(
        f"{key}={value}" for key, value in fields.items() if value is not None
    )
    rendered = message if not suffix else f"{message} | {suffix}"
    logger.log(
        getattr(logging, str(level).upper()), rendered,
        extra={"stage": stage, "event": event},
    )


def log_exception(logger, stage, message, **fields):
    if logger is None:
        return
    event = {"stage": stage, **fields}
    suffix = " ".join(
        f"{key}={value}" for key, value in fields.items() if value is not None
    )
    rendered = message if not suffix else f"{message} | {suffix}"
    logger.exception(rendered, extra={"stage": stage, "event": event})
