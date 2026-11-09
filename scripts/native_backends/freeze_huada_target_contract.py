"""Freeze Huada LaBraM/REVE target-input contracts without opening BDF data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .contracts import (
    REVE_POSITION_SOURCE_SHA256,
    build_reve_position_bank,
    make_huada_target_contract,
    write_huada_target_contract,
)


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"not a regular non-symlink file: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", required=True, choices=("labram", "reve"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reve-position-source", type=Path)
    parser.add_argument(
        "--expected-reve-position-source-sha256",
        default=REVE_POSITION_SOURCE_SHA256,
    )
    parser.add_argument("--channel-name-mapping", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bank = None
    if args.model_family == "reve":
        if args.reve_position_source is None:
            raise SystemExit("REVE requires --reve-position-source")
        source_path = _regular_file(args.reve_position_source)
        observed = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if observed != args.expected_reve_position_source_sha256:
            raise SystemExit(
                f"REVE position source SHA mismatch: expected={args.expected_reve_position_source_sha256}, actual={observed}"
            )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        mapping = None
        if args.channel_name_mapping is not None:
            mapping = json.loads(
                _regular_file(args.channel_name_mapping).read_text(encoding="utf-8")
            )
        bank = build_reve_position_bank(
            source,
            source_asset_sha256=observed,
            channel_name_mapping=mapping,
        )
    contract = make_huada_target_contract(
        args.model_family, reve_position_bank=bank
    )
    output = write_huada_target_contract(args.output, contract)
    print(output)
    print(f"contract_sha256={contract['contract_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
