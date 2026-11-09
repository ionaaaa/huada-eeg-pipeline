"""Build one native full-recording cache from one audited Huada BDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import load_huada_target_contract
from .preprocessing import preprocess_bdf_to_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", required=True, choices=("labram", "reve"))
    parser.add_argument("--source-bdf", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--target-input-contract", type=Path, required=True)
    parser.add_argument("--expected-target-contract-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_huada_target_contract(
        args.target_input_contract,
        expected_family=args.model_family,
        expected_sha256=args.expected_target_contract_file_sha256,
    )
    resolved = {
        "model_family": args.model_family,
        "source_bdf": str(args.source_bdf),
        "recording_id": args.recording_id,
        "target_input_contract": str(args.target_input_contract),
        "target_contract_sha256": contract["contract_sha256"],
        "output_dir": str(args.output_dir),
        "dry_run": bool(args.dry_run),
        "fbmae_token_lance_reused": False,
    }
    print(json.dumps(resolved, ensure_ascii=False, sort_keys=True, indent=2))
    if args.dry_run:
        return 0
    output = preprocess_bdf_to_cache(
        args.source_bdf,
        args.output_dir,
        family=args.model_family,
        contract=dict(contract),
        recording_id=args.recording_id,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
