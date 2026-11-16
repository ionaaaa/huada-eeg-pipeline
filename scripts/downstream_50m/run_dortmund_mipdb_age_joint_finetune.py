#!/usr/bin/env python3
"""Joint Dortmund+MIPDB entry using the shared age full-finetune loop.

Only the two frozen B/C profiles are accepted here.  All model, optimizer,
scheduler, DDP, selection, checkpoint, and reporting logic remains in the same
runner used by Dortmund-only experiments.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from age_joint_training import JOINT_PROFILE_IDS
from run_dortmund_age_full_finetune import main as shared_main


def validate_joint_cli(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--training-profile", required=True)
    known, _ = parser.parse_known_args(argv)
    if known.training_profile not in JOINT_PROFILE_IDS:
        raise ValueError(
            f"Joint entry accepts only {list(JOINT_PROFILE_IDS)}; got {known.training_profile!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    validate_joint_cli(values)
    return shared_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
