#!/usr/bin/env python3
"""Two-process CPU/Gloo control smoke used when two GPUs are unavailable."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dortmund_age_finetune import (
    NoPaddingDistributedEvalSampler,
    SubjectBalancedDistributedSampler,
    atomic_torch_save,
    initialize_distributed,
    validate_prediction_coverage,
)


def rows():
    result = []
    index = 0
    for subject in ("s0", "s1", "s2"):
        for condition in ("eyes-open", "eyes-closed"):
            for window in range(2):
                result.append({
                    "index": index, "sample_id": f"{subject}-{condition}-{window}",
                    "subject_id": subject, "eyes_condition": condition,
                })
                index += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    context, device = initialize_distributed("cpu")
    rank, world_size = context["rank"], context["world_size"]
    if world_size != 2 or context["backend"] != "gloo":
        raise RuntimeError("This smoke requires exactly two torchrun CPU/Gloo processes")
    try:
        sampler = SubjectBalancedDistributedSampler(
            rows(), samples_per_epoch=11, seed=42, rank=rank,
            world_size=world_size, per_device_batch_size=2,
        )
        local_positions = list(sampler)
        gathered_positions = [None] * world_size
        dist.all_gather_object(gathered_positions, local_positions)
        if rank == 0:
            global_positions = sampler.global_indices()
            assert gathered_positions[0] == global_positions[0::2]
            assert gathered_positions[1] == global_positions[1::2]
            assert len(gathered_positions[0]) == len(gathered_positions[1])

        evaluation = list(NoPaddingDistributedEvalSampler(7, rank, world_size))
        local_predictions = [{"sample_id": str(index)} for index in evaluation]
        gathered_predictions = [None] * world_size
        dist.all_gather_object(gathered_predictions, local_predictions)
        if rank == 0:
            flat = [row for part in gathered_predictions for row in part]
            validate_prediction_coverage(flat, [str(index) for index in range(7)])

        torch.manual_seed(123)
        model = DistributedDataParallel(torch.nn.Linear(4, 1).to(device), find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4) + rank
        y = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        checkpoint = args.output_dir / "ddp_control_checkpoint.pt"
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            atomic_torch_save({
                "model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": 0, "world_size": world_size,
            }, checkpoint)
        dist.barrier()
        resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if resumed["world_size"] != world_size or resumed["epoch"] != 0:
            raise RuntimeError("DDP control resume compatibility check failed")
        model.module.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        resumed_loss = torch.nn.functional.mse_loss(model(x), y)
        resumed_loss.backward(); optimizer.step()
        fingerprint = torch.cat([parameter.detach().flatten() for parameter in model.module.parameters()])
        gathered_fingerprints = [torch.empty_like(fingerprint) for _ in range(world_size)]
        dist.all_gather(gathered_fingerprints, fingerprint)
        if not all(torch.equal(gathered_fingerprints[0], value) for value in gathered_fingerprints[1:]):
            raise RuntimeError("DDP parameters diverged")
        if rank == 0:
            payload = {
                "status": "passed", "backend": "gloo", "world_size": world_size,
                "rank_devices": ["cpu", "cpu"], "equal_train_steps": True,
                "sampler_global_sequence_sharded": True, "ddp_padding_duplicates": 0,
                "evaluation_no_padding_complete": True, "prediction_duplicates": 0,
                "rank0_only_output": True, "checkpoint_reload": True,
                "resume_completed_one_step": True, "parameters_synchronized": True,
                "two_gpu_ddp_smoke": "pending_gpu_count_1",
            }
            (args.output_dir / "ddp_control_smoke.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (args.output_dir / "_SUCCESS").write_text("passed\n", encoding="utf-8")
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
