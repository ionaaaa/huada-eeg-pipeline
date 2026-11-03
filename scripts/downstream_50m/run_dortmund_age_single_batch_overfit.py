#!/usr/bin/env python3
"""Train-only fixed 16-window overfit diagnostic for Dortmund 50M gradients."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from configs import Config
from dataset import collate_fn
from dortmund_age_finetune import (
    StrictDortmundAgeModel,
    audit_gradient_coverage,
    build_optimizer,
    build_target_transform,
    configure_probe_gradient_scope,
    ensure_finite,
    sha256_file,
    zero_initialize_age_head,
)
from lance_dataset import LanceEEGPretrainDataset, load_subject_manifest
from largeScale_fineTune import build_backbone, compute_shape_params
from utils import set_seed


def tensor_snapshot(parameters):
    return {name: value.detach().cpu().clone() for name, value in parameters}


def changed(before, after, names):
    return any(not torch.equal(before[name], after[name]) for name in names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-random-init-smoke", action="store_true")
    parser.add_argument("--windows", type=int, default=16, choices=range(16, 33))
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-mode", choices=["none", "bf16"], default="bf16")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.micro_batch_size < 1 or args.steps < 2 or args.windows % args.micro_batch_size:
        raise ValueError("windows must be divisible by micro batch size; at least two steps required")
    if args.checkpoint is None and not args.allow_random_init_smoke:
        raise ValueError("Missing checkpoint; random init requires explicit smoke authorization")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_rows = list(load_subject_manifest(args.split_manifest).values())
    transform = build_target_transform(manifest_rows, sha256_file(args.split_manifest))
    cfg = Config(); cfg.finetune.smoke = True
    cfg.finetune.allow_random_init_smoke = args.allow_random_init_smoke
    cfg.finetune.pretrained_ckpt_path = "" if args.checkpoint is None else str(args.checkpoint)
    shape = compute_shape_params(cfg)
    backbone, checkpoint_report = build_backbone(cfg, shape, device, load_pretrained=True)
    model = StrictDortmundAgeModel(
        backbone, cfg.model.d_model, 1, probe_layer_idx=8,
        freeze_backbone=False, pooling="mean", num_tokens=shape["num_tokens"],
    ).to(device)
    zero_initialize_age_head(model)
    scope = configure_probe_gradient_scope(model, 8)
    optimizer, groups = build_optimizer(model, backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)
    dataset = LanceEEGPretrainDataset(
        args.input_lance, task="age_regression",
        subject_split_manifest=args.split_manifest, split="train",
    )
    sampling_rows = dataset.get_sampling_rows()
    selected = sorted(
        range(len(sampling_rows)),
        key=lambda index: (sampling_rows[index]["subject_id"], sampling_rows[index]["sample_id"]),
    )[:args.windows]
    selected_rows = [sampling_rows[index] for index in selected]
    if len({row["subject_id"] for row in selected_rows}) < 1:
        raise RuntimeError("No train subjects selected")
    loader = DataLoader(
        Subset(dataset, selected), batch_size=args.micro_batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )
    before = tensor_snapshot(model.named_parameters())
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    frozen_names = [name for name, value in model.named_parameters() if not value.requires_grad]
    rows = []
    first_gradient_audit = None
    for step in range(args.steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0; total_absolute = 0.0; total_count = 0
        for batch in loader:
            token_inputs = batch["token_inputs"].to(device)
            channels = batch["token_channel_indices"].to(device)
            times = batch["token_time_indices"].to(device)
            mask = batch["token_valid_mask"].to(device)
            years = batch["label"].float().to(device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp_mode == "bf16" and device.type == "cuda" else torch.autocast(device.type, enabled=False)
            with autocast:
                prediction = model(token_inputs, channels, times, mask).view(-1)
                target = transform.transform_tensor(years)
                loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
            ensure_finite("overfit loss", loss)
            (loss / len(loader)).backward()
            total_loss += float(loss.item()) * len(years)
            total_absolute += float((transform.inverse_tensor(prediction.float()) - years).abs().sum().item())
            total_count += len(years)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if first_gradient_audit is None:
            first_gradient_audit = audit_gradient_coverage(model)
            if first_gradient_audit["unexplained_trainable_without_gradient"]:
                raise RuntimeError(first_gradient_audit)
        optimizer.step()
        rows.append({
            "step": step, "z_loss": total_loss / total_count,
            "mae_years": total_absolute / total_count,
            "gradient_norm": float(grad_norm.item()),
        })
    after = tensor_snapshot(model.named_parameters())
    ensure_finite("final parameters", torch.cat([value.flatten() for value in model.parameters()]))
    backbone_names = [name for name in trainable_names if name.startswith("backbone.")]
    head_names = [name for name in trainable_names if name.startswith("head.")]
    result = {
        "status": "passed",
        "diagnostic_only": True,
        "validation_dataset_created": False,
        "test_dataset_created": False,
        "window_count": args.windows,
        "subject_ids": sorted({row["subject_id"] for row in selected_rows}),
        "sample_ids": [row["sample_id"] for row in selected_rows],
        "effective_batch_size": args.windows,
        "micro_batch_size": args.micro_batch_size,
        "accumulation_steps": len(loader),
        "initial_z_loss": rows[0]["z_loss"],
        "final_z_loss": rows[-1]["z_loss"],
        "loss_reduction_fraction": (rows[0]["z_loss"] - rows[-1]["z_loss"]) / rows[0]["z_loss"],
        "backbone_trainable_parameters_changed": changed(before, after, backbone_names),
        "age_head_parameters_changed": changed(before, after, head_names),
        "frozen_parameters_unchanged": not changed(before, after, frozen_names),
        "gradient_coverage": first_gradient_audit,
        "gradient_scope": scope,
        "optimizer_groups": groups,
        "target_transform": asdict(transform),
        "checkpoint": checkpoint_report,
    }
    if not result["backbone_trainable_parameters_changed"] or not result["age_head_parameters_changed"] or not result["frozen_parameters_unchanged"]:
        raise RuntimeError(f"Overfit parameter-scope assertion failed: {result}")
    if rows[-1]["z_loss"] >= rows[0]["z_loss"]:
        raise RuntimeError(f"Fixed-batch loss did not decrease: {rows}")
    with (args.output_dir / "overfit_trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "overfit_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").write_text("passed\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
