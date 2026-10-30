"""50M downstream configuration, ported from upstream ``configs.py``.

Only Huada-required task, manifest, and output controls were added. Model and
optimization defaults retain the upstream values.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    use_multi_lance_parts: bool = False
    lance_path: str = ""
    storage_options_path: str | None = None
    convert_v_to_uv: bool = True
    clip_seconds: float = 10.0
    clip_stride_seconds: float = 10.0
    target_sfreq: float = 100.0
    n_channels: int = 64
    patch_seconds: float = 1.0
    patch_stride_seconds: float = 1.0
    target_frame_seconds: float = 0.2
    stft_window_seconds: float = 1.0
    stft_hop_seconds: float = 0.1
    band_defs: list = field(default_factory=lambda: [
        ("delta", 1.0, 4.0), ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0),
        ("gamma_low", 30.0, 45.0),
    ])
    eps: float = 1e-6


@dataclass
class MaskConfig:
    mask_ratio: float = 0.5
    min_block_tokens: int = 4
    max_block_tokens: int = 32


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    depth: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 0
    num_epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-5


@dataclass
class FinetuneConfig:
    """Downstream defaults retained from upstream unless noted."""

    seed: int = 42
    device: str = "cuda"
    experiment_name: str = "huada_50m_downstream"
    output_root: str = ""
    run_dir: str = ""
    checkpoint_dir: str = ""
    metrics_dir: str = ""
    batch_size: int = 64
    num_workers: int = 0
    use_prefetch_loader: bool = True
    prefetch: int = 2
    num_epochs: int = 5
    lr: float = 1e-3
    backbone_lr: float = 1e-3
    head_lr: float = 1e-3
    weight_decay: float = 1e-3
    probe_layer_ratio: float = 0.75
    save_path: str = ""
    scaling_result_csv: str = ""
    num_classes: int = 2
    mode: str = "linear_probe"
    use_mlp_head: bool = False
    mlp_hidden_dim: int = 2048
    head_dropout: float = 0.2
    metric_for_best: str = "val_acc"

    # Huada adapters. Full runs require an explicit subject split manifest.
    task: str = "sex_classification"
    subject_split_manifest: str = ""
    pretrained_ckpt_path: str = ""
    downstream_checkpoint: str = ""
    split_seed: int = 42
    protocol: str = "legacy_single_split"
    folds: int = 4
    smoke: bool = False
    smoke_max_samples: int = 4
    smoke_max_samples_per_subject: int = 1
    allow_random_init_smoke: bool = False
    evaluate_only: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
