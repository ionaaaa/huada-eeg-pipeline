import torch
import torch.nn as nn


class PatchTokenizer(nn.Module):
    """
    把每个 channel-time patch 映射成 Transformer token embedding。

    输入:
        [B, S, L]

    输出:
        [B, S, d_model]
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)


class EEGTransformerEncoder(nn.Module):
    """
    Transformer Encoder。

    额外支持取指定深度的中间层输出，用于下游 linear probe / finetune。
    """

    def __init__(
        self,
        d_model=2048,
        n_heads=16,
        depth=24,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.depth = int(depth)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

    def forward(self, x):
        return self.encoder(x)

    def forward_until_layer(self, x, return_layer_idx):
        """
        返回指定 Transformer 层之后的 token embedding。

        Args:
            x: [B, S, d_model]
            return_layer_idx: int，0-based layer index

        Returns:
            x: [B, S, d_model]
        """
        if return_layer_idx < 0 or return_layer_idx >= self.depth:
            raise ValueError(
                f"Invalid return_layer_idx={return_layer_idx}. "
                f"Expected range: [0, {self.depth - 1}]"
            )

        for layer_idx, layer in enumerate(self.encoder.layers):
            x = layer(x)

            if layer_idx == return_layer_idx:
                return x

        return x


class TimeFreqTokenHead(nn.Module):
    """
    对每个 channel-time token 输出频段能量时间轨迹。

    输入:
        [B, S, d_model]

    输出:
        [B, S, F, K]
    """

    def __init__(
        self,
        d_model: int,
        n_bands: int,
        frames_per_patch: int,
    ):
        super().__init__()

        self.n_bands = int(n_bands)
        self.frames_per_patch = int(frames_per_patch)

        output_dim = self.n_bands * self.frames_per_patch

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        out = self.head(x)

        batch_size, num_tokens, _ = out.shape

        return out.view(
            batch_size,
            num_tokens,
            self.n_bands,
            self.frames_per_patch,
        )


class RawEEGTokenHead(nn.Module):
    """
    对每个 channel-time token 重构原始 EEG patch。

    输入:
        [B, S, d_model]

    输出:
        [B, S, L]
    """

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()

        self.patch_len = int(patch_len)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.patch_len),
        )

    def forward(self, x):
        return self.head(x)


class EEGPretrainModel(nn.Module):
    """
    ViT-style EEG 预训练模型，同时支持两种预训练目标。

    一个 token = 一个通道上的一个时间 patch。

    objective="timefreq":
        预训练输出 [B, S, F, K]

    objective="raw_eeg":
        预训练输出 [B, S, L]

    两种目标共享 tokenizer、channel/time embedding 和 Transformer encoder，
    因此下游任务可统一通过 extract_token_embeddings() 提取表示。
    """

    SUPPORTED_OBJECTIVES = {"timefreq", "raw_eeg"}

    def __init__(
        self,
        input_dim,
        d_model,
        n_heads,
        depth,
        mlp_ratio,
        dropout,
        n_channels,
        n_time_patches,
        n_bands,
        frames_per_patch,
        objective="timefreq",
    ):
        super().__init__()

        objective = str(objective).lower()
        if objective not in self.SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"Unsupported objective={objective!r}. "
                f"Expected one of {sorted(self.SUPPORTED_OBJECTIVES)}."
            )

        self.objective = objective
        self.n_channels = int(n_channels)
        self.n_time_patches = int(n_time_patches)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.input_dim = int(input_dim)
        self.n_bands = int(n_bands)
        self.frames_per_patch = int(frames_per_patch)

        self.tokenizer = PatchTokenizer(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.channel_embed = nn.Embedding(
            num_embeddings=n_channels,
            embedding_dim=d_model,
        )

        self.time_embed = nn.Embedding(
            num_embeddings=n_time_patches,
            embedding_dim=d_model,
        )

        self.encoder = EEGTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        if objective == "timefreq":
            self.head = TimeFreqTokenHead(
                d_model=d_model,
                n_bands=n_bands,
                frames_per_patch=frames_per_patch,
            )
        else:
            self.head = RawEEGTokenHead(
                d_model=d_model,
                patch_len=input_dim,
            )

    def embed_tokens(
        self,
        token_inputs,
        token_channel_indices,
        token_time_indices,
    ):
        """
        完成 patch tokenizer + channel/time positional embedding。

        Returns:
            x: [B, S, d_model]
        """
        x = self.tokenizer(token_inputs)

        ch_pos = self.channel_embed(token_channel_indices)
        time_pos = self.time_embed(token_time_indices)

        return x + ch_pos + time_pos

    def extract_token_embeddings(
        self,
        token_inputs,
        token_channel_indices,
        token_time_indices,
        return_layer_idx=None,
    ):
        """
        提取指定 Transformer 层之后的 token embedding。

        Args:
            token_inputs: [B, S, L]
            token_channel_indices: [B, S]
            token_time_indices: [B, S]
            return_layer_idx:
                None 表示取最后一层；
                int 表示取第 return_layer_idx 层之后的输出，0-based。

        Returns:
            token_embeddings: [B, S, d_model]
        """
        if return_layer_idx is None:
            return_layer_idx = self.depth - 1

        x = self.embed_tokens(
            token_inputs=token_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
        )

        return self.encoder.forward_until_layer(
            x=x,
            return_layer_idx=return_layer_idx,
        )

    def forward(
        self,
        token_inputs,
        token_channel_indices,
        token_time_indices,
    ):
        x = self.embed_tokens(
            token_inputs=token_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
        )

        x = self.encoder(x)
        return self.head(x)
