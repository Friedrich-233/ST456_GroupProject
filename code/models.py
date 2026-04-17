"""
Neural network architectures for the PCam histopathology SSL project.

Includes:
    - SimCLRModel: ResNet-18 encoder + projection head for contrastive pre-training.
    - MAEViT: a small masked autoencoder ViT for generative pre-training.
    - MAEViTImproved: a deeper MAE variant (depth=12, patch=4, + proper init).
    - ResNetBinaryClassifier and MAEBinaryClassifier: downstream binary heads.
    - build_*_classifier(): factory functions that load SSL checkpoints into a
      fresh classifier ready for fine-tuning.

Usage:
    from src.models import (
        SimCLRModel, MAEViT, MAEViTImproved,
        ResNetBinaryClassifier, MAEBinaryClassifier,
        build_simclr_classifier, build_supervised_scratch_classifier,
        build_mae_classifier, build_mae_improved_classifier,
        configure_trainable_parameters,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    from timm.models.vision_transformer import Block
except ImportError:
    Block = None  # MAE models will raise a clearer error if used


# ---------------------------------------------------------------------------
# SimCLR
# ---------------------------------------------------------------------------
class SimCLRModel(nn.Module):
    """ResNet-18 encoder + 2-layer MLP projector for SimCLR.

    The encoder is the only part saved to disk: the projector is discarded
    before downstream fine-tuning, which is the standard SimCLR recipe.
    """

    def __init__(self, base_model: str = "resnet18", out_dim: int = 128):
        super().__init__()
        if base_model != "resnet18":
            raise ValueError("This project currently supports resnet18 only.")

        self.encoder = tv_models.resnet18(weights=None)
        # Adapt the stem for 96x96 inputs: small kernel, no aggressive downsampling.
        self.encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.encoder.maxpool = nn.Identity()

        n_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_features, out_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


# ---------------------------------------------------------------------------
# MAE (base)
# ---------------------------------------------------------------------------
class MAEViT(nn.Module):
    """A compact Masked Autoencoder ViT for 96x96 images.

    Encoder depth 6, patch size 8, embed dim 192. Used as the 'base MAE'
    reference point; the Improved version is the one reported in the main
    paradigm comparison.
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        if Block is None:
            raise ImportError("timm is required for MAE. Install it with `pip install timm`.")

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, qkv_bias=True)
             for _ in range(decoder_depth)]
        )
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        n, l, d = x.shape
        len_keep = int(l * (1 - mask_ratio))
        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, imgs: torch.Tensor):
        """Pre-training forward pass: mask -> encode -> decode -> reconstruct."""
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_all = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_all = torch.gather(
            x_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_all], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :], mask

    def forward_encoder(self, imgs: torch.Tensor) -> torch.Tensor:
        """Downstream forward pass: no masking, returns the CLS token."""
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


# ---------------------------------------------------------------------------
# MAE Improved
# ---------------------------------------------------------------------------
class MAEViTImproved(nn.Module):
    """Deeper MAE with patch_size 4, depth 12, proper weight init, decoder norm.

    Used as the main MAE variant in the paradigm comparison.
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 6,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        if Block is None:
            raise ImportError("timm is required for MAE. Install it with `pip install timm`.")

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, qkv_bias=True)
             for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        n, l, d = x.shape
        len_keep = int(l * (1 - mask_ratio))
        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, imgs: torch.Tensor):
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_all = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_all = torch.gather(
            x_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_all], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :], mask

    def forward_encoder(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


# ---------------------------------------------------------------------------
# Downstream classifiers
# ---------------------------------------------------------------------------
class ResNetBinaryClassifier(nn.Module):
    """ResNet encoder + linear binary classification head."""

    def __init__(self, encoder: nn.Module, feature_dim: int = 512):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(feature_dim, 2)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)


class MAEBinaryClassifier(nn.Module):
    """MAE encoder + linear binary classification head. Uses the CLS token."""

    def __init__(self, encoder: nn.Module, feature_dim: int = 192):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(feature_dim, 2)

    def forward(self, x):
        features = self.encoder.forward_encoder(x)
        return self.fc(features)


# ---------------------------------------------------------------------------
# Fine-tuning strategy configuration
# ---------------------------------------------------------------------------
def configure_trainable_parameters(model: nn.Module, strategy: str) -> None:
    """Freeze or unfreeze parameters in-place according to the chosen strategy.

    strategy == 'frozen' : encoder is fully frozen, only fc is trained.
    strategy == 'partial': encoder last stage / last 2 transformer blocks are
                           also unfrozen.
    strategy == 'full'   : everything is trained.
    """
    if strategy not in {"frozen", "partial", "full"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    for param in model.parameters():
        param.requires_grad = False

    # The fc head is always trained.
    for param in model.fc.parameters():
        param.requires_grad = True

    if strategy == "full":
        for param in model.encoder.parameters():
            param.requires_grad = True
    elif strategy == "partial":
        if isinstance(model, ResNetBinaryClassifier):
            for name, param in model.encoder.named_parameters():
                if name.startswith("layer4"):
                    param.requires_grad = True
        elif isinstance(model, MAEBinaryClassifier):
            n_blocks = len(model.encoder.blocks)
            for block in model.encoder.blocks[max(0, n_blocks - 2):]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in model.encoder.norm.parameters():
                param.requires_grad = True


# ---------------------------------------------------------------------------
# Classifier factories
# ---------------------------------------------------------------------------
def _build_resnet_encoder() -> nn.Module:
    """Build a 96x96-friendly ResNet-18 encoder (no fc) used everywhere."""
    encoder = tv_models.resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.Identity()
    encoder.fc = nn.Identity()
    return encoder


def build_simclr_classifier(checkpoint_path: Path) -> ResNetBinaryClassifier:
    """Load a SimCLR-pretrained encoder and wrap it with a fresh fc head."""
    encoder = _build_resnet_encoder()
    feature_dim = 512  # resnet18
    state = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(state)
    return ResNetBinaryClassifier(encoder=encoder, feature_dim=feature_dim)


def build_supervised_scratch_classifier() -> ResNetBinaryClassifier:
    """Randomly initialised ResNet-18 classifier (supervised-from-scratch baseline)."""
    encoder = _build_resnet_encoder()
    feature_dim = 512
    return ResNetBinaryClassifier(encoder=encoder, feature_dim=feature_dim)


def build_mae_classifier(checkpoint_path: Path) -> MAEBinaryClassifier:
    """Load a base-MAE encoder and wrap it with a fresh fc head."""
    encoder = MAEViT()
    state = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(state)
    return MAEBinaryClassifier(encoder=encoder, feature_dim=192)


def build_mae_improved_classifier(checkpoint_path: Path) -> MAEBinaryClassifier:
    """Load an improved-MAE encoder and wrap it with a fresh fc head."""
    encoder = MAEViTImproved()
    state = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(state)
    return MAEBinaryClassifier(encoder=encoder, feature_dim=192)
