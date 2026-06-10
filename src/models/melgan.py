"""
MelGAN-Style Generator for Spectrogram Domain Style Transfer
=============================================================
Adapted from Kumar et al. (2019) "MelGAN: Generative Adversarial Networks
for Conditional Waveform Synthesis" — re-purposed here for spectrogram-space
speaker style transfer rather than waveform synthesis.

Architecture key:
  - 1-D dilated residual convolution stack over the time axis
  - Upsample → residual dilation → downsample back to original resolution
  - Feature-matching loss for content preservation (no cycle constraint)

This is Model 2a in the thesis GAN comparison chapter.
Compare with: CycleGAN (Model 2b) and MelGAN-Cycle hybrid (Model 2c).

Reference:
    Kumar, K. et al. (2019). MelGAN: Generative Adversarial Networks for
    Conditional Waveform Synthesis. NeurIPS.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dilated Residual Block (1-D, operates on time axis)
# ---------------------------------------------------------------------------

class DilatedResBlock(nn.Module):
    """
    Residual block with dilated convolution — the core MelGAN building block.

    Dilation expands the receptive field exponentially without increasing
    parameter count. A stack of dilation=[1,3,9] covers 27 frames (~0.3s)
    with only 3 layers.

    Args:
        channels: Number of feature channels.
        dilation:  Dilation factor for the 3-tap conv.
    """

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# MelGAN Generator
# ---------------------------------------------------------------------------

class MelGANGenerator(nn.Module):
    """
    MelGAN-inspired generator operating in 2-D spectrogram space.

    The spectrogram (B, 1, n_mels, T) is treated as a 1-D sequence
    over the time axis, with mel bands as channels. This lets the dilated
    conv stack model long-range temporal dependencies — important for
    capturing prosodic patterns (speaking rhythm, stress).

    Pipeline:
        Input (B,1,80,T) → squeeze → (B,80,T)
        Pre-conv          → (B, nc*4, T)
        Upsample ×4       → (B, nc,   T*4)
        Dilated res stack  → (B, nc,   T*4)
        Post-conv          → (B, 80,   T*4)
        AdaptivePool       → (B, 80,   T)     ← back to original length
        Unsqueeze          → (B, 1,   80, T)

    Args:
        n_mels:      Number of mel frequency bands (default 80).
        base_ch:     Base channel multiplier (default 32).
        dilations:   Dilation schedule for residual stack.
    """

    def __init__(
        self,
        n_mels: int = 80,
        base_ch: int = 32,
        dilations: List[int] = [1, 3, 9],
    ):
        super().__init__()
        nc = base_ch

        self.pre_conv = nn.Conv1d(n_mels, nc * 4, kernel_size=7, padding=3)

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(nc * 4, nc * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(nc * 2, nc,     kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res_stack = nn.Sequential(
            *[DilatedResBlock(nc, d) for d in dilations]
        )

        self.post_conv = nn.Sequential(
            nn.Conv1d(nc, n_mels // 4, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(n_mels // 4, n_mels, kernel_size=1),
            nn.Tanh(),
        )

        self.resize = nn.AdaptiveAvgPool1d(128)   # normalise output length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spectrogram tensor (B, 1, n_mels, T).
        Returns:
            Style-transferred spectrogram (B, 1, n_mels, T).
        """
        h = x.squeeze(1)          # (B, n_mels, T)
        h = self.pre_conv(h)       # (B, nc*4,   T)
        h = self.upsample(h)       # (B, nc,      T*4)
        h = self.res_stack(h)      # (B, nc,      T*4)
        h = self.post_conv(h)      # (B, n_mels,  T*4)
        h = self.resize(h)         # (B, n_mels,  128)
        return h.unsqueeze(1)      # (B, 1, n_mels, 128)


# ---------------------------------------------------------------------------
# Multi-Scale Discriminator (MelGAN-style)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator classifying spectrogram patches as real/fake.

    Operates in 2-D spectrogram space. Patch-level discrimination forces
    the generator to produce locally realistic spectro-temporal patterns.
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# MelGAN Trainer
# ---------------------------------------------------------------------------

class MelGAN:
    """
    Training wrapper for standalone MelGAN voice style transfer.

    Training objective:
        L_total = L_adv + λ_fm * L_feature_matching

    Feature matching loss (Larsen et al. 2016):
        Encourages the generator output to match the style reference
        in raw spectrogram space. This serves as a soft content
        preservation signal in the absence of cycle consistency.

    Note: Without cycle-consistency, there is no hard guarantee that
    linguistic content is preserved — this is the key limitation vs
    CycleGAN, and is the central finding of the GAN comparison chapter.
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.cfg = cfg
        self.lambda_fm = cfg["training"].get("lambda_fm", 2.0)

        self.G  = MelGANGenerator(**cfg.get("generator", {})).to(self.device)
        self.D  = PatchDiscriminator(**cfg.get("discriminator", {})).to(self.device)

        self.adv_loss = nn.MSELoss()
        self.fm_loss  = nn.L1Loss()

        self.opt_G = optim.Adam(self.G.parameters(),
                                lr=cfg["training"]["lr"], betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(),
                                lr=cfg["training"]["lr"], betas=(0.5, 0.999))

    def train_step(
        self,
        real_X: torch.Tensor,
        real_Y: torch.Tensor,
    ) -> Dict[str, float]:
        real_X = real_X.to(self.device)
        real_Y = real_Y.to(self.device)

        # Generator
        self.opt_G.zero_grad()
        fake_Y = self.G(real_X)
        loss_adv = self.adv_loss(self.D(fake_Y),
                                  torch.ones_like(self.D(fake_Y)))
        loss_fm  = self.fm_loss(fake_Y, real_X) * self.lambda_fm
        loss_G   = loss_adv + loss_fm
        loss_G.backward()
        self.opt_G.step()

        # Discriminator
        self.opt_D.zero_grad()
        loss_D = 0.5 * (
            self.adv_loss(self.D(real_Y),        torch.ones_like(self.D(real_Y))) +
            self.adv_loss(self.D(fake_Y.detach()),torch.zeros_like(self.D(fake_Y.detach())))
        )
        loss_D.backward()
        self.opt_D.step()

        return {
            "loss_G": loss_G.item(),
            "loss_adv": loss_adv.item(),
            "loss_fm": loss_fm.item(),
            "loss_D": loss_D.item(),
        }

    def translate(self, spec_X: torch.Tensor) -> torch.Tensor:
        self.G.eval()
        with torch.no_grad():
            return self.G(spec_X.to(self.device)).cpu()

    def save_checkpoint(self, path: str, epoch: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "G": self.G.state_dict(),
                    "D": self.D.state_dict()}, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        return ckpt["epoch"]
