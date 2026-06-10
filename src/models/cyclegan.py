"""
CycleGAN for Unpaired Voice Conversion
=======================================
Zhu et al. (2017) adapted for spectrogram-domain cross-speaker voice conversion.

No parallel corpus required. Cycle-consistency enforces content preservation:
    F(G(x)) ≈ x    and    G(F(y)) ≈ y

Reference:
    Zhu, J. Y. et al. (2017). Unpaired Image-to-Image Translation using
    Cycle-Consistent Adversarial Networks. ICCV.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, c, 3, padding=1), nn.InstanceNorm2d(c))
    def forward(self, x): return x + self.b(x)


class Generator(nn.Module):
    """ResNet generator: encode → residual bottleneck → decode."""

    def __init__(self, nc: int = 16, n_res: int = 2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, nc, 7, padding=3), nn.InstanceNorm2d(nc), nn.ReLU(True),
            nn.Conv2d(nc, nc*2, 4, stride=2, padding=1), nn.InstanceNorm2d(nc*2), nn.ReLU(True))
        self.res = nn.Sequential(*[ResBlock(nc*2) for _ in range(n_res)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(nc*2, nc, 4, stride=2, padding=1),
            nn.InstanceNorm2d(nc), nn.ReLU(True),
            nn.Conv2d(nc, 1, 7, padding=3), nn.Tanh())

    def forward(self, x): return self.dec(self.res(self.enc(x)))


class PatchDiscriminator(nn.Module):
    """PatchGAN: classify NxN spectrogram patches as real/fake."""

    def __init__(self, nc: int = 16):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(1, nc, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nc, nc*2, 4, stride=2, padding=1), nn.InstanceNorm2d(nc*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nc*2, 1, 4, padding=1))
    def forward(self, x): return self.m(x)


# ---------------------------------------------------------------------------
# CycleGAN trainer
# ---------------------------------------------------------------------------

class CycleGAN:
    """
    Full CycleGAN training for unpaired voice conversion.

    Domains:
        X = content speaker (GANINP4 recordings)
        Y = style speaker   (APJ Abdul Kalam recordings)
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.device     = torch.device(device)
        self.lambda_cyc = cfg["training"].get("lambda_cyc",  10.0)
        self.lambda_idt = cfg["training"].get("lambda_idt",   5.0)

        self.G  = Generator(**cfg.get("generator", {})).to(self.device)
        self.F  = Generator(**cfg.get("generator", {})).to(self.device)
        self.DX = PatchDiscriminator(**cfg.get("discriminator", {})).to(self.device)
        self.DY = PatchDiscriminator(**cfg.get("discriminator", {})).to(self.device)

        self.adv  = nn.MSELoss()
        self.cyc  = nn.L1Loss()
        self.idt  = nn.L1Loss()

        self.opt_G = optim.Adam(
            list(self.G.parameters()) + list(self.F.parameters()),
            lr=cfg["training"]["lr"], betas=(0.5, 0.999))
        self.opt_D = optim.Adam(
            list(self.DX.parameters()) + list(self.DY.parameters()),
            lr=cfg["training"]["lr"], betas=(0.5, 0.999))

    def train_step(self, real_X: torch.Tensor, real_Y: torch.Tensor) -> Dict[str, float]:
        real_X, real_Y = real_X.to(self.device), real_Y.to(self.device)

        # Generators
        self.opt_G.zero_grad()
        fake_Y = self.G(real_X);  fake_X = self.F(real_Y)
        rec_X  = self.F(fake_Y);  rec_Y  = self.G(fake_X)

        l_adv = (self.adv(self.DY(fake_Y), torch.ones_like(self.DY(fake_Y))) +
                 self.adv(self.DX(fake_X), torch.ones_like(self.DX(fake_X))))
        l_cyc = (self.cyc(rec_X, real_X) + self.cyc(rec_Y, real_Y)) * self.lambda_cyc
        l_idt = (self.idt(self.G(real_Y), real_Y) +
                 self.idt(self.F(real_X), real_X)) * self.lambda_idt
        l_G   = l_adv + l_cyc + l_idt
        l_G.backward(); self.opt_G.step()

        # Discriminators
        self.opt_D.zero_grad()
        l_DY = 0.5 * (self.adv(self.DY(real_Y), torch.ones_like(self.DY(real_Y))) +
                      self.adv(self.DY(fake_Y.detach()), torch.zeros_like(self.DY(fake_Y.detach()))))
        l_DX = 0.5 * (self.adv(self.DX(real_X), torch.ones_like(self.DX(real_X))) +
                      self.adv(self.DX(fake_X.detach()), torch.zeros_like(self.DX(fake_X.detach()))))
        l_D  = l_DY + l_DX
        l_D.backward(); self.opt_D.step()

        return {"loss_G": l_G.item(), "loss_adv": l_adv.item(),
                "loss_cyc": l_cyc.item(), "loss_D": l_D.item()}

    def translate(self, spec_X: torch.Tensor) -> torch.Tensor:
        self.G.eval()
        with torch.no_grad():
            return self.G(spec_X.to(self.device)).cpu()

    def save_checkpoint(self, path: str, epoch: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "G": self.G.state_dict(), "F": self.F.state_dict(),
                    "DX": self.DX.state_dict(), "DY": self.DY.state_dict()}, path)

    def load_checkpoint(self, path: str) -> int:
        c = torch.load(path, map_location=self.device)
        self.G.load_state_dict(c["G"]); self.F.load_state_dict(c["F"])
        self.DX.load_state_dict(c["DX"]); self.DY.load_state_dict(c["DY"])
        return c["epoch"]
