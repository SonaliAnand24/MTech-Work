"""
Variational Autoencoder with Content-Style Disentanglement
===========================================================
Explicit factorisation:
    z_content  — speaker-independent (linguistic)
    z_style    — speaker-dependent (timbral identity)

At inference:
    Decoder( z_content[self], z_style[Kalam] ) → transferred speech

Style injection via Adaptive Instance Normalisation (AdaIN) at every
decoder layer — modulates scale and shift of normalised activations.

Reference:
    Chou, J. C. et al. (2019). One-shot Voice Conversion by Separating
    Speaker and Content Representations. Interspeech.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def reparametrize(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)


class ContentEncoder(nn.Module):
    def __init__(self, nc=16, cdim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, nc, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(nc, nc*2, 4, stride=2, padding=1), nn.InstanceNorm2d(nc*2), nn.ReLU(),
            nn.Conv2d(nc*2, cdim, 1))
    def forward(self, x): return self.enc(x)


class StyleEncoder(nn.Module):
    def __init__(self, nc=16, sdim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, nc, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(nc, nc*2, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(nc*2, nc*4, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.mu_fc  = nn.Linear(nc*4, sdim)
        self.lv_fc  = nn.Linear(nc*4, sdim)
        self.sdim   = sdim
    def forward(self, x):
        h = self.conv(x).mean([2, 3])
        return self.mu_fc(h), self.lv_fc(h)


class AdaINResBlock(nn.Module):
    def __init__(self, c, sdim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(c, affine=False)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.proj = nn.Linear(sdim, c * 2)
    def forward(self, h, zs):
        g, b = self.proj(zs).chunk(2, dim=1)
        normed = self.norm(h)
        out = g.view(*g.shape, 1, 1) * normed + b.view(*b.shape, 1, 1)
        return h + self.conv(F.relu(out))


class Decoder(nn.Module):
    def __init__(self, cdim=64, sdim=32, nc=16):
        super().__init__()
        self.init = nn.Conv2d(cdim, nc*4, 1)
        self.adain1 = AdaINResBlock(nc*4, sdim)
        self.adain2 = AdaINResBlock(nc*4, sdim)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nc*4, nc*2, 3, padding=1), nn.InstanceNorm2d(nc*2), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nc*2, nc,   3, padding=1), nn.InstanceNorm2d(nc),   nn.ReLU(),
            nn.Conv2d(nc, 1, 3, padding=1), nn.Tanh())
        self.sdim = sdim
    def forward(self, zc, zs):
        h = self.init(zc)
        h = self.adain1(h, zs)
        h = self.adain2(h, zs)
        return self.up(h)


class DisentangledVAE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        cdim = cfg.get("latent_dim", 64)
        sdim = cfg.get("style_dim",  32)
        nc   = cfg.get("base_ch",    16)
        self.ce = ContentEncoder(nc=nc, cdim=cdim)
        self.se = StyleEncoder(nc=nc, sdim=sdim)
        self.de = Decoder(cdim=cdim, sdim=sdim, nc=nc)
        self.sdim = sdim

    def forward(self, x):
        zc = self.ce(x)
        mu, lv = self.se(x)
        zs = reparametrize(mu, lv)
        return self.de(zc, zs), mu, lv

    def transfer_style(self, x_content: torch.Tensor, style_refs: list) -> torch.Tensor:
        zc = self.ce(x_content)
        mus = [self.se(r)[0] for r in style_refs]
        zs  = torch.stack(mus).mean(0)
        return self.de(zc, zs)


def vae_loss(x, xhat, mu, lv, beta=0.5):
    recon = F.l1_loss(xhat, x)
    kl    = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return {"total": recon + beta * kl, "recon": recon, "kl": kl}
