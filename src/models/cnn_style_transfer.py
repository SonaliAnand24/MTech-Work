"""
CNN-based Audio Style Transfer — Gram Matrix Optimisation
==========================================================
Adapted from Gatys et al. (2016) for log-mel spectrogram space.

Style is captured by the Gram matrix of CNN feature activations;
content by the activations themselves. A noise spectrogram is
optimised to simultaneously match both.

Reference:
    Gatys, L. A., Ecker, A. S., & Bethge, M. (2016).
    "A Neural Algorithm of Artistic Style." CVPR.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


def gram_matrix(f: torch.Tensor) -> torch.Tensor:
    """Gram matrix of feature map (B,C,H,W) → (B,C,C)."""
    B, C, H, W = f.shape
    x = f.view(B, C, H * W)
    return torch.bmm(x, x.transpose(1, 2)) / (C * H * W)


class SpectrogramCNN(nn.Module):
    """3-layer fixed CNN feature extractor for spectrogram style transfer."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        nc = base_channels
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, nc,    3, padding=1), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(nc,           nc*2, 3, padding=1), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(nc*2,         nc*4, 3, padding=1), nn.ReLU())

    def forward(self, x):
        a1 = self.c1(x)
        a2 = self.c2(a1)
        a3 = self.c3(a2)
        return a1, a2, a3


class CNNStyleTransfer:
    """
    Gram matrix style transfer for audio spectrograms.

    Content layers : [2]   (deep — spatial structure)
    Style layers   : [0,1] (shallow — timbral texture / Gram matrices)

    Usage:
        transfer = CNNStyleTransfer(device="cpu")
        output   = transfer.run(content_spec, style_spec, n_steps=200)
    """

    CONTENT_LAYERS: List[int] = [2]
    STYLE_LAYERS:   List[int] = [0, 1]

    def __init__(self, device="cpu", content_weight=1.0, style_weight=1e5):
        self.device  = torch.device(device)
        self.alpha   = content_weight
        self.beta    = style_weight
        self.cnn     = SpectrogramCNN().to(self.device)
        self.cnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad_(False)

    def _features(self, x, grad=False):
        if grad:
            return self.cnn(x)
        with torch.no_grad():
            return self.cnn(x)

    def run(self, content_spec, style_spec, n_steps=200, lr=0.05, log_every=50):
        content_spec = content_spec.to(self.device)
        style_spec   = style_spec.to(self.device)
        cf = self._features(content_spec)
        sf = self._features(style_spec)
        output = content_spec.clone().requires_grad_(True)
        opt    = optim.Adam([output], lr=lr)

        for step in range(1, n_steps + 1):
            opt.zero_grad()
            of = self._features(output, grad=True)
            lc = sum(nn.functional.mse_loss(of[i], cf[i]) for i in self.CONTENT_LAYERS)
            ls = sum(nn.functional.mse_loss(gram_matrix(of[i]), gram_matrix(sf[i]))
                     for i in self.STYLE_LAYERS)
            loss = self.alpha * lc + self.beta * ls
            loss.backward()
            opt.step()
            with torch.no_grad():
                output.clamp_(content_spec.min(), content_spec.max())
            if step % log_every == 0:
                logger.info(f"Step {step}/{n_steps} | content={lc.item():.4f} "
                            f"style={ls.item():.4f} total={loss.item():.6f}")

        return output.detach()


if __name__ == "__main__":
    import argparse, yaml, soundfile as sf
    from src.utils.preprocessing import load_spectrogram
    from src.utils.audio_utils import spectrogram_to_audio

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/cnn_config.yaml")
    parser.add_argument("--content", required=True)
    parser.add_argument("--style",   required=True)
    parser.add_argument("--output",  default="results/audio_samples/cnn/output.wav")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    content = load_spectrogram(args.content, **cfg["spectrogram"])
    style   = load_spectrogram(args.style,   **cfg["spectrogram"])

    t = CNNStyleTransfer(device=args.device,
                         content_weight=cfg["training"]["content_weight"],
                         style_weight=cfg["training"]["style_weight"])
    out = t.run(content, style, n_steps=cfg["training"]["n_steps"])
    audio = spectrogram_to_audio(out)
    sf.write(args.output, audio, samplerate=22050)
    print(f"Saved → {args.output}")
