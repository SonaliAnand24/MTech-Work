"""
Data Augmentation for Audio Style Transfer
==========================================
Augmentation strategies applied to log-mel spectrogram chunks
during training to improve model generalisation.

Why augmentation matters here:
  - Small dataset (710 content chunks, 254 style chunks from 1 file each)
  - Without augmentation, GAN discriminators memorise training samples
  - Spectral augmentation (SpecAugment-style) is the most effective
    approach for speech — pitch shift changes the F0 distribution
    which is exactly what style transfer should learn, not memorise

All augmentations operate on normalised log-mel spectrograms (float32 tensors).
They are stochastic — applied with a probability p per sample.

Classes:
  PitchShiftAugment     → shift mel bands (simulate F0 change)
  TimeStretchAugment    → warp time axis (simulate speaking rate)
  SpecAugment           → mask frequency/time bands (regularise)
  AddNoiseAugment       → Gaussian noise (robustness)
  AugmentationPipeline  → compose multiple augmentations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import random


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseAugment:
    """Base class for all augmentations."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying this augmentation per sample.
        """
        assert 0.0 <= p <= 1.0, "Probability must be in [0, 1]"
        self.p = p

    def apply(self, spec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.apply(spec)
        return spec


# ---------------------------------------------------------------------------
# Pitch Shift (mel band shift)
# ---------------------------------------------------------------------------

class PitchShiftAugment(BaseAugment):
    """
    Simulate pitch shift by rolling the mel frequency axis.

    In log-mel space, a pitch shift corresponds to translating the
    spectrogram vertically (shifting mel bands up or down). This is
    an approximation — true pitch shifting requires phase manipulation —
    but it is computationally cheap and effective as a regulariser.

    Application: helps the model not over-fit to the specific F0
    of the training speaker. The content speaker (211.8 Hz) and style
    speaker (176.7 Hz) differ by 35 Hz ≈ ~3–4 mel bands at that range.

    Args:
        max_shift: Maximum number of mel bands to shift (±).
        p:         Probability of applying.
    """

    def __init__(self, max_shift: int = 4, p: float = 0.4):
        super().__init__(p)
        self.max_shift = max_shift

    def apply(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Log-mel spectrogram (n_mels, T) or (1, n_mels, T).
        Returns:
            Pitch-shifted spectrogram, same shape.
        """
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return spec
        # Roll along mel (frequency) axis
        if spec.dim() == 2:
            return torch.roll(spec, shift, dims=0)
        elif spec.dim() == 3:
            return torch.roll(spec, shift, dims=1)
        else:
            return torch.roll(spec, shift, dims=-2)


# ---------------------------------------------------------------------------
# Time Stretch (temporal warping)
# ---------------------------------------------------------------------------

class TimeStretchAugment(BaseAugment):
    """
    Simulate speaking rate variation by warping the time axis.

    Implemented via linear interpolation of spectrogram frames.
    A stretch factor > 1 slows the speech (more frames); < 1 speeds it up.
    The output is resampled back to the original length (128 frames) using
    adaptive pooling so chunk size stays consistent.

    Application: Kalam's speaking rate differs from the content speaker
    (voiced ratio 50.4% vs 70.9% — more pauses). This augmentation
    prevents the model from learning a fixed temporal alignment.

    Args:
        stretch_range: (min, max) stretch factor. Default (0.8, 1.2) = ±20%.
        p:             Probability of applying.
    """

    def __init__(self, stretch_range=(0.85, 1.15), p: float = 0.3):
        super().__init__(p)
        self.stretch_range = stretch_range

    def apply(self, spec: torch.Tensor) -> torch.Tensor:
        factor = random.uniform(*self.stretch_range)
        if abs(factor - 1.0) < 0.01:
            return spec

        orig_shape = spec.shape
        # Work in (1, 1, n_mels, T) for grid_sample
        if spec.dim() == 2:
            s = spec.unsqueeze(0).unsqueeze(0)
        elif spec.dim() == 3:
            s = spec.unsqueeze(0)
        else:
            s = spec

        # Interpolate along time axis
        new_T = max(1, int(s.shape[-1] * factor))
        s_stretched = F.interpolate(s, size=(s.shape[-2], new_T),
                                     mode='bilinear', align_corners=False)

        # Resample back to original T
        s_out = F.adaptive_avg_pool2d(s_stretched, (s.shape[-2], s.shape[-1]))

        return s_out.view(orig_shape)


# ---------------------------------------------------------------------------
# SpecAugment (frequency + time masking)
# ---------------------------------------------------------------------------

class SpecAugment(BaseAugment):
    """
    SpecAugment: random masking of frequency bands and time frames.

    Adapted from Park et al. (2019) for spectrogram-domain style transfer.
    Masks force the model to learn robust representations that don't
    depend on specific frequency bands or time positions.

    Two masking modes:
        Frequency masking: zero out F consecutive mel bands.
        Time masking:      zero out T consecutive frames.

    Args:
        freq_mask_param: Max mel bands to mask (F). Default 10 (=12.5% of 80).
        time_mask_param: Max frames to mask (T). Default 20 (=15.6% of 128).
        num_freq_masks:  Number of frequency masks to apply.
        num_time_masks:  Number of time masks to apply.
        mask_value:      Value to fill masked positions (0.0 = silence in normalised spec).
        p:               Probability of applying.

    Reference:
        Park, D. S. et al. (2019). SpecAugment: A Simple Data Augmentation
        Method for Automatic Speech Recognition. Interspeech.
    """

    def __init__(
        self,
        freq_mask_param: int   = 10,
        time_mask_param: int   = 20,
        num_freq_masks:  int   = 2,
        num_time_masks:  int   = 2,
        mask_value:      float = 0.0,
        p:               float = 0.5,
    ):
        super().__init__(p)
        self.F  = freq_mask_param
        self.T  = time_mask_param
        self.nF = num_freq_masks
        self.nT = num_time_masks
        self.mask_value = mask_value

    def apply(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.clone()

        # Determine mel and time dimensions
        if spec.dim() == 2:
            n_mels, n_frames = spec.shape
            freq_dim, time_dim = 0, 1
        elif spec.dim() == 3:
            _, n_mels, n_frames = spec.shape
            freq_dim, time_dim = 1, 2
        else:
            _, _, n_mels, n_frames = spec.shape
            freq_dim, time_dim = 2, 3

        # Frequency masks
        for _ in range(self.nF):
            f  = random.randint(0, min(self.F, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            if freq_dim == 0:
                spec[f0:f0+f, :] = self.mask_value
            elif freq_dim == 1:
                spec[:, f0:f0+f, :] = self.mask_value
            else:
                spec[:, :, f0:f0+f, :] = self.mask_value

        # Time masks
        for _ in range(self.nT):
            t  = random.randint(0, min(self.T, n_frames - 1))
            t0 = random.randint(0, n_frames - t)
            if time_dim == 1:
                spec[:, t0:t0+t] = self.mask_value
            elif time_dim == 2:
                spec[:, :, t0:t0+t] = self.mask_value
            else:
                spec[:, :, :, t0:t0+t] = self.mask_value

        return spec


# ---------------------------------------------------------------------------
# Gaussian Noise
# ---------------------------------------------------------------------------

class AddNoiseAugment(BaseAugment):
    """
    Add zero-mean Gaussian noise to the log-mel spectrogram.

    Simulates microphone noise and recording imperfections.
    Encourages the model to learn speaker-style features that are
    robust to mild additive noise — important since Kalam reference
    audio has varying recording quality.

    Args:
        noise_std: Standard deviation of noise (relative to spec std).
                   Default 0.05 = 5% noise level.
        p:         Probability of applying.
    """

    def __init__(self, noise_std: float = 0.05, p: float = 0.3):
        super().__init__(p)
        self.noise_std = noise_std

    def apply(self, spec: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(spec) * self.noise_std
        return spec + noise


# ---------------------------------------------------------------------------
# Gain Jitter (amplitude scaling)
# ---------------------------------------------------------------------------

class GainJitterAugment(BaseAugment):
    """
    Random amplitude scaling of the spectrogram.

    In log-mel space, gain jitter = adding a constant offset (since
    log(a*x) = log(a) + log(x)). This simulates microphone distance
    variation and recording level differences — common between the
    self-recorded content and archival Kalam references.

    Args:
        gain_range: (min, max) log-scale gain offset in dB. Default ±3 dB.
        p:          Probability of applying.
    """

    def __init__(self, gain_range=(-3.0, 3.0), p: float = 0.3):
        super().__init__(p)
        self.gain_range = gain_range

    def apply(self, spec: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(*self.gain_range)
        return spec + gain   # additive in log space = multiplicative in linear


# ---------------------------------------------------------------------------
# Augmentation Pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    Compose multiple augmentations applied sequentially.

    Each augmentation has its own probability — the pipeline does NOT
    apply all-or-nothing; each transform independently decides whether
    to activate on a given sample.

    Default pipeline for this thesis (empirically tuned):
        1. PitchShift    p=0.4  — most impactful for voice style transfer
        2. SpecAugment   p=0.5  — general regularisation
        3. GainJitter    p=0.3  — recording level variation
        4. AddNoise      p=0.2  — noise robustness
        5. TimeStretch   p=0.2  — speaking rate variation (used sparingly)

    Usage:
        augmenter = AugmentationPipeline.default()
        aug_chunk = augmenter(chunk_tensor)

        # Or custom:
        augmenter = AugmentationPipeline([
            PitchShiftAugment(max_shift=6, p=0.5),
            SpecAugment(freq_mask_param=15, p=0.6),
        ])
    """

    def __init__(self, transforms: List[BaseAugment]):
        self.transforms = transforms

    @classmethod
    def default(cls) -> "AugmentationPipeline":
        """Standard pipeline used in all training runs."""
        return cls([
            PitchShiftAugment(max_shift=4,  p=0.4),
            SpecAugment(freq_mask_param=10, time_mask_param=20, p=0.5),
            GainJitterAugment(gain_range=(-3.0, 3.0), p=0.3),
            AddNoiseAugment(noise_std=0.05, p=0.2),
            TimeStretchAugment(stretch_range=(0.9, 1.1), p=0.2),
        ])

    @classmethod
    def minimal(cls) -> "AugmentationPipeline":
        """Light augmentation — use when dataset is large enough."""
        return cls([
            PitchShiftAugment(max_shift=2, p=0.3),
            SpecAugment(freq_mask_param=8, time_mask_param=15, p=0.4),
        ])

    @classmethod
    def none(cls) -> "AugmentationPipeline":
        """No augmentation — for ablation studies."""
        return cls([])

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            spec = t(spec)
        return spec

    def __repr__(self) -> str:
        lines = ["AugmentationPipeline("]
        for t in self.transforms:
            lines.append(f"  {t.__class__.__name__}(p={t.p})")
        lines.append(")")
        return "\n".join(lines)
