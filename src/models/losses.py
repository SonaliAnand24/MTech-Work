"""
Shared Loss Functions
=====================
All loss functions used across the five models in one place.
Import from here to keep model files clean and consistent.

Losses defined:
  - GramStyleLoss         → CNN style transfer (Gram matrix MSE)
  - ContentLoss           → CNN content preservation (activation MSE)
  - LSGANLoss             → All GAN models (Least Squares GAN)
  - CycleLoss             → CycleGAN + MelGAN-Cycle
  - IdentityLoss          → CycleGAN + MelGAN-Cycle
  - FeatureMatchingLoss   → MelGAN standalone
  - VAELoss               → VAE (reconstruction + KL divergence)
  - SpectralConvergence   → Auxiliary spectrogram quality metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


# ---------------------------------------------------------------------------
# Gram Matrix (shared utility)
# ---------------------------------------------------------------------------

def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix of a CNN feature map.

    Captures the correlation between feature channels — encoding
    the timbral 'texture' of the spectrogram at a given layer.
    This is the mathematical heart of CNN style transfer.

    Args:
        feature_map: (B, C, H, W)
    Returns:
        Gram matrix: (B, C, C), normalised by spatial size
    """
    B, C, H, W = feature_map.shape
    f = feature_map.view(B, C, H * W)
    return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)


# ---------------------------------------------------------------------------
# CNN Style Transfer Losses
# ---------------------------------------------------------------------------

class GramStyleLoss(nn.Module):
    """
    Style loss via Gram matrix MSE.

    Used on shallow CNN layers (layers 0, 1) where correlations
    capture timbral texture rather than spatial content structure.

    L_style = Σ_l w_l * ||G_l(output) - G_l(style)||²_F
    """

    def __init__(self, layer_weights: List[float] = None):
        super().__init__()
        self.layer_weights = layer_weights

    def forward(
        self,
        output_features: List[torch.Tensor],
        style_features:  List[torch.Tensor],
        layers: List[int] = None,
    ) -> torch.Tensor:
        layers = layers or list(range(len(output_features)))
        weights = self.layer_weights or [1.0] * len(layers)
        loss = torch.tensor(0.0, requires_grad=True)
        for i, w in zip(layers, weights):
            G_out   = gram_matrix(output_features[i])
            G_style = gram_matrix(style_features[i])
            loss    = loss + w * F.mse_loss(G_out, G_style)
        return loss


class ContentLoss(nn.Module):
    """
    Content loss via deep-layer activation MSE.

    Used on deeper CNN layers (layer 2+) where activations encode
    spatial and sequential structure — preserving linguistic content.

    L_content = ||A_l(output) - A_l(content)||²
    """

    def forward(
        self,
        output_features:  List[torch.Tensor],
        content_features: List[torch.Tensor],
        layers: List[int] = None,
    ) -> torch.Tensor:
        layers = layers or [len(output_features) - 1]
        return sum(F.mse_loss(output_features[i], content_features[i]) for i in layers)


# ---------------------------------------------------------------------------
# GAN Losses (shared by MelGAN, CycleGAN, MelGAN-Cycle)
# ---------------------------------------------------------------------------

class LSGANLoss(nn.Module):
    """
    Least Squares GAN adversarial loss (Mao et al., 2017).

    More stable than BCE GAN — avoids vanishing gradients when
    discriminator is confident. Ideal equilibrium: D_loss ≈ 0.5.

    Real target = 1.0, Fake target = 0.0
    L_D = 0.5 * [E(D(real) - 1)² + E(D(fake))²]
    L_G = E(D(fake) - 1)²

    Reference:
        Mao, X. et al. (2017). Least Squares Generative Adversarial
        Networks. ICCV.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def discriminator_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Combined real + fake discriminator loss (×0.5 each)."""
        return 0.5 * (
            self.mse(pred_real, torch.ones_like(pred_real)) +
            self.mse(pred_fake, torch.zeros_like(pred_fake))
        )

    def generator_loss(self, pred_fake: torch.Tensor) -> torch.Tensor:
        """Generator adversarial loss — fool the discriminator."""
        return self.mse(pred_fake, torch.ones_like(pred_fake))


# ---------------------------------------------------------------------------
# CycleGAN / MelGAN-Cycle Losses
# ---------------------------------------------------------------------------

class CycleLoss(nn.Module):
    """
    Cycle-consistency loss.

    Enforces that translating X → Y → X recovers the original X.
    This is the key constraint that preserves linguistic content
    in unpaired voice conversion.

    L_cyc = E[||F(G(X)) - X||₁ + ||G(F(Y)) - Y||₁]

    L1 (not L2) is used because it is more robust to occasional
    large errors in spectrogram reconstruction.

    Args:
        lambda_cyc: Weight multiplier (default 10.0, from Zhu et al.)
    """

    def __init__(self, lambda_cyc: float = 10.0):
        super().__init__()
        self.lambda_cyc = lambda_cyc
        self.l1 = nn.L1Loss()

    def forward(
        self,
        rec_X:  torch.Tensor,   # F(G(real_X))
        real_X: torch.Tensor,
        rec_Y:  torch.Tensor,   # G(F(real_Y))
        real_Y: torch.Tensor,
    ) -> torch.Tensor:
        return self.lambda_cyc * (self.l1(rec_X, real_X) + self.l1(rec_Y, real_Y))


class IdentityLoss(nn.Module):
    """
    Identity mapping loss.

    Regularises generators: G(Y) ≈ Y and F(X) ≈ X.
    Prevents colour (timbre) shift when the input is already
    in the target domain. Critical for eliminating pitch warbling
    artefacts in long vowels — observed in CycleGAN v2 without it.

    L_idt = λ_idt * [||G(Y) - Y||₁ + ||F(X) - X||₁]

    Args:
        lambda_idt: Weight (default 5.0 = half of lambda_cyc)
    """

    def __init__(self, lambda_idt: float = 5.0):
        super().__init__()
        self.lambda_idt = lambda_idt
        self.l1 = nn.L1Loss()

    def forward(
        self,
        idt_Y:  torch.Tensor,   # G(real_Y)
        real_Y: torch.Tensor,
        idt_X:  torch.Tensor,   # F(real_X)
        real_X: torch.Tensor,
    ) -> torch.Tensor:
        return self.lambda_idt * (self.l1(idt_Y, real_Y) + self.l1(idt_X, real_X))


# ---------------------------------------------------------------------------
# MelGAN Feature Matching Loss
# ---------------------------------------------------------------------------

class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for MelGAN standalone training.

    In the absence of cycle-consistency, this L1 loss between the
    real content spectrogram and the generated output provides a
    soft content preservation signal. It is weaker than cycle loss —
    this is the key limitation of MelGAN vs CycleGAN in small-data
    regimes, as demonstrated in Chapter 4 of this thesis.

    L_fm = λ_fm * ||G(X) - X||₁

    Note: This encourages identity mapping (output ≈ input) rather
    than true domain translation. A proper feature-matching loss
    would compare intermediate discriminator layer activations, which
    requires a multi-scale discriminator (planned for future work).

    Args:
        lambda_fm: Weight multiplier (default 2.0)
    """

    def __init__(self, lambda_fm: float = 2.0):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.l1 = nn.L1Loss()

    def forward(
        self,
        fake_Y: torch.Tensor,   # G(real_X)
        real_X: torch.Tensor,   # content reference
    ) -> torch.Tensor:
        return self.lambda_fm * self.l1(fake_Y, real_X)


# ---------------------------------------------------------------------------
# VAE Loss (ELBO)
# ---------------------------------------------------------------------------

class VAELoss(nn.Module):
    """
    VAE Evidence Lower Bound (ELBO) loss.

    L_ELBO = E[||x - x̂||₁]  +  β * D_KL(q(z|x) || p(z))

    Components:
        Reconstruction: L1 loss between input and reconstruction.
            L1 preferred over L2 for spectrograms — more robust to
            outlier frames (clicks, transients).

        KL divergence: Closed-form for diagonal Gaussian.
            D_KL = -0.5 * Σ(1 + log σ² - μ² - σ²)

    Beta-VAE (β > 1):
        Higher β encourages more disentangled latent representations
        at the cost of reconstruction quality. Optimal β=4.0 found
        by grid search (see docs/RESEARCH_NOTES.md).

    KL Collapse (observed at epoch 15 in small-data runs):
        When the encoder learns to ignore input and output prior
        N(0,I), KL → 0 and style code becomes uninformative.
        Fix: KL annealing — ramp β from 0 → beta_max over warmup epochs.

    Args:
        beta:         KL weight (default 0.5 for training stability)
        beta_max:     Maximum beta for annealing schedule
        warmup_steps: Steps over which to linearly ramp beta
    """

    def __init__(
        self,
        beta:         float = 0.5,
        beta_max:     float = 4.0,
        warmup_steps: int   = 0,    # 0 = no annealing (fixed beta)
    ):
        super().__init__()
        self.beta         = beta
        self.beta_max     = beta_max
        self.warmup_steps = warmup_steps
        self._step        = 0

    def get_beta(self) -> float:
        """Return current beta value (with annealing if configured)."""
        if self.warmup_steps == 0:
            return self.beta
        progress = min(1.0, self._step / self.warmup_steps)
        return self.beta_max * progress

    def forward(
        self,
        x:      torch.Tensor,   # input spectrogram
        x_hat:  torch.Tensor,   # reconstructed spectrogram
        mu:     torch.Tensor,   # style posterior mean
        log_var:torch.Tensor,   # style posterior log-variance
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with 'total', 'recon', 'kl' for logging.
        """
        recon = F.l1_loss(x_hat, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        beta  = self.get_beta()
        total = recon + beta * kl
        self._step += 1
        return {"total": total, "recon": recon, "kl": kl, "beta": beta}


# ---------------------------------------------------------------------------
# Auxiliary: Spectral Convergence
# ---------------------------------------------------------------------------

class SpectralConvergenceLoss(nn.Module):
    """
    Spectral convergence loss — measures how well the magnitude spectrum
    of the output matches the target.

    L_sc = ||S_target - S_output||_F / ||S_target||_F

    Used as an auxiliary metric (not in primary training objective)
    to diagnose whether models are capturing spectral envelope shape.

    Lower is better. Value of 0 = perfect spectral match.
    """

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        num  = torch.norm(target - output, p="fro")
        denom= torch.norm(target, p="fro") + 1e-8
        return num / denom
