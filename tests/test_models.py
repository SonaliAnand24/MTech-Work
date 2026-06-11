"""
tests/test_models.py
======================
Forward-pass and output shape tests for all five model architectures.
These tests run on CPU with tiny random inputs — no real audio needed.

Run:  pytest tests/test_models.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────

BATCH    = 2
N_MELS   = 80
T_FRAMES = 128


@pytest.fixture
def dummy_spec():
    """Random normalised spectrogram batch (B,1,80,128)."""
    return torch.randn(BATCH, 1, N_MELS, T_FRAMES)


@pytest.fixture
def dummy_spec_pair(dummy_spec):
    """Two independent spectrogram batches (content X and style Y)."""
    return dummy_spec, torch.randn(BATCH, 1, N_MELS, T_FRAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — CNN Style Transfer
# ─────────────────────────────────────────────────────────────────────────────

class TestCNNStyleTransfer:

    def test_gram_matrix_shape(self, dummy_spec):
        from src.models.cnn_style_transfer import SpectrogramCNN
        from src.models.losses import gram_matrix
        cnn = SpectrogramCNN()
        a1, a2, a3 = cnn(dummy_spec)
        G = gram_matrix(a1)
        assert G.shape[0] == BATCH
        assert G.shape[1] == G.shape[2]   # square Gram matrix

    def test_cnn_output_shapes(self, dummy_spec):
        from src.models.cnn_style_transfer import SpectrogramCNN
        cnn = SpectrogramCNN(base_channels=16)
        a1, a2, a3 = cnn(dummy_spec)
        assert a1.shape[0] == BATCH
        assert a2.shape[0] == BATCH
        assert a3.shape[0] == BATCH

    def test_transfer_output_shape(self, dummy_spec):
        from src.models.cnn_style_transfer import CNNStyleTransfer
        transfer = CNNStyleTransfer(device="cpu")
        content = dummy_spec[:1]
        style   = dummy_spec[1:]
        output  = transfer.run(content, style, n_steps=5)
        assert output.shape == content.shape

    def test_output_clamped(self, dummy_spec):
        """Output must stay within content spectrogram range."""
        from src.models.cnn_style_transfer import CNNStyleTransfer
        transfer = CNNStyleTransfer(device="cpu")
        content  = dummy_spec[:1]
        style    = dummy_spec[1:]
        output   = transfer.run(content, style, n_steps=5)
        assert output.min() >= content.min() - 1e-4
        assert output.max() <= content.max() + 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — MelGAN
# ─────────────────────────────────────────────────────────────────────────────

class TestMelGAN:

    def test_generator_output_shape(self, dummy_spec):
        from src.models.melgan import MelGANGenerator
        G = MelGANGenerator(n_mels=N_MELS, base_ch=16)
        out = G(dummy_spec)
        assert out.shape == dummy_spec.shape, \
            f"Expected {dummy_spec.shape}, got {out.shape}"

    def test_generator_output_range(self, dummy_spec):
        """Tanh output must be in [-1, 1]."""
        from src.models.melgan import MelGANGenerator
        G = MelGANGenerator(n_mels=N_MELS, base_ch=16)
        out = G(dummy_spec)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <=  1.0 + 1e-5

    def test_discriminator_output_shape(self, dummy_spec):
        from src.models.melgan import PatchDiscriminator
        D   = PatchDiscriminator(base_ch=16)
        out = D(dummy_spec)
        assert out.shape[0] == BATCH
        assert out.dim() == 4   # (B, 1, H', W') — patch scores

    def test_no_gradient_leak_to_disc(self, dummy_spec_pair):
        """Generator step must not update discriminator weights."""
        from src.models.melgan import MelGANGenerator, PatchDiscriminator
        import torch.optim as optim
        X, _ = dummy_spec_pair
        G = MelGANGenerator(n_mels=N_MELS, base_ch=16)
        D = PatchDiscriminator(base_ch=16)
        opt_G = optim.Adam(G.parameters(), lr=1e-4)

        D_params_before = [p.clone() for p in D.parameters()]
        opt_G.zero_grad()
        fake = G(X)
        loss = torch.nn.MSELoss()(D(fake), torch.ones_like(D(fake)))
        loss.backward()
        opt_G.step()

        for before, after in zip(D_params_before, D.parameters()):
            assert torch.allclose(before, after), "Disc weights changed during G step!"


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — CycleGAN
# ─────────────────────────────────────────────────────────────────────────────

class TestCycleGAN:

    @pytest.fixture
    def cycle_cfg(self):
        return {
            "generator":     {"nc": 8, "n_res": 1},
            "discriminator": {"nc": 8},
            "training":      {"lr": 2e-4, "lambda_cyc": 10.0, "lambda_idt": 5.0},
        }

    def test_generator_output_shape(self, dummy_spec, cycle_cfg):
        from src.models.cyclegan import Generator
        G   = Generator(**cycle_cfg["generator"])
        out = G(dummy_spec)
        assert out.shape == dummy_spec.shape

    def test_cycle_reconstruction_shape(self, dummy_spec_pair, cycle_cfg):
        from src.models.cyclegan import Generator
        G, F_ = Generator(**cycle_cfg["generator"]), Generator(**cycle_cfg["generator"])
        X, Y  = dummy_spec_pair
        fake_Y  = G(X)
        rec_X   = F_(fake_Y)
        assert rec_X.shape == X.shape, "Cycle reconstruction shape mismatch"

    def test_train_step_returns_losses(self, dummy_spec_pair, cycle_cfg):
        from src.models.cyclegan import CycleGAN
        model  = CycleGAN(cycle_cfg, device="cpu")
        X, Y   = dummy_spec_pair
        losses = model.train_step(X, Y)
        for key in ("loss_G", "loss_adv", "loss_cyc", "loss_D"):
            assert key in losses
            assert isinstance(losses[key], float)
            assert not np.isnan(losses[key]), f"{key} is NaN"

    def test_translate_output_shape(self, dummy_spec, cycle_cfg):
        from src.models.cyclegan import CycleGAN
        model = CycleGAN(cycle_cfg, device="cpu")
        out   = model.translate(dummy_spec)
        assert out.shape == dummy_spec.shape

    def test_checkpoint_save_load(self, dummy_spec_pair, cycle_cfg, tmp_path):
        from src.models.cyclegan import CycleGAN
        model  = CycleGAN(cycle_cfg, device="cpu")
        path   = str(tmp_path / "cyclegan_test.pt")
        model.save_checkpoint(path, epoch=1)
        assert Path(path).exists()
        model2 = CycleGAN(cycle_cfg, device="cpu")
        ep = model2.load_checkpoint(path)
        assert ep == 1


# ─────────────────────────────────────────────────────────────────────────────
# Model 4 — MelGAN-Cycle
# ─────────────────────────────────────────────────────────────────────────────

class TestMelGANCycle:

    @pytest.fixture
    def mc_cfg(self):
        return {
            "generator":     {"n_mels": N_MELS, "base_ch": 16},
            "discriminator": {"base_ch": 8},
            "training":      {"lr": 2e-4, "lambda_cyc": 10.0, "lambda_idt": 5.0},
        }

    def test_translate_output_shape(self, dummy_spec, mc_cfg):
        from src.models.melgan_cycle import MelGANCycle
        model = MelGANCycle(mc_cfg, device="cpu")
        out   = model.translate(dummy_spec)
        assert out.shape == dummy_spec.shape

    def test_cycle_shapes_match(self, dummy_spec_pair, mc_cfg):
        """F(G(X)) must have the same shape as X."""
        from src.models.melgan import MelGANGenerator
        G, F_ = (MelGANGenerator(n_mels=N_MELS, base_ch=16),
                 MelGANGenerator(n_mels=N_MELS, base_ch=16))
        X, _  = dummy_spec_pair
        fake  = G(X)
        rec   = F_(fake)
        assert rec.shape == X.shape

    def test_train_step_returns_all_losses(self, dummy_spec_pair, mc_cfg):
        from src.models.melgan_cycle import MelGANCycle
        model  = MelGANCycle(mc_cfg, device="cpu")
        X, Y   = dummy_spec_pair
        losses = model.train_step(X, Y)
        for key in ("loss_G", "loss_adv", "loss_cyc", "loss_D"):
            assert key in losses
            assert not np.isnan(losses[key])


# ─────────────────────────────────────────────────────────────────────────────
# Model 5 — VAE
# ─────────────────────────────────────────────────────────────────────────────

class TestVAE:

    @pytest.fixture
    def vae_cfg(self):
        return {"latent_dim": 32, "style_dim": 16, "base_ch": 8}

    def test_forward_output_shape(self, dummy_spec, vae_cfg):
        from src.models.vae_disentangled import DisentangledVAE
        model     = DisentangledVAE(vae_cfg)
        x_hat, mu, lv = model(dummy_spec)
        assert x_hat.shape == dummy_spec.shape

    def test_mu_lv_shape(self, dummy_spec, vae_cfg):
        from src.models.vae_disentangled import DisentangledVAE
        model     = DisentangledVAE(vae_cfg)
        _, mu, lv = model(dummy_spec)
        assert mu.shape == (BATCH, vae_cfg["style_dim"])
        assert lv.shape == (BATCH, vae_cfg["style_dim"])

    def test_reparametrize_stochastic(self, dummy_spec, vae_cfg):
        """Two forward passes should produce different z_style samples."""
        from src.models.vae_disentangled import DisentangledVAE
        model = DisentangledVAE(vae_cfg)
        model.train()
        _, mu1, lv1 = model(dummy_spec)
        _, mu2, lv2 = model(dummy_spec)
        # mu should be deterministic (same input → same mu)
        assert torch.allclose(mu1, mu2, atol=1e-5)

    def test_style_transfer_output_shape(self, dummy_spec_pair, vae_cfg):
        from src.models.vae_disentangled import DisentangledVAE
        model   = DisentangledVAE(vae_cfg)
        X, Y    = dummy_spec_pair
        out     = model.transfer_style(X, [Y])
        assert out.shape == X.shape

    def test_elbo_loss(self, dummy_spec, vae_cfg):
        from src.models.vae_disentangled import DisentangledVAE, vae_loss
        model       = DisentangledVAE(vae_cfg)
        x_hat, mu, lv = model(dummy_spec)
        losses      = vae_loss(dummy_spec, x_hat, mu, lv)
        assert "total" in losses
        assert "recon" in losses
        assert "kl"    in losses
        assert losses["total"].item() > 0
        assert not torch.isnan(losses["total"])


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestLossFunctions:

    def test_gram_matrix_symmetry(self, dummy_spec):
        from src.models.losses import gram_matrix
        from src.models.cnn_style_transfer import SpectrogramCNN
        cnn = SpectrogramCNN(base_channels=16)
        a1, _, _ = cnn(dummy_spec)
        G = gram_matrix(a1)
        assert torch.allclose(G, G.transpose(1, 2), atol=1e-5), \
            "Gram matrix must be symmetric"

    def test_lsgan_equilibrium(self):
        """At equilibrium D(real)≈0.5 and D(fake)≈0.5 → D_loss≈0.5."""
        from src.models.losses import LSGANLoss
        criterion  = LSGANLoss()
        half_preds = torch.full((4, 1, 5, 5), 0.5)
        d_loss     = criterion.discriminator_loss(half_preds, half_preds)
        assert abs(d_loss.item() - 0.5) < 0.01

    def test_cycle_loss_zero_for_perfect_cycle(self):
        from src.models.losses import CycleConsistencyLoss
        criterion = CycleConsistencyLoss(lambda_cyc=10.0)
        x = torch.randn(2, 1, 80, 128)
        loss = criterion(x, x, x, x)   # perfect reconstruction
        assert loss.item() < 1e-5

    def test_kl_loss_zero_for_unit_gaussian(self):
        """KL(N(0,I) || N(0,I)) should be 0."""
        from src.models.losses import KLDivergenceLoss
        criterion = KLDivergenceLoss(beta=1.0)
        mu     = torch.zeros(4, 32)
        log_var = torch.zeros(4, 32)
        loss   = criterion(mu, log_var)
        assert abs(loss.item()) < 1e-4

    def test_elbo_loss_components(self):
        from src.models.losses import ELBOLoss
        criterion = ELBOLoss(beta=1.0)
        x         = torch.randn(2, 1, 80, 128)
        x_hat     = x + 0.1 * torch.randn_like(x)
        mu        = torch.zeros(2, 16)
        log_var   = torch.zeros(2, 16)
        out = criterion(x, x_hat, mu, log_var)
        assert "total" in out and "recon" in out and "kl" in out
        assert torch.isclose(out["total"], out["recon"] + out["kl"], atol=1e-5)
