# Experimental Results

**Content:** GANINP4.mp3 (8 min 49s) · **Style:** APJ_3.mp3 (3 min 9s)  
**Hardware:** CPU · **Framework:** PyTorch 2.12

---

## Dataset Statistics

| Property | Content (GANINP4) | Style (APJ Kalam #3) |
|---|---|---|
| Duration | 528.67s | 189.80s |
| Mean F0 (voiced) | **211.8 Hz** | **176.7 Hz** |
| F0 gap | | **−35 Hz** |
| RMS energy | 0.030 | 0.110 (×3.7) |
| Voiced ratio | 70.9% | 50.4% |
| Chunks (128-frame, 50% overlap) | 710 | 254 |

---

## MCD Results — All 5 Models

| Model | Chunk 1 | Chunk 2 | Chunk 3 | **Mean** | Final D Loss |
|---|---|---|---|---|---|
| CNN | 119.4 | 242.8 | 117.5 | **159.9** | — |
| MelGAN | 694.3 | 577.3 | 625.3 | 632.3 | 0.215 |
| CycleGAN | 555.8 | 440.3 | 518.8 | 504.9 | **0.463** |
| MelGAN-Cycle | 719.4 | 627.3 | 708.8 | 685.2 | 0.360 |
| VAE | 574.8 | 438.2 | 508.3 | 507.1 | — |

*Chunks: idx=10 (~0:25s), idx=350 (~4:47s), idx=680 (~9:14s) from GANINP4.mp3*

---

## GAN Training Dynamics

| Model | Epochs | Final G Loss | Final D Loss | Δ from ideal (0.5) |
|---|---|---|---|---|
| MelGAN | 15 | 1.327 | 0.215 | **0.285** — D dominates |
| CycleGAN | 15 | 7.510 | 0.463 | **0.037** — near equilibrium |
| MelGAN-Cycle | 15 | 10.921 | 0.360 | **0.140** — partial balance |

**Key finding:** CycleGAN achieves the most balanced G/D equilibrium (Δ=0.037). MelGAN without cycle constraint allows discriminator dominance (Δ=0.285), confirming that cycle-consistency is a critical stabiliser in small-data regimes.

---

## VAE Training

| Epoch | Reconstruction (L1) | KL Divergence |
|---|---|---|
| 1 | ~0.43 | ~0.0008 |
| 5 | 0.3695 | 0.0007 |
| 10 | 0.3348 | 0.0007 |
| 15 | 0.3091 | 0.0002 |
| 20 | **0.2920** | **0.0001** |

KL collapse observed at epoch 15 (posterior → prior). Prescribed fix: KL annealing (β warmup from 0 → 4 over first 30 epochs).

---

## Figures

| Figure | Description |
|---|---|
| `all5_master_comparison.png` | Side-by-side spectrograms: content, style, all 5 outputs |
| `all5_mcd_comparison.png` | MCD bar chart per model and chunk |
| `all5_models_loss_curves.png` | All 5 loss curves |
| `gan_discriminator_convergence.png` | D loss convergence, 3 GAN variants |
| `cnn_spectrogram_grid.png` | CNN: 3 chunks × content/style/output |
| `melgan_spectrogram_grid.png` | MelGAN: same |
| `cyclegan_spectrogram_grid.png` | CycleGAN: same |
| `melgan_cycle_spectrogram_grid.png` | MelGAN-Cycle: same |
| `vae_spectrogram_grid.png` | VAE: same |

