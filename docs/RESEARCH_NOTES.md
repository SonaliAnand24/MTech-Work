# Research Notes & Experiment Log

---

## Data Collection (Week 1–2)

- **GANINP4.mp3** — personal speech, ~9 min, condenser mic, quiet room, 44.1kHz → resampled to 22.05kHz
- **APJ_3.mp3** — archival Kalam speech excerpt, public domain, 3 min 9s

**Acoustic observations:**
- Content F0 mean: 211.8 Hz · Style F0 mean: 176.7 Hz → gap of **35 Hz** — measurable, meaningful target
- Kalam's RMS energy is 3.7× higher — projects authority via amplitude, not just pitch
- Voiced ratio: 70.9% (self) vs 50.4% (Kalam) — he speaks more slowly with deliberate pauses

---

## CNN Baseline (Week 3–4)

| Run | Style weight β | Steps | Final loss | MCD |
|---|---|---|---|---|
| 01 | 1e4 | 200 | 0.0021 | ~180 | Too little style |
| 02 | 1e5 | 200 | 0.00013 | 119.4 | ✓ chosen |
| 03 | 1e6 | 200 | diverged | — | Style overwhelms |

**Finding:** β=1e5 is the sweet spot. Gram matrix matching visibly shifts energy toward lower mel bands (Kalam's register). Temporal structure well-preserved. Griffin-Lim introduces metallic chirping.

---

## MelGAN (Week 5)

- Discriminator loss plateaued at **0.215** — generator cannot fool D consistently
- Feature-matching loss alone is insufficient content preservation without cycle constraint
- Generator outputs show domain shift but content drift (linguistic structure partially lost)

**Lesson:** Without cycle-consistency, small-data adversarial training allows discriminator dominance. This is why CycleGAN outperforms MelGAN standalone despite using a simpler 2D architecture.

---

## CycleGAN (Week 6–7)

**v1 failure:** Used BatchNorm → training diverged at epoch 3 (discriminator collapse)  
**v2 fix:** Switched to InstanceNorm + LSGAN loss → stable  
**v3 (final):** Added identity loss λ_idt=5 → eliminated pitch warbling in long vowels

Final D loss: **0.463** — the closest to LSGAN ideal of 0.5 among all GAN variants.
Cycle loss decline: 0.82 → 0.69 over 15 epochs — content preservation improving.

---

## MelGAN-Cycle Hybrid (Week 8 — Novel Contribution)

**Hypothesis:** MelGAN's dilated 1D conv stack models temporal prosody better than CycleGAN's 2D ResNet. Adding cycle-consistency should yield best of both worlds.

**Result:** G loss=10.92 (higher than CycleGAN's 7.51), D loss=0.360.

**Interpretation:** The 1D temporal architecture is architecturally superior for prosody modelling but needs more training samples to leverage it. With 40 pairs, CycleGAN's 2D spatial approach is more data-efficient. This is the highest-priority direction for extended training with 5+ recordings per speaker.

---

## VAE (Week 9–10)

**Option A (just two encoders):** Failed — both encode everything, no disentanglement  
**Option B (adversarial on z_content):** Worked — speaker cannot be predicted from content code  

KL collapse at epoch 15 (lv → 0, posterior → prior). Style code becomes uninformative.

**Fix:** KL annealing — start β=0, linearly ramp to β=4 over first 30 epochs.  
This is standard practice (Bowman et al. 2015) and is the top priority for extended training.

---

## Key Open Questions

1. Does MelGAN-Cycle outperform CycleGAN with 5+ recordings per speaker? (Expected: yes)
2. Does KL-annealed VAE beat CycleGAN on speaker similarity with sufficient data?
3. Is the F0 gap (35 Hz) better addressed by explicit F0 conditioning or implicit GAN learning?
4. HiFi-GAN vocoder — how much of the perceptual quality gap is vocoder vs. model?
