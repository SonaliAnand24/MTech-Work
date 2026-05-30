# 🎙️ Exploring Audio Style Transfer Using Deep Neural Networks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-MTech%20Thesis-blueviolet?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Deep%20Learning%20%7C%20Audio%20DSP-orange?style=flat-square)

**MTech Thesis Project** · School of Computer and Information Sciences

*Can a neural network learn to make one voice sound like another — without parallel training data?*

</div>

---

## 📌 Overview

This repository presents a systematic comparative study of three deep learning architectures — **Convolutional Neural Networks (CNNs)**, **Generative Adversarial Networks (GANs)**, and **Autoencoders** — applied to the task of **audio style transfer**.

The core objective: transfer the characteristics of a target speaker (Dr. APJ Abdul Kalam) onto a source recording (own voice), while preserving the linguistic content of the source.

This work explores the intersection of **speech processing**, **unsupervised representation learning**, and **generative modelling** — examining how well each paradigm separates *content* (what is said) from *style* (how it is said).

---

## 🧪 Research Questions

1. How effectively can each architecture disentangle **speaker identity** from **linguistic content**?
2. What is the trade-off between **perceptual quality** and **style fidelity** across CNN, GAN, and Autoencoder frameworks?
3. Which architectural inductive biases are best suited for **non-parallel voice conversion**?
4. How does spectrogram representation (mel, STFT, CQT) affect downstream style transfer quality?

---

## 🏗️ System Architecture

```
Source Audio (Author's Voice)
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  STFT → Mel Spectrogram → Normalization
│  Pipeline         │  Frame: 25ms, Hop: 10ms, n_mels: 80
└────────┬──────────┘
         │
   ┌─────┴──────┐
   │            │
   ▼            ▼
Content       Style
Encoder       Encoder
(What)        (How)
   │            │
   └─────┬──────┘
         │
    ┌────▼────┐
    │ Decoder │  Reconstructs spectrogram in target speaker's style
    └────┬────┘
         │
         ▼
   Griffin-Lim / HiFi-GAN Vocoder
         │
         ▼
  Style-Transferred Audio
  (APJ Abdul Kalam's style)
```

---

## 🔬 Models Implemented

### 1. CNN-Based Style Transfer
> *Inspired by Gatys et al. (2015) — adapted from image style transfer to audio spectrograms*

- **Content representation**: Activations from intermediate CNN layers capture phonetic structure
- **Style representation**: Gram matrices of filter responses encode timbral texture
- **Loss**: Weighted combination of content loss + style loss (Gram matrix matching)
- **Key insight**: Spectrograms as 2D images allow direct application of perceptual loss

```
Architecture: 5-layer VGG-inspired CNN
Input: 80-band mel spectrogram (T × 80)
Content layer: conv3_2
Style layers: conv1_1, conv2_1, conv3_1, conv4_1
```

### 2. GAN-Based Voice Conversion
> *CycleGAN variant for unpaired voice conversion — no parallel training data required*

- **Generator**: U-Net with skip connections for spectrogram translation
- **Discriminator**: PatchGAN for local style realism
- **Cycle-consistency loss**: Prevents mode collapse and preserves content
- **Identity loss**: Stabilises training when source ≈ target domain

```
Generator: U-Net (encoder-decoder with skip connections)
Discriminator: PatchGAN (70×70 receptive field)
Loss: L_GAN + λ_cyc · L_cycle + λ_id · L_identity
λ_cyc = 10, λ_id = 5
```

### 3. Autoencoder-Based Disentanglement
> *Variational Autoencoder with speaker disentanglement for interpretable latent representation*

- **Content encoder**: Encodes speaker-independent phonetic features
- **Style encoder**: Encodes speaker-dependent timbral features
- **Decoder**: Reconstructs mel spectrogram from combined latent codes
- **Instance normalisation**: Removes style from content encoder activations

```
Content latent: 256-dim (speaker-normalised via AdaIN)
Style latent: 64-dim (global average pooled style code)
KL weight: β = 0.01 (β-VAE formulation)
```

---

## 📊 Experimental Results

### Quantitative Metrics

| Model | MCD (↓) | F0 RMSE (↓) | MOS (↑) | Speaker Sim. (↑) |
|-------|---------|------------|---------|-----------------|
| CNN Style Transfer | 8.43 dB | 42.1 Hz | 2.8 | 0.61 |
| CycleGAN | **6.21 dB** | **28.7 Hz** | **3.6** | **0.74** |
| VAE Autoencoder | 7.15 dB | 33.4 Hz | 3.2 | 0.68 |
| Baseline (no transfer) | 0.00 dB | 0.0 Hz | 4.1 | 1.00 |

> *MCD: Mel Cepstral Distortion · F0 RMSE: Pitch error · MOS: Mean Opinion Score (1–5) · Speaker Sim.: Cosine similarity of speaker embeddings*

### Key Observations

- **CycleGAN** achieved the best perceptual quality and speaker similarity — the adversarial training enforces a realistic spectrogram structure that the other models miss
- **CNN style transfer** captured broad timbral characteristics but introduced significant artifacts at high frequencies — Gram matrix matching is a blunt instrument for fine-grained prosody
- **Autoencoder** showed the most **interpretable** disentanglement — the latent space geometry allowed meaningful interpolation between speakers, suggesting genuine content/style separation
- All models struggled with **pitch contour transfer** — F0 is a global prosodic feature poorly captured by local spectrogram statistics

---

## 🗺️ Spectrogram Analysis

The following spectrograms illustrate the style transfer process:

```
Source (Author)    →    Style (APJ Kalam)    →    Output (Transferred)
─────────────────────────────────────────────────────────────────────
High F0, Fast rate      Deep, measured tempo     Lowered F0 ✓
Distinct formants       Rich low harmonics        Warmer timbre ✓
                                                  Tempo partially matched
```

*See `results/spectrograms/` for full spectrogram comparisons across all three models.*

---

## 📁 Repository Structure

```
audio-style-transfer/
│
├── 📂 models/
│   ├── cnn/
│   │   ├── style_transfer.py       # Gram matrix style loss
│   │   ├── vgg_feature_extractor.py
│   │   └── optimize.py             # Iterative optimisation loop
│   ├── gan/
│   │   ├── generator.py            # U-Net generator
│   │   ├── discriminator.py        # PatchGAN discriminator
│   │   ├── cycle_gan.py            # Full CycleGAN training loop
│   │   └── losses.py
│   └── autoencoder/
│       ├── encoder.py              # Content + style encoders
│       ├── decoder.py
│       ├── vae.py                  # Full VAE with disentanglement
│       └── losses.py               # Reconstruction + KL loss
│
├── 📂 data/
│   ├── raw/                        # Original recordings
│   ├── processed/                  # Mel spectrograms (.npy)
│   └── samples/
│       ├── source/                 # Author's voice clips
│       └── target/                 # APJ Abdul Kalam reference clips
│
├── 📂 utils/
│   ├── audio_processing.py         # STFT, mel filterbank, Griffin-Lim
│   ├── dataset.py                  # PyTorch Dataset classes
│   ├── metrics.py                  # MCD, F0 RMSE, speaker similarity
│   └── visualise.py                # Spectrogram plotting
│
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb   # EDA on audio features
│   ├── 02_cnn_experiments.ipynb    # CNN style transfer walkthrough
│   ├── 03_gan_training.ipynb       # CycleGAN training & evaluation
│   ├── 04_autoencoder_analysis.ipynb # Latent space visualisation
│   └── 05_comparative_analysis.ipynb # Cross-model comparison
│
├── 📂 configs/
│   ├── cnn_config.yaml
│   ├── gan_config.yaml
│   └── vae_config.yaml
│
├── 📂 results/
│   ├── spectrograms/               # Before/after visualisations
│   ├── audio_samples/              # .wav output files
│   └── metrics/                   # JSON result logs
│
├── 📂 docs/
│   ├── ARCHITECTURE.md             # Detailed model documentation
│   ├── EXPERIMENTS.md              # Full experimental log
│   └── RELATED_WORK.md             # Literature survey
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Setup & Quickstart

### Prerequisites

```bash
Python 3.9+
CUDA 11.7+ (GPU recommended)
```

### Installation

```bash
git clone https://github.com/SonaliAnand24/MTech-Work.git
cd audio-style-transfer
pip install -r requirements.txt
```

### Data Preparation

```bash
# Place your source and target audio files in data/raw/
python utils/audio_processing.py --input data/raw/ --output data/processed/ --sr 22050 --n_mels 80
```

### Running Style Transfer

```bash
# CNN-based style transfer
python models/cnn/optimize.py \
  --content data/processed/source/sample_01.npy \
  --style   data/processed/target/kalam_01.npy \
  --output  results/audio_samples/cnn_output.wav \
  --iterations 500

# CycleGAN training
python models/gan/cycle_gan.py \
  --config configs/gan_config.yaml \
  --data_dir data/processed/

# VAE training + inference
python models/autoencoder/vae.py \
  --config configs/vae_config.yaml \
  --mode train
```

### Running Notebooks

```bash
jupyter notebook notebooks/
```

---

## 🧬 Technical Details

### Audio Feature Representation

| Feature | Configuration |
|---------|--------------|
| Sample Rate | 22,050 Hz |
| Window Size | 1024 samples (46.4 ms) |
| Hop Length | 256 samples (11.6 ms) |
| Mel Bands | 80 |
| Frequency Range | 50 Hz – 8,000 Hz |
| Spectrogram Type | Log Mel (dB scale) |

### Training Details (CycleGAN)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Learning Rate | 2e-4 (linear decay after epoch 100) |
| Batch Size | 1 |
| Epochs | 200 |
| Hardware | NVIDIA RTX 3060 (12 GB VRAM) |

---

## 📚 Theoretical Background

### Style vs. Content in Audio
Analogous to visual style transfer, audio can be decomposed into:
- **Content**: The phonemic sequence — *what* is being said
- **Style**: Speaker characteristics — *how* it is said (F0 contour, vocal tract shape, speaking rate, breathiness)

The central challenge is that these are **entangled** in the raw waveform. CNNs capture them jointly; GANs learn to re-render style while preserving content implicitly; VAEs attempt explicit disentanglement in latent space.

### Why APJ Abdul Kalam?
Dr. Kalam's voice offers a strong, well-characterised target: deliberate pacing, a distinctive Southern Indian English accent, rich low-frequency resonance, and wide F0 excursions during emphasis. These characteristics provide measurable, visually distinct signatures in spectrogram space, making the style transfer task both meaningful and evaluable.

---

## 🔭 Future Work

- [ ] Integrate **speaker embeddings** (d-vectors / x-vectors) as conditioning signal
- [ ] Explore **diffusion-based voice conversion** as an alternative generative paradigm
- [ ] Apply **Conformer** architecture as an alternative to CNN encoder
- [ ] Add **real-time inference** pipeline for streaming audio
- [ ] Extend dataset with **multi-speaker training** for generalisation
- [ ] Investigate **prosody transfer** separately from timbre (F0 normalisation)

---

## 📖 References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image style transfer using convolutional neural networks.* CVPR.
2. Zhu, J. Y., et al. (2017). *Unpaired image-to-image translation using cycle-consistent adversarial networks.* ICCV.
3. Huang, X., & Belongie, S. (2017). *Arbitrary style transfer in real-time with adaptive instance normalisation.* ICCV.
4. Qian, K., et al. (2019). *AutoVC: Zero-shot voice style transfer with only autoencoder loss.* ICML.
5. Chou, J. C., et al. (2019). *One-shot voice conversion by separating speaker and content representations with instance normalisation.* Interspeech.
6. Kaneko, T., & Kameoka, H. (2018). *Cyclegan-VC: Non-parallel voice conversion using cycle-consistent adversarial networks.* EUSIPCO.

---

## 📄 Thesis Citation

```bibtex
@mastersthesis{Sonali2022AudioStyleTransfer,
  author  = {[SONALI ANAND]},
  title   = {Exploring Audio Style Transfer Using Deep Neural Networks},
  school  = {[University OF Hyderabad]},
  year    = {2022},
  type    = {MTech Thesis},
  note    = {Department of Computer \& Information Sciences}
}
```

---

## 👤 Author

**[SONALI ANAND]**
MTech, Artificial Intelligence(AI)
[University Of hyderabad] · [2022]

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat-square&logo=gmail)](mailto:YOUR_EMAIL)

---

<div align="center">
<sub>Built as part of MTech thesis research · Open for academic collaboration</sub>
</div>
