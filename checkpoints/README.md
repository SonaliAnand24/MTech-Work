# checkpoints/

Saved model weights from training runs on:
- **Content:** `GANINP4.mp3` (8 min 49s personal speech recording)  
- **Style:** `APJ_3.mp3` (3 min 9s APJ Abdul Kalam speech)

---

## Saved Weights

| Folder | File | Size | Architecture | Epochs | Final Loss |
|---|---|---|---|---|---|
| `melgan/` | `weights.pt` | 521 KB | MelGAN Generator + PatchDisc | 15 | G=1.327, D=0.215 |
| `cyclegan/` | `weights.pt` | 441 KB | 2×ResNet Generator + 2×PatchDisc | 15 | G=7.510, D=0.463 |
| `melgan_cycle/` | `weights.pt` | 1.1 MB | 2×MelGAN Generator + 2×PatchDisc | 15 | G=10.921, D=0.360 |
| `vae/` | `weights.pt` | 607 KB | ContentEncoder + StyleEncoder + Decoder | 20 | recon=0.292, KL=0.0001 |

> CNN has no checkpoint — it is an optimisation-based method (no learned weights).

---

## Loading a Checkpoint

```python
# CycleGAN
from src.models.cyclegan import CycleGAN
import yaml

with open("configs/cyclegan_config.yaml") as f:
    cfg = yaml.safe_load(f)

model = CycleGAN(cfg, device="cpu")
model.load_checkpoint("checkpoints/cyclegan/weights.pt")

# MelGAN
from src.models.melgan import MelGAN
model = MelGAN(cfg, device="cpu")
model.load_checkpoint("checkpoints/melgan/weights.pt")

# VAE
import torch
from src.models.vae_disentangled import DisentangledVAE
model = DisentangledVAE(cfg["model"])
ckpt  = torch.load("checkpoints/vae/weights.pt", map_location="cpu")
model.load_state_dict(ckpt)
```

Or use the unified inference script:

```bash
python inference.py \
    --model       cyclegan \
    --content     data/raw/self_recordings/GANINP4.mp3 \
    --style_dir   data/raw/kalam_references/ \
    --checkpoint  checkpoints/cyclegan/weights.pt \
    --output      results/audio_samples/my_output.wav
```

---

## Training Notes

These checkpoints are **early-training snapshots** (15–20 epochs, 40 training pairs).
Full convergence requires 100+ epochs with the complete dataset.
See `docs/RESULTS.md` for the improvement trajectory with more data.

To resume training from a checkpoint:

```bash
python src/models/cyclegan.py \
    --config  configs/cyclegan_config.yaml \
    --resume  checkpoints/cyclegan/weights.pt
```
