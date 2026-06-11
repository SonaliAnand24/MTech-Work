# data/processed/

Pre-processed log-mel spectrogram chunks generated from the raw audio files.

## Contents

### `self/`  — Content speaker (GANINP4)

| File | Description |
|---|---|
| `GANINP4.npy` | Shape `(710, 80, 128)` — 710 chunks × 80 mel bands × 128 frames |
| `GANINP4_norm_stats.npy` | `[mean, std]` used for normalisation — required for denormalisation at inference |

- Source: `data/raw/self_recordings/GANINP4.mp3`
- Duration after trim: 528.67s @ 22,050 Hz
- Chunk size: 128 frames = ~1.49s per chunk, 50% overlap stride

### `kalam/`  — Style speaker (APJ Abdul Kalam)

| File | Description |
|---|---|
| `APJ_3.npy` | Shape `(254, 80, 128)` — 254 chunks × 80 mel bands × 128 frames |
| `APJ_3_norm_stats.npy` | `[mean, std]` for normalisation |

- Source: `data/raw/kalam_references/APJ_3.mp3`
- Duration after trim: 189.80s @ 22,050 Hz

## Preprocessing Parameters

```
sample_rate  : 22,050 Hz
n_fft        : 1,024
hop_length   : 256
n_mels       : 80
fmin         : 50 Hz
fmax         : 8,000 Hz
chunk_frames : 128
chunk_stride : 64  (50% overlap)
normalisation: z-score per file (mean=0, std=1)
```

## Adding New Recordings

```bash
python src/utils/preprocessing.py \
    --input_dir  data/raw/self_recordings/ \
    --output_dir data/processed/self/

python src/utils/preprocessing.py \
    --input_dir  data/raw/kalam_references/ \
    --output_dir data/processed/kalam/
```

New `.npy` files will be created alongside the existing ones.
All models load all `.npy` files in the directory automatically.

## Note on Git

Large `.npy` files are **not** tracked by git (see `.gitignore`).
To share processed data, either use Git LFS or include the raw audio
files and re-run preprocessing on the target machine.
