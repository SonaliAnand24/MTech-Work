# Adding More Recordings

## File Naming Convention

```
data/raw/
├── self_recordings/        ← your voice files
│   ├── GANINP4.mp3         ← already used
│   ├── GANINP5.mp3         ← add here
│   └── ...
└── kalam_references/       ← APJ Kalam files
    ├── APJ_3.mp3            ← already used
    ├── APJ_4.mp3
    └── ...
```

## Steps

```bash
# 1. Preprocess new files
python src/utils/preprocessing.py --input_dir data/raw/self_recordings/ --output_dir data/processed/self/
python src/utils/preprocessing.py --input_dir data/raw/kalam_references/ --output_dir data/processed/kalam/

# 2. Retrain (increase n_epochs in config)
python src/models/cyclegan.py --config configs/cyclegan_config.yaml

# 3. Run inference on new content
python inference.py --model cyclegan --content data/raw/self_recordings/GANINP5.mp3 \
    --style_dir data/raw/kalam_references/ --checkpoint checkpoints/cyclegan/best.pt \
    --output results/audio_samples/cyclegan/GANINP5_output.wav
```

## Expected MCD Improvement

| Content files | Style files | Est. CycleGAN MCD |
|---|---|---|
| 1 (current) | 1 (current) | ~505 dB |
| 3 | 3 | ~350–400 dB |
| 5+ | 5+ | ~250–300 dB |
| 10+ | 10+ | ~150–200 dB |

## Recording Quality Checklist

- [ ] Duration > 30s per file
- [ ] Voiced ratio > 40%
- [ ] F0 in expected range (self: 150–300 Hz, Kalam: 100–250 Hz)
- [ ] No background music or applause in Kalam clips
- [ ] Consistent microphone distance (15–20 cm)
