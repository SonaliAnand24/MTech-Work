"""Audio preprocessing: raw MP3/WAV → normalised log-mel spectrogram chunks."""

import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

SR      = 22050
N_FFT   = 1024
HOP     = 256
N_MELS  = 80
FMIN    = 50
FMAX    = 8000


def load_audio(path: str, sr: int = SR, trim: bool = True) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if trim:
        y, _ = librosa.effects.trim(y, top_db=30)
    mx = np.abs(y).max()
    return (y / mx).astype(np.float32) if mx > 0 else y.astype(np.float32)


def audio_to_logmel(y: np.ndarray, sr=SR, n_fft=N_FFT, hop=HOP,
                    n_mels=N_MELS, fmin=FMIN, fmax=FMAX) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                          n_mels=n_mels, fmin=fmin, fmax=fmax)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def normalise(spec: np.ndarray, mean=None, std=None):
    mean = mean if mean is not None else spec.mean()
    std  = (std  if std  is not None else spec.std()) + 1e-8
    return ((spec - mean) / std).astype(np.float32), mean, std


def chunk(spec: np.ndarray, frames=128, stride=64) -> np.ndarray:
    segs = [spec[:, s:s+frames] for s in range(0, spec.shape[1]-frames+1, stride)]
    return np.stack(segs) if segs else np.empty((0, spec.shape[0], frames))


def load_spectrogram(path: str, **kw) -> torch.Tensor:
    """End-to-end: audio file → (1,1,n_mels,T) tensor."""
    y    = load_audio(path)
    spec = audio_to_logmel(y, **{k: v for k, v in kw.items()
                                  if k in ('sr','n_fft','hop','n_mels','fmin','fmax')})
    spec, _, _ = normalise(spec)
    return torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)


def preprocess_directory(input_dir: str, output_dir: str,
                          chunk_frames: int = 128, **kw) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    files = list(Path(input_dir).rglob("*.mp3")) + list(Path(input_dir).rglob("*.wav"))
    logger.info(f"Processing {len(files)} files from {input_dir}")
    for f in files:
        try:
            y    = load_audio(str(f))
            spec = audio_to_logmel(y)
            spec, _, _ = normalise(spec)
            chunks = chunk(spec, chunk_frames)
            np.save(str(out / (f.stem + ".npy")), chunks)
            logger.info(f"  {f.name} → {chunks.shape}")
        except Exception as e:
            logger.warning(f"  Skipped {f.name}: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",    required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--chunk_frames", type=int, default=128)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    preprocess_directory(a.input_dir, a.output_dir, a.chunk_frames)
