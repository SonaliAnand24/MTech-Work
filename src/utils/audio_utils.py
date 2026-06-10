"""Spectrogram ↔ audio conversion utilities."""

import numpy as np
import librosa
import soundfile as sf

SR     = 22050
N_FFT  = 1024
HOP    = 256
N_MELS = 80
FMIN   = 50
FMAX   = 8000


def spectrogram_to_audio(spec_norm: np.ndarray, mean: float = -57.0, std: float = 15.0,
                          n_iter: int = 32) -> np.ndarray:
    """Denormalise log-mel spectrogram → waveform via Griffin-Lim."""
    spec_db = spec_norm * std + mean
    power   = librosa.db_to_power(spec_db)
    return librosa.feature.inverse.mel_to_audio(
        power, sr=SR, n_fft=N_FFT, hop_length=HOP,
        fmin=FMIN, fmax=FMAX, n_iter=n_iter)


def save_audio(audio: np.ndarray, path: str, sr: int = SR):
    sf.write(path, audio, samplerate=sr)
