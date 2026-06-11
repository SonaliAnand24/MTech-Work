"""
tests/test_preprocessing.py
=============================
Unit tests for src/utils/preprocessing.py

Run:  pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import soundfile as sf
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import (
    load_audio,
    audio_to_logmel,
    normalise,
    chunk,
    load_spectrogram,
    preprocess_directory,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_audio():
    """1 second of 440 Hz sine wave at 22,050 Hz."""
    sr = 22050
    t  = np.linspace(0, 1, sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32), sr


@pytest.fixture
def dummy_wav_file(dummy_audio, tmp_path):
    """Write dummy audio to a temp .wav file, return path."""
    audio, sr = dummy_audio
    path = str(tmp_path / "test_audio.wav")
    sf.write(path, audio, samplerate=sr)
    return path


@pytest.fixture
def dummy_spec(dummy_audio):
    """Log-mel spectrogram from dummy audio."""
    audio, sr = dummy_audio
    return audio_to_logmel(audio, sr=sr)


# ─────────────────────────────────────────────────────────────────────────────
# load_audio
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAudio:

    def test_returns_float32(self, dummy_wav_file):
        audio = load_audio(dummy_wav_file)
        assert audio.dtype == np.float32

    def test_peak_normalised(self, dummy_wav_file):
        audio = load_audio(dummy_wav_file)
        assert np.abs(audio).max() <= 1.0 + 1e-5

    def test_correct_sample_rate(self, dummy_wav_file):
        import librosa
        audio = load_audio(dummy_wav_file, sr=16000)
        # Duration should still be ~1s
        assert 0.9 < len(audio) / 16000 < 1.1

    def test_trim_reduces_length(self, dummy_wav_file):
        audio_trimmed   = load_audio(dummy_wav_file, trim=True)
        audio_untrimmed = load_audio(dummy_wav_file, trim=False)
        # Sine has no silence so lengths should be roughly equal
        assert abs(len(audio_trimmed) - len(audio_untrimmed)) < 1000


# ─────────────────────────────────────────────────────────────────────────────
# audio_to_logmel
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioToLogmel:

    def test_output_shape(self, dummy_audio):
        audio, sr = dummy_audio
        spec = audio_to_logmel(audio, sr=sr, n_mels=80)
        assert spec.shape[0] == 80, "First dim must be n_mels=80"
        assert spec.shape[1] > 0,   "Time axis must be non-empty"

    def test_output_dtype(self, dummy_audio):
        audio, sr = dummy_audio
        spec = audio_to_logmel(audio, sr=sr)
        assert spec.dtype == np.float32

    def test_values_in_db_range(self, dummy_audio):
        audio, sr = dummy_audio
        spec = audio_to_logmel(audio, sr=sr)
        # power_to_db with ref=max → max value should be 0 dBFS
        assert spec.max() <= 0.1   # tiny tolerance for float precision

    def test_custom_n_mels(self, dummy_audio):
        audio, sr = dummy_audio
        spec = audio_to_logmel(audio, sr=sr, n_mels=40)
        assert spec.shape[0] == 40

    def test_fmin_fmax_respected(self, dummy_audio):
        """440 Hz sine should have energy when fmin < 440 < fmax."""
        audio, sr = dummy_audio
        spec_with    = audio_to_logmel(audio, sr=sr, fmin=200, fmax=8000)
        spec_without = audio_to_logmel(audio, sr=sr, fmin=1000, fmax=8000)
        # Spectrogram with 440 Hz in range should have higher max energy
        assert spec_with.max() > spec_without.max()


# ─────────────────────────────────────────────────────────────────────────────
# normalise
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalise:

    def test_zero_mean(self, dummy_spec):
        norm, _, _ = normalise(dummy_spec)
        assert abs(norm.mean()) < 1e-5

    def test_unit_std(self, dummy_spec):
        norm, _, _ = normalise(dummy_spec)
        assert abs(norm.std() - 1.0) < 1e-4

    def test_returns_stats(self, dummy_spec):
        _, mean, std = normalise(dummy_spec)
        assert isinstance(mean, float)
        assert isinstance(std,  float)
        assert std > 0

    def test_external_stats(self, dummy_spec):
        _, mean, std = normalise(dummy_spec)
        norm2, m2, s2 = normalise(dummy_spec, mean=mean, std=std)
        # Stats should be passed through unchanged
        assert m2 == mean
        assert s2 == std

    def test_output_dtype(self, dummy_spec):
        norm, _, _ = normalise(dummy_spec)
        assert norm.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# chunk
# ─────────────────────────────────────────────────────────────────────────────

class TestChunk:

    def test_output_shape(self, dummy_spec):
        """Each chunk should be (n_mels, chunk_frames)."""
        chunks = chunk(dummy_spec, frames=128, stride=64)
        assert chunks.ndim == 3
        assert chunks.shape[1] == dummy_spec.shape[0]  # n_mels preserved
        assert chunks.shape[2] == 128

    def test_stride_reduces_count(self, dummy_spec):
        chunks_50 = chunk(dummy_spec, frames=128, stride=64)   # 50% overlap
        chunks_no = chunk(dummy_spec, frames=128, stride=128)  # no overlap
        assert len(chunks_50) >= len(chunks_no)

    def test_no_partial_chunks(self, dummy_spec):
        """All chunks should be exactly chunk_frames wide."""
        chunks = chunk(dummy_spec, frames=128, stride=64)
        assert all(c.shape[1] == 128 for c in chunks)

    def test_short_spec_returns_empty(self):
        """Spec shorter than chunk_frames should return empty array."""
        short = np.zeros((80, 50))
        chunks = chunk(short, frames=128, stride=64)
        assert chunks.shape[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
# load_spectrogram  (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSpectrogram:

    def test_output_shape(self, dummy_wav_file):
        spec = load_spectrogram(dummy_wav_file)
        assert spec.shape[0] == 1   # batch=1
        assert spec.shape[1] == 1   # channel=1
        assert spec.shape[2] == 80  # n_mels

    def test_output_is_tensor(self, dummy_wav_file):
        spec = load_spectrogram(dummy_wav_file)
        assert isinstance(spec, torch.Tensor)

    def test_output_dtype(self, dummy_wav_file):
        spec = load_spectrogram(dummy_wav_file)
        assert spec.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_directory
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessDirectory:

    def test_creates_npy_files(self, dummy_wav_file, tmp_path):
        in_dir  = str(Path(dummy_wav_file).parent)
        out_dir = str(tmp_path / "processed")
        preprocess_directory(in_dir, out_dir, chunk_frames=128)
        npy_files = list(Path(out_dir).glob("*.npy"))
        assert len(npy_files) >= 1

    def test_npy_shape(self, dummy_wav_file, tmp_path):
        in_dir  = str(Path(dummy_wav_file).parent)
        out_dir = str(tmp_path / "processed")
        preprocess_directory(in_dir, out_dir, chunk_frames=128)
        npy = np.load(list(Path(out_dir).glob("*.npy"))[0])
        assert npy.ndim == 3         # (N_chunks, n_mels, chunk_frames)
        assert npy.shape[1] == 80
        assert npy.shape[2] == 128
