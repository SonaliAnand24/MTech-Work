"""Evaluation metrics: MCD, PESQ, STOI, speaker cosine similarity."""

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


def compute_mcd(ref: np.ndarray, syn: np.ndarray, sr: int = 22050, n_mfcc: int = 13) -> float:
    """Mel Cepstral Distortion (dB). Lower = better spectral match."""
    mr = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc+1)[1:]
    ms = librosa.feature.mfcc(y=syn, sr=sr, n_mfcc=n_mfcc+1)[1:]
    L  = min(mr.shape[1], ms.shape[1])
    d  = mr[:, :L] - ms[:, :L]
    return float((10 / np.log(10)) * np.sqrt(2 * np.mean(np.sum(d**2, axis=0))))


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = 16000, mode: str = "wb") -> float:
    """PESQ score ∈ [-0.5, 4.5]. Requires: pip install pesq"""
    try:
        from pesq import pesq
        L = min(len(ref), len(deg))
        return float(pesq(sr, ref[:L], deg[:L], mode))
    except ImportError:
        raise ImportError("Install PESQ: pip install pesq")


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int = 22050) -> float:
    """STOI score ∈ [0, 1]. Requires: pip install pystoi"""
    try:
        from pystoi import stoi
        L = min(len(ref), len(deg))
        return float(stoi(ref[:L], deg[:L], sr))
    except ImportError:
        raise ImportError("Install pystoi: pip install pystoi")


def compute_speaker_similarity(emb_ref: np.ndarray, emb_out: np.ndarray) -> float:
    """Cosine similarity between speaker d-vector embeddings."""
    r = emb_ref / (np.linalg.norm(emb_ref) + 1e-8)
    o = emb_out / (np.linalg.norm(emb_out) + 1e-8)
    return float(np.dot(r, o))


def evaluate_all(ref: np.ndarray, out: np.ndarray, sr: int = 22050) -> dict:
    results = {"mcd": compute_mcd(ref, out, sr)}
    for name, fn in [("pesq", lambda: compute_pesq(
            librosa.resample(ref, orig_sr=sr, target_sr=16000),
            librosa.resample(out, orig_sr=sr, target_sr=16000))),
                     ("stoi", lambda: compute_stoi(ref, out, sr))]:
        try:
            results[name] = fn()
        except Exception as e:
            logger.warning(f"{name} skipped: {e}")
            results[name] = None
    return results
