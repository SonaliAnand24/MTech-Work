"""PyTorch Dataset classes for all model training."""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random


class UnpairedAudioDataset(Dataset):
    """Unpaired (X, Y) chunks for CycleGAN / MelGAN-Cycle training."""

    def __init__(self, dir_X: str, dir_Y: str):
        self.cx = self._load(dir_X)
        self.cy = self._load(dir_Y)

    @staticmethod
    def _load(d):
        chunks = []
        for f in sorted(Path(d).glob("*.npy")):
            arr = np.load(str(f))
            chunks.extend(arr[i] for i in range(arr.shape[0]))
        return chunks

    def __len__(self): return max(len(self.cx), len(self.cy))

    def __getitem__(self, idx):
        x = torch.from_numpy(self.cx[idx % len(self.cx)]).unsqueeze(0)
        y = torch.from_numpy(self.cy[random.randint(0, len(self.cy)-1)]).unsqueeze(0)
        return {"X": x, "Y": y}


class SingleSpeakerDataset(Dataset):
    """Single-speaker chunks for VAE training, with speaker label."""

    def __init__(self, dir_path: str, speaker_id: int):
        self.chunks     = UnpairedAudioDataset._load(dir_path)
        self.speaker_id = speaker_id

    def __len__(self): return len(self.chunks)

    def __getitem__(self, idx):
        return {"spec": torch.from_numpy(self.chunks[idx]).unsqueeze(0),
                "speaker": self.speaker_id}
