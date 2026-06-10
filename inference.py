"""
inference.py — Unified Style Transfer Entry Point
==================================================
Run any of the five trained models on new audio.

Usage:
    python inference.py --model cyclegan \\
        --content  data/raw/self_recordings/GANINP4.mp3 \\
        --style_dir data/raw/kalam_references/ \\
        --checkpoint checkpoints/cyclegan/best.pt \\
        --output   results/audio_samples/my_output.wav

Models: cnn | melgan | cyclegan | melgan_cycle | vae
"""

import argparse, logging, yaml, torch
import numpy as np
import soundfile as sf
from pathlib import Path

from src.utils.preprocessing import load_spectrogram, load_audio, audio_to_logmel, normalise
from src.utils.audio_utils import spectrogram_to_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIGS = {
    "cnn":          "configs/cnn_config.yaml",
    "melgan":       "configs/melgan_config.yaml",
    "cyclegan":     "configs/cyclegan_config.yaml",
    "melgan_cycle": "configs/melgan_cycle_config.yaml",
    "vae":          "configs/vae_config.yaml",
}


def run_cnn(args, cfg):
    from src.models.cnn_style_transfer import CNNStyleTransfer
    content = load_spectrogram(args.content)
    style   = load_spectrogram(next(Path(args.style_dir).glob("*.mp3")))
    t = CNNStyleTransfer(device=args.device,
                         content_weight=cfg["training"]["content_weight"],
                         style_weight=cfg["training"]["style_weight"])
    return t.run(content, style, n_steps=cfg["training"]["n_steps"])


def run_melgan(args, cfg):
    from src.models.melgan import MelGAN
    m = MelGAN(cfg, device=args.device)
    m.load_checkpoint(args.checkpoint)
    return m.translate(load_spectrogram(args.content))


def run_cyclegan(args, cfg):
    from src.models.cyclegan import CycleGAN
    m = CycleGAN(cfg, device=args.device)
    m.load_checkpoint(args.checkpoint)
    return m.translate(load_spectrogram(args.content))


def run_melgan_cycle(args, cfg):
    from src.models.melgan_cycle import MelGANCycle
    m = MelGANCycle(cfg, device=args.device)
    m.load_checkpoint(args.checkpoint)
    return m.translate(load_spectrogram(args.content))


def run_vae(args, cfg):
    from src.models.vae_disentangled import DisentangledVAE
    m = DisentangledVAE(cfg["model"]).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    m.load_state_dict(ckpt["model"]); m.eval()
    content = load_spectrogram(args.content).to(args.device)
    refs = [load_spectrogram(str(f)).to(args.device)
            for f in list(Path(args.style_dir).glob("*.mp3"))[:10]]
    return m.transfer_style(content, refs)


DISPATCH = {
    "cnn":          run_cnn,
    "melgan":       run_melgan,
    "cyclegan":     run_cyclegan,
    "melgan_cycle": run_melgan_cycle,
    "vae":          run_vae,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, choices=list(DISPATCH))
    p.add_argument("--content",    required=True)
    p.add_argument("--style_dir",  required=True)
    p.add_argument("--output",     default="results/audio_samples/output.wav")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config",     default=None)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg_path = args.config or DEFAULT_CONFIGS[args.model]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    log.info(f"Running {args.model.upper()} on {args.device}")
    out_spec = DISPATCH[args.model](args, cfg)
    audio    = spectrogram_to_audio(out_spec.squeeze().cpu().numpy())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, samplerate=22050)
    log.info(f"✓ Saved → {args.output}")


if __name__ == "__main__":
    main()
