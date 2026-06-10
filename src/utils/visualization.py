"""
Visualization Utilities
========================
All plotting functions used across the thesis — spectrograms,
loss curves, MCD comparisons, F0 analysis, and latent space plots.

Every function saves a .png to results/figures/ AND optionally
returns the matplotlib figure for use inside Jupyter notebooks.

Usage:
    from src.utils.visualization import (
        plot_spectrogram_comparison,
        plot_loss_curves,
        plot_mcd_comparison,
        plot_f0_analysis,
        plot_mel_profile,
        plot_latent_interpolation,
    )
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts + notebooks)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Design tokens ────────────────────────────────────────────
BG      = "#0d1117"     # GitHub dark background
AX      = "#161b22"     # axes face
GRID    = "#30363d"     # grid lines / spines
TXT     = "#c9d1d9"     # primary text
MUTED   = "#8b949e"     # axis labels, ticks

MODEL_COLORS = {
    "content":      "#79c0ff",
    "style":        "#f78166",
    "cnn":          "#58a6ff",
    "melgan":       "#d2a8ff",
    "cyclegan":     "#3fb950",
    "melgan_cycle": "#ffa657",
    "vae":          "#f0883e",
}

MODEL_LABELS = {
    "cnn":          "CNN (Gram opt.)",
    "melgan":       "MelGAN (adv. only)",
    "cyclegan":     "CycleGAN (2D ResNet)",
    "melgan_cycle": "MelGAN-Cycle (hybrid)",
    "vae":          "VAE (disentangled)",
}

FIGURES_DIR = Path("results/figures")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str = "", fs: int = 9):
    """Apply dark-theme styling to a matplotlib axes."""
    ax.set_facecolor(AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    if title:
        ax.set_title(title, color=TXT, fontsize=fs, pad=5)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def _save(fig, filename: str, dpi: int = 140):
    """Save figure to results/figures/ directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=BG)
    logger.info(f"Saved → {path}")
    return str(path)


# ---------------------------------------------------------------------------
# 1. Single spectrogram
# ---------------------------------------------------------------------------

def plot_spectrogram(
    spec:     np.ndarray,
    title:    str  = "Log-Mel Spectrogram",
    sr:       int  = 22050,
    hop:      int  = 256,
    n_mels:   int  = 80,
    cmap:     str  = "magma",
    save_as:  Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single log-mel spectrogram with time and frequency axes.

    Args:
        spec:    (n_mels, T) normalised log-mel spectrogram.
        title:   Plot title.
        sr:      Sample rate (for time axis labelling).
        hop:     Hop length (for time axis labelling).
        n_mels:  Number of mel bins.
        cmap:    Matplotlib colormap.
        save_as: Filename to save (e.g. "my_spec.png"). None = no save.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)

    duration = spec.shape[1] * hop / sr
    img = ax.imshow(
        spec, aspect="auto", origin="lower", cmap=cmap,
        extent=[0, duration, 0, n_mels],
    )
    _style_ax(ax, title, fs=11)
    ax.set_xlabel(f"Time (s)", color=MUTED)
    ax.set_ylabel("Mel band", color=MUTED)
    cb = fig.colorbar(img, ax=ax, shrink=0.8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("Amplitude (normalised)", color=MUTED)

    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 2. Content / Style / Output comparison (3-panel)
# ---------------------------------------------------------------------------

def plot_spectrogram_comparison(
    content:  np.ndarray,
    style:    np.ndarray,
    output:   np.ndarray,
    model:    str  = "model",
    chunk_id: int  = 1,
    mcd:      Optional[float] = None,
    save_as:  Optional[str]   = None,
) -> plt.Figure:
    """
    Three-panel comparison: content | style | model output.

    Args:
        content:  Content speaker spectrogram (n_mels, T).
        style:    Style reference spectrogram  (n_mels, T).
        output:   Model output spectrogram     (n_mels, T).
        model:    Model name (for title and colour).
        chunk_id: Chunk index (for labelling).
        mcd:      MCD score to show in title (optional).
        save_as:  Filename to save.
    """
    col = MODEL_COLORS.get(model, TXT)
    mcd_str = f"  |  MCD = {mcd:.2f} dB" if mcd is not None else ""

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor=BG)
    fig.suptitle(
        f"{MODEL_LABELS.get(model, model)} — Chunk {chunk_id}{mcd_str}",
        color=col, fontsize=12, fontweight="bold",
    )

    vmin = min(content.min(), style.min(), output.min())
    vmax = max(content.max(), style.max(), output.max())

    specs  = [content,           style,               output]
    titles = ["Content (GANINP4)", "Style (APJ Kalam #3)", f"{MODEL_LABELS.get(model, model)} Output"]
    colors = [MODEL_COLORS["content"], MODEL_COLORS["style"], col]

    for ax, sp, title, tc in zip(axes, specs, titles, colors):
        ax.imshow(sp, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        _style_ax(ax, title, fs=9)
        ax.set_title(title, color=tc, fontsize=9, fontweight="bold", pad=5)
        for spine in ax.spines.values():
            spine.set_color(tc)
            spine.set_linewidth(1.5)
        ax.set_xlabel("Frame", color=MUTED, fontsize=7)
        ax.set_ylabel("Mel band", color=MUTED, fontsize=7)

    plt.tight_layout()
    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 3. Per-model spectrogram grid (3 chunks × 3 panels)
# ---------------------------------------------------------------------------

def plot_spectrogram_grid(
    model:   str,
    chunks:  List[Dict[str, np.ndarray]],  # list of {"content","style","output","mcd"}
    save_as: Optional[str] = None,
) -> plt.Figure:
    """
    3×3 grid: rows = chunks, columns = content / style / output.

    Args:
        model:   Model name key.
        chunks:  List of dicts, one per chunk, each with keys:
                 "content", "style", "output", optionally "mcd".
        save_as: Filename to save.
    """
    n = len(chunks)
    col = MODEL_COLORS.get(model, TXT)

    fig = plt.figure(figsize=(18, 4 * n + 1), facecolor=BG)
    fig.suptitle(
        f"{MODEL_LABELS.get(model, model)} — Spectrogram Grid\n"
        f"Content: GANINP4  ·  Style: APJ Abdul Kalam #3",
        color=col, fontsize=13, fontweight="bold", y=1.0,
    )
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.55, wspace=0.3)

    for row, chunk in enumerate(chunks):
        mcd_str = f"  (MCD={chunk['mcd']:.1f})" if "mcd" in chunk else ""
        vmin = min(chunk["content"].min(), chunk["style"].min(), chunk["output"].min())
        vmax = max(chunk["content"].max(), chunk["style"].max(), chunk["output"].max())

        specs  = [chunk["content"], chunk["style"], chunk["output"]]
        labels = [
            f"Chunk {row+1} — Content",
            f"Chunk {row+1} — Style",
            f"Chunk {row+1} — Output{mcd_str}",
        ]
        title_colors = [MODEL_COLORS["content"], MODEL_COLORS["style"], col]

        for c, (sp, lbl, tc) in enumerate(zip(specs, labels, title_colors)):
            ax = fig.add_subplot(gs[row, c])
            ax.imshow(sp, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            _style_ax(ax, lbl, fs=8)
            ax.set_title(lbl, color=tc, fontsize=8, pad=4)
            ax.set_xlabel("Frame", color=MUTED, fontsize=7)
            ax.set_ylabel("Mel band", color=MUTED, fontsize=7)

    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 4. All-5-model master comparison (chunk 2)
# ---------------------------------------------------------------------------

def plot_master_comparison(
    content:      np.ndarray,
    style:        np.ndarray,
    model_outputs: Dict[str, np.ndarray],   # {"cnn": spec, "melgan": spec, ...}
    save_as:      Optional[str] = None,
) -> plt.Figure:
    """
    Two-row layout:
        Row 1: content (centre) and style (centre-right) — with blank flanking cols
        Row 2: all 5 model outputs side by side

    Args:
        content:       Content spectrogram (n_mels, T).
        style:         Style reference spectrogram (n_mels, T).
        model_outputs: Dict mapping model name → output spectrogram.
        save_as:       Filename to save.
    """
    ordered = ["cnn", "melgan", "cyclegan", "melgan_cycle", "vae"]
    fig = plt.figure(figsize=(24, 10), facecolor=BG)
    fig.suptitle(
        "Full Spectrogram Comparison — All 5 Models  |  Chunk 2 (~4:47s)\n"
        "Content: GANINP4  ·  Target Style: APJ Abdul Kalam #3",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.28)

    all_specs = [content, style] + [model_outputs.get(m, np.zeros_like(content))
                                     for m in ordered]
    vmin = min(s.min() for s in all_specs)
    vmax = max(s.max() for s in all_specs)

    # Row 1: content and style (in columns 1 and 3)
    for col_pos, (sp, lbl, tc) in [
        (1, (content, "Content\n(GANINP4)",      MODEL_COLORS["content"])),
        (3, (style,   "Style Target\n(APJ Kalam #3)", MODEL_COLORS["style"])),
    ]:
        ax = fig.add_subplot(gs[0, col_pos])
        ax.imshow(sp, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(lbl, color=tc, fontsize=10, fontweight="bold", pad=5)
        ax.set_facecolor(AX)
        ax.tick_params(colors=MUTED, labelsize=6)
        for spi in ax.spines.values():
            spi.set_color(tc)
            spi.set_linewidth(2)
    for c in [0, 2, 4]:
        fig.add_subplot(gs[0, c]).set_visible(False)

    # Row 2: 5 model outputs
    for col, model in enumerate(ordered):
        sp  = model_outputs.get(model, np.zeros_like(content))
        tc  = MODEL_COLORS.get(model, TXT)
        lbl = f"{MODEL_LABELS.get(model, model)}\nOutput"
        ax  = fig.add_subplot(gs[1, col])
        ax.imshow(sp, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(lbl, color=tc, fontsize=8, fontweight="bold", pad=5)
        ax.set_facecolor(AX)
        ax.tick_params(colors=MUTED, labelsize=6)
        for spi in ax.spines.values():
            spi.set_color(tc)
            spi.set_linewidth(2)

    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 5. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    curves:    Dict[str, Dict[str, List[float]]],
    save_as:   Optional[str] = None,
) -> plt.Figure:
    """
    Plot training / optimisation loss curves for all models.

    Args:
        curves: Nested dict — {model_name: {loss_name: [values]}}
                e.g. {"cnn": {"total": [...]},
                       "cyclegan": {"G": [...], "D": [...], "cyc": [...]}}
        save_as: Filename to save.
    """
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor=BG)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Training / Optimisation Loss Curves — All Models",
        color="white", fontsize=13, fontweight="bold",
    )

    line_styles = ["-", "--", ":", "-."]
    line_colors = [MODEL_COLORS["cnn"], MODEL_COLORS["style"],
                   MODEL_COLORS["content"], MODEL_COLORS["vae"]]

    for ax, (model, loss_dict) in zip(axes, curves.items()):
        mc = MODEL_COLORS.get(model, TXT)
        for i, (loss_name, values) in enumerate(loss_dict.items()):
            x = np.arange(1, len(values) + 1)
            ax.plot(x, values,
                    color=line_colors[i % len(line_colors)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=1.6, label=loss_name)
        _style_ax(ax, MODEL_LABELS.get(model, model), fs=10)
        ax.set_xlabel("Step / Epoch", color=MUTED)
        ax.set_ylabel("Loss", color=MUTED)
        ax.legend(facecolor=AX, labelcolor=TXT, edgecolor=GRID, fontsize=8)
        ax.set_facecolor(AX)

    plt.tight_layout()
    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 6. GAN discriminator convergence
# ---------------------------------------------------------------------------

def plot_gan_convergence(
    histories: Dict[str, Dict[str, List[float]]],
    save_as:   Optional[str] = None,
) -> plt.Figure:
    """
    Plot G and D loss per epoch for GAN variants.
    Annotates each panel with final D loss and deviation from ideal 0.5.

    Args:
        histories: {model: {"G": [...], "D": [...], "cyc": [...]}}
        save_as:   Filename to save.
    """
    n   = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor=BG)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "GAN Discriminator Convergence  |  Ideal LSGAN D loss ≈ 0.5",
        color="white", fontsize=12, fontweight="bold",
    )

    for ax, (model, hist) in zip(axes, histories.items()):
        mc  = MODEL_COLORS.get(model, TXT)
        ep  = np.arange(1, len(hist["G"]) + 1)
        ax.plot(ep, hist["G"], color=mc,                    linewidth=2.0, label="Generator")
        ax.plot(ep, hist["D"], color=MODEL_COLORS["content"], linewidth=1.5,
                label="Discriminator", linestyle="--")
        if "cyc" in hist:
            ax.plot(ep, hist["cyc"], color=MODEL_COLORS["style"], linewidth=1.2,
                    label="Cycle loss", linestyle=":")
        ax.axhline(0.5, color="white", linestyle=":", linewidth=0.8, alpha=0.5, label="Ideal D=0.5")

        final_d = hist["D"][-1]
        gap     = abs(final_d - 0.5)
        ax.text(0.97, 0.95, f"Final D = {final_d:.3f}\nΔ ideal = {gap:.3f}",
                transform=ax.transAxes, ha="right", va="top", color=TXT, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=AX, edgecolor=mc))

        _style_ax(ax, MODEL_LABELS.get(model, model), fs=10)
        ax.set_xlabel("Epoch", color=MUTED)
        ax.set_ylabel("Loss", color=MUTED)
        ax.legend(facecolor=AX, labelcolor=TXT, edgecolor=GRID, fontsize=8)
        ax.set_facecolor(AX)

    plt.tight_layout()
    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 7. MCD comparison bar chart
# ---------------------------------------------------------------------------

def plot_mcd_comparison(
    mcd_scores: Dict[str, np.ndarray],    # {"cnn": [c1,c2,c3], "cyclegan": [...], ...}
    chunk_labels: List[str] = None,
    save_as:    Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart + mean MCD horizontal bars for all models.

    Args:
        mcd_scores:   Dict mapping model name → array of per-chunk MCD scores.
        chunk_labels: Labels for each chunk (x-axis).
        save_as:      Filename to save.
    """
    models  = list(mcd_scores.keys())
    n_chunks = len(next(iter(mcd_scores.values())))
    chunk_labels = chunk_labels or [f"Chunk {i+1}" for i in range(n_chunks)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), facecolor=BG,
                                    gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(
        "Mel Cepstral Distortion — All 5 Models  |  GANINP4 → APJ Abdul Kalam #3\n"
        "(Lower MCD = output spectrogram structure closer to content)",
        color="white", fontsize=12, fontweight="bold",
    )

    # Left: grouped bars
    ax1.set_facecolor(AX)
    x       = np.arange(n_chunks)
    offsets = np.linspace(-0.3, 0.3, len(models))
    w       = 0.55 / len(models)

    for offset, model in zip(offsets, models):
        scores = mcd_scores[model]
        bars   = ax1.bar(x + offset, scores, width=w * 0.9,
                         color=MODEL_COLORS.get(model, TXT), alpha=0.88,
                         label=MODEL_LABELS.get(model, model))
        for bar in bars:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                color=TXT, fontsize=6.5,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(chunk_labels, color=TXT)
    ax1.set_ylabel("MCD (dB)  ↓  lower is better", color=MUTED)
    _style_ax(ax1, "Per-Chunk MCD — All Models", fs=11)
    ax1.legend(facecolor=AX, labelcolor=TXT, edgecolor=GRID, fontsize=9)

    # Right: mean MCD horizontal bars
    ax2.set_facecolor(AX)
    means  = [np.mean(mcd_scores[m]) for m in models]
    colors = [MODEL_COLORS.get(m, TXT) for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    bars = ax2.barh(labels, means, color=colors, alpha=0.88, height=0.5)
    for bar, val in zip(bars, means):
        ax2.text(val + 3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f} dB", va="center", color=TXT, fontsize=9, fontweight="bold")

    best_idx = int(np.argmin(means))
    ax2.annotate(
        "← Best", xy=(means[best_idx], best_idx),
        xytext=(means[best_idx] + 60, best_idx),
        color="#3fb950", fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.2),
    )
    _style_ax(ax2, "Mean MCD (all chunks)", fs=11)
    ax2.set_xlabel("Mean MCD (dB)  ↓", color=MUTED)
    ax2.tick_params(axis="y", colors=TXT, labelsize=9)

    plt.tight_layout()
    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 8. F0 analysis
# ---------------------------------------------------------------------------

def plot_f0_analysis(
    f0_content: np.ndarray,
    f0_style:   np.ndarray,
    sr:         int   = 22050,
    hop:        int   = 256,
    save_as:    Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel F0 analysis: contour overlay + histogram comparison.

    Args:
        f0_content: F0 array from content speaker (Hz, NaN for unvoiced).
        f0_style:   F0 array from style speaker.
        sr:         Sample rate (for time axis).
        hop:        Hop length (for time axis).
        save_as:    Filename to save.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)
    fig.suptitle(
        "F0 (Fundamental Frequency) Analysis\n"
        "Content: GANINP4  ·  Style: APJ Abdul Kalam #3",
        color="white", fontsize=12, fontweight="bold",
    )

    t_c = np.arange(len(f0_content)) * hop / sr
    t_s = np.arange(len(f0_style))   * hop / sr

    # Contour
    ax1.plot(t_c, f0_content, color=MODEL_COLORS["content"],
             linewidth=0.6, alpha=0.8, label="Content (GANINP4)")
    ax1.plot(t_s, f0_style,   color=MODEL_COLORS["style"],
             linewidth=0.6, alpha=0.8, label="APJ Kalam #3")

    c_clean = f0_content[~np.isnan(f0_content)]
    s_clean = f0_style[~np.isnan(f0_style)]
    ax1.axhline(c_clean.mean(), color=MODEL_COLORS["content"], linestyle=":",
                linewidth=1.2, alpha=0.7)
    ax1.axhline(s_clean.mean(), color=MODEL_COLORS["style"],   linestyle=":",
                linewidth=1.2, alpha=0.7)

    _style_ax(ax1, f"F0 Contour  |  Content μ={c_clean.mean():.0f}Hz  ·  "
              f"Style μ={s_clean.mean():.0f}Hz  (Δ={c_clean.mean()-s_clean.mean():.0f}Hz)", fs=9)
    ax1.set_xlabel("Time (s)", color=MUTED)
    ax1.set_ylabel("F0 (Hz)", color=MUTED)
    ax1.legend(facecolor=AX, labelcolor=TXT, edgecolor=GRID, fontsize=9)

    # Histogram
    ax2.hist(c_clean, bins=60, color=MODEL_COLORS["content"], alpha=0.7, density=True,
             label=f"GANINP4  μ={c_clean.mean():.0f} Hz")
    ax2.hist(s_clean, bins=60, color=MODEL_COLORS["style"],   alpha=0.7, density=True,
             label=f"APJ Kalam μ={s_clean.mean():.0f} Hz")
    _style_ax(ax2, "F0 Distribution — Content vs Style", fs=9)
    ax2.set_xlabel("F0 (Hz)", color=MUTED)
    ax2.set_ylabel("Density", color=MUTED)
    ax2.legend(facecolor=AX, labelcolor=TXT, edgecolor=GRID, fontsize=9)

    plt.tight_layout()
    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 9. Mel band mean profile
# ---------------------------------------------------------------------------

def plot_mel_profile(
    specs:    Dict[str, np.ndarray],   # {"content": ..., "style": ..., "cnn": ...}
    save_as:  Optional[str] = None,
) -> plt.Figure:
    """
    Per-mel-band mean energy profile comparison.
    Useful for seeing how well each model shifts energy toward the style register.

    Args:
        specs:   Dict of label → spectrogram (n_mels, T).
        save_as: Filename to save.
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    fig.suptitle("Mean Energy per Mel Band — Style Register Shift",
                 color="white", fontsize=12)

    for label, spec in specs.items():
        col = MODEL_COLORS.get(label, TXT)
        lbl = MODEL_LABELS.get(label, label)
        ls  = "-" if label in ("content", "style") else "--"
        lw  = 2.0 if label in ("content", "style") else 1.4
        ax.plot(spec.mean(axis=1), np.arange(spec.shape[0]),
                color=col, linewidth=lw, linestyle=ls, label=lbl)

    _style_ax(ax, "", fs=10)
    ax.set_xlabel("Mean amplitude (normalised)", color=MUTED)
    ax.set_ylabel("Mel band", color=MUTED)
    ax.legend(facecolor=AX, labelcolor=TXT, edgecolor=GRID, fontsize=9)
    ax.set_title("Mean Energy per Mel Band\n"
                 "Lower bands (~5–25) carry F0 register — style shift visible here",
                 color=TXT, fontsize=10)

    plt.tight_layout()
    if save_as:
        _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 10. VAE latent interpolation
# ---------------------------------------------------------------------------

def plot_latent_interpolation(
    interpolated_specs: List[np.ndarray],   # ordered α=0 → α=1
    alphas:             List[float],
    save_as:            Optional[str] = None,
) -> plt.Figure:
    """
    Visualise smooth interpolation between content and style in VAE latent space.

    Args:
        interpolated_specs: List of spectrograms at each α.
        alphas:             Interpolation weights (0.0 = content, 1.0 = style).
        save_as:            Filename to save.
    """
    n   = len(interpolated_specs)
    fig = plt.figure(figsize=(4 * n, 4), facecolor=BG)
    fig.suptitle(
        "VAE Latent Space Interpolation: Content Style → APJ Kalam Style\n"
        "α = 0.0 (pure content) → α = 1.0 (pure Kalam style)",
        color="white", fontsize=11, fontweight="bold",
    )

    vmin = min(s.min() for s in interpolated_specs)
    vmax = max(s.max() for s in interpolated_specs)

    for i, (spec, alpha) in enumerate(zip(interpolated_specs, alphas)):
        ax = fig.add_subplot(1, n, i + 1)
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        blend = MODEL_COLORS["content"] if alpha < 0.5 else MODEL_COLORS["style"]
        ax.set_title(f"α = {alpha:.2f}", color=blend, fontsize=9, fontweight="bold")
        ax.set_facecolor(AX)
        ax.tick_params(colors=MUTED, labelsize=6)
        for sp in ax.spines.values():
            sp.set_color(blend)

    plt.tight_layout()
    if save_as:
        _save(fig, save_as)
    return fig
