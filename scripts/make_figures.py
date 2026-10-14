"""Portfolio figures from AGGREGATE results already published in docs/results/ (no loan-level data is read).

Run: uv run --group docs python scripts/make_figures.py   -> docs/figures/*_{light,dark}.svg
Every number below is copied from the cited results document; change the document first, then this file.
Colours follow a CVD-validated categorical palette (slot 1 blue, slot 2 orange; light and dark steps).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("docs/figures")

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "neutral": "#a3a29b",
              "s1": "#2a78d6", "s2": "#eb6834", "band": "#e9f1fb"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "grid": "#33332f", "neutral": "#6f6e67",
             "s1": "#3987e5", "s2": "#d95926", "band": "#1f2b3a"},
}

# docs/results/empirical_baseline_v1.md, holdout_2023_2024.md, model_refresh_v5.md
LEVEL = [("2015–16 model on 2022–23 test\n(1.57M loans)", 1.79, "neutral"),
         ("2015–16 model on 2023–25 holdout\n(83,597 loans)", 0.62, "neutral"),
         ("Current champion on 2023 cohorts\n(42,061 loans)", 0.66, "neutral"),
         ("v5 refit (C2) on 2023 cohorts\n(same loans)", 0.98, "s1")]
# docs/results/model_refresh_v5.md (credit-score bands, 2023 selection set)
BANDS = ["≤ 660", "660–700", "700–740", "740–780", "> 780"]
OE_C0 = [1.05, 0.77, 0.63, 0.53, 0.30]
OE_C2 = [1.19, 0.97, 0.95, 0.97, 0.67]
# docs/results/lgd_el_v1.md (illustrative 12-month EL, $15.1B book)
EL = [("Base", 1.4), ("Adverse\nPD ×2, house prices −10%", 3.9), ("Severe\nPD ×4, house prices −20%", 10.5)]


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], labelsize=10, length=0)
    ax.grid(axis="x" if ax.get_label() == "h" else "y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(t, w=8.0, h=4.2):
    fig = plt.figure(figsize=(w, h), facecolor=t["surface"])
    plt.rcParams["font.family"] = "DejaVu Sans"
    return fig


def _title(fig, t, title, sub):
    fig.text(0.02, 0.95, title, color=t["text"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.88, sub, color=t["text2"], fontsize=10, va="top")


def level(t, mode):
    fig = _fig(t)
    ax = fig.add_axes([0.36, 0.14, 0.60, 0.62], label="h")
    _style(ax, t)
    ax.axvspan(0.8, 1.25, color=t["band"], zorder=0)
    ax.axvline(1.0, color=t["text2"], linewidth=1, linestyle=(0, (3, 3)))
    ys = range(len(LEVEL))[::-1]
    for y, (_lab, v, c) in zip(ys, LEVEL, strict=True):
        ax.barh(y, v, height=0.5, color=t[c])
        ax.text(v + 0.03, y, f"{v:.2f}", va="center", color=t["text"], fontsize=10, zorder=5,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": t["surface"], "edgecolor": "none"})
    ax.set_yticks(list(ys), [x[0] for x in LEVEL], color=t["text2"])
    ax.set_xlim(0, 2.0)
    ax.text(1.025, len(LEVEL) - 0.45, "acceptable band 0.8–1.25", color=t["text2"], fontsize=9, ha="center")
    ax.set_xlabel("Observed / expected default rate (1.0 = PD level right)", color=t["text2"], fontsize=10)
    _title(fig, t, "The old fit's PD level drifted; the 2022 refit brings it back",
           "12-month 90+ DPD, out-of-time. Ranking held throughout (AUC 0.74–0.82). Refit: selection stage.")
    fig.savefig(OUT / f"level_drift_{mode}.svg", facecolor=t["surface"])
    plt.close(fig)


def bands(t, mode):
    fig = _fig(t)
    ax = fig.add_axes([0.08, 0.14, 0.74, 0.62])
    _style(ax, t)
    ax.axhspan(0.8, 1.25, color=t["band"], zorder=0)
    ax.axhline(1.0, color=t["text2"], linewidth=1, linestyle=(0, (3, 3)))
    x = range(len(BANDS))
    ax.plot(x, OE_C0, color=t["neutral"], linewidth=2, marker="o", markersize=8,
            markeredgecolor=t["surface"], markeredgewidth=2)
    ax.plot(x, OE_C2, color=t["s1"], linewidth=2, marker="o", markersize=8,
            markeredgecolor=t["surface"], markeredgewidth=2)
    ax.text(4.12, OE_C0[-1], "Current champion\n(2015–16 fit)", color=t["text"], fontsize=9, va="center")
    ax.text(4.12, OE_C2[-1], "v5 refit (C2)", color=t["text"], fontsize=9, va="center")
    ax.set_xticks(list(x), BANDS)
    ax.set_xlim(-0.3, 4.1)
    ax.set_ylim(0, 1.4)
    ax.set_xlabel("Credit score band", color=t["text2"], fontsize=10)
    ax.set_ylabel("Observed / expected", color=t["text2"], fontsize=10)
    _title(fig, t, "The refit removes most of the credit-score distortion (finding F-007)",
           "2023 cohorts, 42,061 loans, 251 defaults; shaded band 0.8–1.25. The old fit over-predicts good scores.")
    fig.savefig(OUT / f"credit_score_bands_{mode}.svg", facecolor=t["surface"])
    plt.close(fig)


def expected_loss(t, mode):
    fig = _fig(t, h=3.6)
    ax = fig.add_axes([0.10, 0.16, 0.86, 0.56])
    _style(ax, t)
    for i, (_lab, v) in enumerate(EL):
        ax.bar(i, v, width=0.45, color=t["s1"])
        ax.text(i, v + 0.3, f"{v} bps", ha="center", color=t["text"], fontsize=10)
    ax.set_xticks(range(len(EL)), [e[0] for e in EL])
    ax.set_ylim(0, 12.5)
    ax.set_ylabel("12-month EL, bps of exposure", color=t["text2"], fontsize=10)
    _title(fig, t, "Expected loss under stress: 1.4 → 10.5 bps",
           "Illustrative EL = PD overlay × P(liquidation) × severity × EAD on a $15.1B book; not CECL/IFRS 9.")
    fig.savefig(OUT / f"expected_loss_{mode}.svg", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"  # keep text as text (selectable, smaller files)
    for mode, t in THEMES.items():
        level(t, mode)
        bands(t, mode)
        expected_loss(t, mode)
    print(sorted(p.name for p in OUT.glob("*.svg")))
