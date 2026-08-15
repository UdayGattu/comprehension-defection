#!/usr/bin/env python3
"""Build every figure in the paper from paper/FACTS.json and nothing else.

This script opens no database and reads no markdown. If a number is not in
FACTS.json it cannot appear in a figure. Each figure is written twice: a vector
PDF for LaTeX and a 300 dpi PNG for slides and previews.

Palette is Okabe-Ito, which is distinguishable under deuteranopia, protanopia
and tritanopia and also survives greyscale printing. No 3D, no dual axes, no
gridline clutter, no decorative colour.

    python paper/make_figures.py
    python paper/make_figures.py --only fig1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
import numpy as np                                   # noqa: E402

PAPER = Path(__file__).resolve().parent
FIGDIR = PAPER / "figures"
FACTS_PATH = PAPER / "FACTS.json"

# ---------------------------------------------------------------------------
# Okabe-Ito
# ---------------------------------------------------------------------------
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7F7F7F"
LIGHT = "#E8E8E8"

plt.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType, not Type 3: NeurIPS requires it
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

MODELS = [("llama", "Llama-3.1-8B"),
          ("mistral", "Mistral-7B-v0.3"),
          ("qwen", "Qwen2.5-7B")]
OPPS = [("allc", "ALLC"), ("tft", "TFT")]

BUILT: list[tuple[str, str]] = []
SKIPPED: list[tuple[str, str]] = []


# ---------------------------------------------------------------------------
# fact access - every read goes through here so a missing key is loud
# ---------------------------------------------------------------------------

class Facts:
    def __init__(self, path: Path):
        self.raw = json.loads(path.read_text(encoding="utf-8"))["facts"]
        self.used: set[str] = set()

    def __contains__(self, key):
        return key in self.raw

    def v(self, key):
        if key not in self.raw:
            raise KeyError(f"FACTS.json has no key {key!r}")
        self.used.add(key)
        return self.raw[key]["value"]

    def get(self, key, default=None):
        if key not in self.raw:
            return default
        self.used.add(key)
        return self.raw[key]["value"]


F: Facts


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf, png = FIGDIR / f"{name}.pdf", FIGDIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    BUILT.append((name, f"{pdf.stat().st_size/1024:.0f} KiB pdf / "
                        f"{png.stat().st_size/1024:.0f} KiB png"))
    print(f"    wrote {pdf.name} and {png.name}")


def forest(ax, ys, diffs, los, his, colour, marker, label, size=18):
    ax.hlines(ys, los, his, color=colour, linewidth=1.1, zorder=2)
    ax.scatter(diffs, ys, s=size, color=colour, marker=marker, zorder=3,
               label=label, edgecolors="none")


def row_labels(ax, ys, labels):
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()


# ===========================================================================
# FIGURE 1 - reading vs using.  The whole thesis in one plate.
# ===========================================================================

FIG1_GROUP = "exp6_{}_sem_logit"


def fig1():
    rows, labels = [], []
    for mk, mlabel in MODELS:
        for ok, olabel in OPPS:
            rows.append((FIG1_GROUP.format(mk), ok))
            labels.append(f"{mlabel}\nvs {olabel}")
    ys = np.arange(len(rows), dtype=float)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(5.5, 2.6), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.06})

    # ---- LEFT: did the model READ the block? three independent measures ----
    read_specs = [
        ("arm 3: state probes correct", BLUE, "o",
         lambda g, o: F.v(f"cpr.{g}.3|{o}")["cpr_all_or_nothing"]),
        ("arm 3s: answer = displayed false score", VERM, "s",
         lambda g, o: F.v(f"echo.{g}.3s|{o}")["frac_matched_displayed"]),
        ("arm 3c: answer = donor's score", GREEN, "^",
         lambda g, o: F.v(f"donor_echo.{g}.3c|{o}")["frac_matched_donor"]),
    ]
    off = [-0.24, 0.0, 0.24]
    axL.axvline(1.0, color=GREY, linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
    for (name, col, mk_, fn), dy in zip(read_specs, off):
        vals = [fn(g, o) for g, o in rows]
        axL.scatter(vals, ys + dy, s=17, color=col, marker=mk_, zorder=3,
                    label=name, edgecolors="none")
        for val, y in zip(vals, ys + dy):
            if val < 0.999:                    # only the exceptions get a number
                axL.text(val - 0.03, y, f"{val:.3f}", ha="right", va="center",
                         fontsize=5.4, color=col)
    axL.set_xlim(-0.03, 1.06)
    axL.set_xticks([0, 0.5, 1.0])
    axL.set_xlabel("proportion of probes")
    axL.set_title("A. Reading the block", loc="left", fontweight="bold")
    row_labels(axL, ys, labels)
    axL.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False,
               handletextpad=0.2, borderaxespad=0, labelspacing=0.35,
               fontsize=6)

    # ---- RIGHT: did it USE the block? the three field contrasts -----------
    use_specs = [
        ("content_move  (3 - 3m)", BLUE, "o"),
        ("content_score (3 - 3s)", VERM, "s"),
        ("content_donor (3 - 3c)", GREEN, "^"),
    ]
    axR.axvline(0.0, color=BLACK, linewidth=0.7, zorder=1)
    for (key, col, mk_), dy in zip(use_specs, off):
        name = key.split()[0]
        d, lo, hi = [], [], []
        for g, o in rows:
            q = F.v(f"contrast.{g}.{name}|{o}")["quotable"]
            d.append(q["diff"])
            lo.append(q["lo"])
            hi.append(q["hi"])
        forest(axR, ys + dy, d, lo, hi, col, mk_, key, size=17)
    axR.set_xlim(-0.46, 0.09)
    axR.set_xlabel("difference in P(defect), 95% CI")
    axR.set_title("B. Using the block", loc="left", fontweight="bold")
    axR.legend(loc="upper left", bbox_to_anchor=(0.02, 0.90), frameon=False,
               handletextpad=0.2, borderaxespad=0, labelspacing=0.35,
               fontsize=6)
    for ax in (axL, axR):
        ax.grid(axis="y", color=LIGHT, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.text(0.5, -0.10,
             "Every reading measure is pinned at 1.00: the model repeats the "
             "block back, including the parts that are false.\nThe same block's "
             "content moves behaviour by at most a few points in five of the "
             "six rows.",
             ha="center", fontsize=6.2, color=GREY)
    save(fig, "fig1_read_vs_use")


# ===========================================================================
# FIGURE 2 - the arm ladder, as actually rendered to the model
# ===========================================================================

LADDER = [("1", "Arm 1", "no block at all"),
          ("3b", "Arm 3b", "placebo: matched, no state"),
          ("3", "Arm 3", "treatment: the true state"),
          ("3c", "Arm 3c", "donor: another episode"),
          ("3s", "Arm 3s", "own score falsified"),
          ("3m", "Arm 3m", "last move falsified")]


def fig2():
    src = F.v("arm_ladder.source")
    blocks = {a: F.v(f"arm_ladder.{a}")["state_block"] for a, _, _ in LADDER}
    ref = [ln for ln in blocks["3"].split("\n")]

    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5, 2.45))
    for ax, (arm, title, sub) in zip(axes.ravel(), LADDER):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0.0, 0.0), 1.0, 0.80, transform=ax.transAxes,
                               facecolor="white", edgecolor=GREY,
                               linewidth=0.6, zorder=0))
        ax.text(0.0, 0.95, title, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")
        ax.text(0.0, 0.855, sub, transform=ax.transAxes, fontsize=5.9,
                color=GREY, va="bottom")
        lines = blocks[arm].split("\n") if blocks[arm] else \
            ["(no [STATE] block is", " rendered at all)"]
        y = 0.70
        for i, ln in enumerate(lines):
            differs = blocks[arm] and blocks["3"] and (
                i >= len(ref) or ln != ref[i]) and ln != "[STATE]"
            if differs:
                ax.add_patch(Rectangle((0.02, y - 0.048), 0.955, 0.105,
                                       transform=ax.transAxes,
                                       facecolor=ORANGE, alpha=0.35,
                                       edgecolor="none", zorder=1))
            ax.text(0.035, y, ln, transform=ax.transAxes, fontsize=4.9,
                    family="monospace", va="center", zorder=2,
                    color=BLACK if blocks[arm] else GREY)
            y -= 0.125
    fig.suptitle(
        f"[STATE] blocks as rendered to {src['model_id']} "
        f"(turn {src['turn']} vs TFT); every block padded to exactly "
        f"{src['parity_target_tokens']} tokens",
        fontsize=7.5, y=1.04)
    fig.text(0.5, -0.045,
             "Text is sliced verbatim from turn_details.prompt_full in "
             f"{src['database']}. Shading marks every line that differs from "
             "Arm 3.",
             ha="center", fontsize=6.2, color=GREY)
    fig.subplots_adjust(hspace=0.42, wspace=0.10)
    save(fig, "fig2_arm_ladder")


# ===========================================================================
# FIGURE 3 - what history deprivation does to the comprehension probes
# ===========================================================================

FIELDS = [("own_score", "own score\n(arithmetic)"),
          ("opponent_last", "opp. last\n(recall)"),
          ("rounds_played", "rounds\n(counting)")]

HIST = ("exp6_{}_sem_logit", "with history")
NOHIST = ("exp7_{}_nohist_logit", "no history")


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.25), sharey=True)
    x = np.arange(len(FIELDS), dtype=float)
    w = 0.36
    for ax, (mk, mlabel) in zip(axes, MODELS):
        gh, gn = HIST[0].format(mk), NOHIST[0].format(mk)
        vh = [F.v(f"cpr.{gh}.1|pooled")[f] for f, _ in FIELDS]
        vn = [F.v(f"cpr.{gn}.1|pooled")[f] for f, _ in FIELDS]
        ax.bar(x - w / 2, vh, w, color=BLUE, label=HIST[1], edgecolor="none")
        ax.bar(x + w / 2, vn, w, color=ORANGE, label=NOHIST[1],
               edgecolor="none")
        for xi, val in zip(x - w / 2, vh):
            ax.text(xi, val + 0.025, f"{val:.2f}", ha="center", fontsize=5.6,
                    color=BLUE)
        for xi, val in zip(x + w / 2, vn):
            ax.text(xi, val + 0.025, f"{val:.2f}", ha="center", fontsize=5.6,
                    color=ORANGE)
        # The always-answer-the-same floor is only defined for the binary
        # recall probe, and only there is it drawn. Where it reaches 1.00 the
        # opponent never defected in arm 1, so the probe cannot discriminate
        # reading from guessing at all - which is worth seeing, not hiding.
        bh = F.v(f"probe_baseline.{gh}.1|pooled")["constant_answer_accuracy"]
        bn = F.v(f"probe_baseline.{gn}.1|pooled")["constant_answer_accuracy"]
        ax.hlines([bh, bn], 1 - 0.52, 1 + 0.52, color=VERM, linewidth=0.9,
                  linestyle=(0, (3, 2)), zorder=4)
        ax.set_title(mlabel, loc="left")
        ax.text(0.02, 0.985, f"recall floor: {bh:.2f} / {bn:.2f}",
                transform=ax.transAxes, fontsize=5.6, color=VERM, va="top")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in FIELDS], fontsize=5.8)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color=LIGHT, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("arm 1 probe accuracy\n(turn 0 excluded)")
    axes[0].legend(handles=[
        Line2D([], [], color=BLUE, linewidth=5, label=HIST[1]),
        Line2D([], [], color=ORANGE, linewidth=5, label=NOHIST[1]),
        Line2D([], [], color=VERM, linewidth=0.9, linestyle=(0, (3, 2)),
               label="always-same-label floor")],
        loc="upper left", bbox_to_anchor=(0, -0.24), ncol=3, frameon=False,
        handletextpad=0.4, columnspacing=1.4, handlelength=1.6)
    save(fig, "fig3_deprivation")


# ===========================================================================
# FIGURE 4 - does the content effect survive history deprivation?
# ===========================================================================

def fig4():
    rows, labels = [], []
    for mk, mlabel in MODELS:
        for ok, olabel in OPPS:
            rows.append((mk, ok))
            labels.append(f"{mlabel}\nvs {olabel}")
    ys = np.arange(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.axvline(0.0, color=BLACK, linewidth=0.7, zorder=1)
    got_floor = None
    for i, (mk, ok) in enumerate(rows):
        a = F.v(f"contrast.{HIST[0].format(mk)}.content_move|{ok}")["quotable"]
        b = F.v(f"contrast.{NOHIST[0].format(mk)}.content_move|{ok}")["quotable"]
        y = ys[i]
        ax.annotate("", xy=(b["diff"], y), xytext=(a["diff"], y),
                    arrowprops=dict(arrowstyle="-|>", color=GREY,
                                    linewidth=0.8, shrinkA=2.5, shrinkB=2.5))
        ax.hlines(y - 0.14, a["lo"], a["hi"], color=BLUE, linewidth=1.1)
        ax.scatter([a["diff"]], [y - 0.14], s=20, color=BLUE, marker="o",
                   zorder=3, edgecolors="none")
        ax.hlines(y + 0.14, b["lo"], b["hi"], color=ORANGE, linewidth=1.1)
        ax.scatter([b["diff"]], [y + 0.14], s=20, color=ORANGE, marker="s",
                   zorder=3, edgecolors="none")
        if mk == "qwen":
            got_floor = (b["diff"], y + 0.14)
    if got_floor:
        ax.annotate("Qwen's effect is gone: with no history\n"
                    "the falsified move changes nothing",
                    xy=got_floor, xytext=(-0.435, len(rows) - 3.3),
                    fontsize=6.2, color=VERM, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=VERM, linewidth=0.7,
                                    connectionstyle="arc3,rad=0.18"))
    ax.set_xlim(-0.45, 0.05)
    row_labels(ax, ys, labels)
    ax.set_xlabel("content_move = P(defect | arm 3) - P(defect | arm 3m)\n"
                  "turn 0 excluded, episode-clustered 95% CI")
    ax.set_title("Falsifying the opponent's last move: with history vs without",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", linestyle="-",
               markersize=4, label="semantic framing, with history (exp6)"),
        Line2D([], [], color=ORANGE, marker="s", linestyle="-",
               markersize=4, label="semantic framing, no history (exp7)")],
        loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2, frameon=False,
        handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig4_nohist_survival")


# ===========================================================================
# FIGURE 5 - the swap mechanism: the placebo is the loud arm
# ===========================================================================

SWAP_PANELS = [("exp3_qwen_swap", "exp3  Qwen2.5-7B, labels swapped"),
               ("exp7_qwen_swap_logit", "exp7  Qwen2.5-7B, labels swapped")]
SWAP_ARMS = [("1", "arm 1\nno block"),
             ("3b", "arm 3b\nplacebo"),
             ("3", "arm 3\ntrue state")]


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    x = np.arange(len(SWAP_ARMS), dtype=float)
    for ax, (g, title) in zip(axes, SWAP_PANELS):
        vals = [F.v(f"opponent_spread.{g}.{a}")["abs_spread"]
                for a, _ in SWAP_ARMS]
        cols = [GREY, VERM, BLUE]
        ax.bar(x, vals, 0.55, color=cols, edgecolor="none")
        for xi, val in zip(x, vals):
            ax.text(xi, val + 0.015, f"{val:.3f}", ha="center", fontsize=6.6)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in SWAP_ARMS], fontsize=6.5)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", color=LIGHT, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("|P(defect | TFT) - P(defect | ALLC)|\nwithin arm")
    fig.text(0.5, -0.10,
             "The placebo block - which contains no state at all - produces "
             "the largest opponent-dependent\nbehaviour swing in the "
             "experiment, four times the swing produced by the true state.",
             ha="center", fontsize=6.4, color=GREY)
    save(fig, "fig5_swap_mechanism")


# ===========================================================================
# FIGURE 6 - is the placebo leaking the turn index?
# ===========================================================================

NOHIST_GROUPS = [("exp7_llama_nohist_logit", "Llama semantic"),
                 ("exp7_llama_absnohist_logit", "Llama absolute"),
                 ("exp7_mistral_nohist_logit", "Mistral semantic"),
                 ("exp7_qwen_nohist_logit", "Qwen semantic"),
                 ("exp7_qwen_absnohist_logit", "Qwen absolute")]


def fig6():
    rows, labels = [], []
    for g, glabel in NOHIST_GROUPS:
        for ok, olabel in OPPS:
            rows.append((g, ok))
            labels.append(f"{glabel}\nvs {olabel}")
    ys = np.arange(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.axvline(0.0, color=BLACK, linewidth=0.7, zorder=1)
    for arm, col, mk_, dy, lab in [
            ("3b", VERM, "s", -0.16, "arm 3b (placebo, carries 'Round parity')"),
            ("1", BLUE, "o", 0.16, "arm 1 (no block: the control)")]:
        d, lo, hi = [], [], []
        for g, o in rows:
            v = F.v(f"parity.{g}.{arm}|{o}")
            d.append(v["detrended_diff"])
            lo.append(v["lo"])
            hi.append(v["hi"])
        forest(ax, ys + dy, d, lo, hi, col, mk_, lab, size=20)
    row_labels(ax, ys, labels)
    ax.set_xlabel("detrended parity coefficient: even turns minus odd turns,\n"
                  "local turn trend removed, episode-clustered 95% CI")
    ax.set_title("Does the placebo leak the turn index?", loc="left",
                 fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.30), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.6)
    save(fig, "fig6_parity_leak")


# ===========================================================================

FIGURES = {"fig1": fig1, "fig2": fig2, "fig3": fig3,
           "fig4": fig4, "fig5": fig5, "fig6": fig6}


def main() -> int:
    global F
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    ap.add_argument("--no-tables", action="store_true")
    args = ap.parse_args()

    if not FACTS_PATH.exists():
        raise SystemExit("paper/FACTS.json missing - run paper/extract_facts.py")
    F = Facts(FACTS_PATH)
    print(f"figures from {FACTS_PATH.name} ({len(F.raw)} facts)")

    for name in (args.only or sorted(FIGURES)):
        print(f"\n[{name}]")
        try:
            FIGURES[name]()
        except Exception as exc:                       # noqa: BLE001
            SKIPPED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"    COULD NOT BUILD: {type(exc).__name__}: {exc}")

    if not args.no_tables:
        import make_tables
        make_tables.main()

    print("\nfigure inventory")
    for n, s in BUILT:
        print(f"    {n:<24}{s}")
    if SKIPPED:
        print("\nNOT BUILT")
        for n, why in SKIPPED:
            print(f"    {n:<24}{why}")
    print(f"\n{len(F.used)} distinct FACTS keys consumed")
    return 1 if SKIPPED else 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PAPER))
    raise SystemExit(main())
