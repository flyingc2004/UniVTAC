from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROWS = [
    ("heavy_rough_soft", "light_rough_soft", "Weight", 2, 0.50, 0.25),
    ("heavy_smooth_rigid", "light_rough_soft", "All 3", 1, 0.00, 0.44),
    ("heavy_smooth_rigid", "light_smooth_rigid", "Weight", 4, 0.50, 0.25),
    ("light_rough_rigid", "heavy_smooth_rigid", "Weight + roughness", 4, 0.50, 9.10),
    ("light_rough_rigid", "light_rough_soft", "Hardness", 2, 1.00, 0.10),
    ("light_smooth_rigid", "heavy_rough_rigid", "Weight + roughness", 4, 0.00, 5.21),
    ("light_smooth_rigid", "heavy_rough_soft", "All 3", 3, 2 / 3, 0.25),
    ("light_smooth_rigid", "heavy_smooth_rigid", "Weight", 4, 0.75, 0.20),
    ("light_smooth_soft", "heavy_rough_rigid", "All 3", 4, 0.75, 0.76),
    ("light_smooth_soft", "heavy_smooth_rigid", "Weight + hardness", 7, 6 / 7, 0.71),
    ("light_smooth_soft", "light_rough_soft", "Roughness", 4, 0.50, 0.47),
]

COLORS = {
    "Weight": "#277DA1",
    "Roughness": "#F9C74F",
    "Hardness": "#43AA8B",
    "Weight + roughness": "#9B5DE5",
    "Weight + hardness": "#F9844A",
    "All 3": "#E45756",
}


def short_name(name: str) -> str:
    weight, roughness, hardness = name.split("_")
    return f"{'H' if weight == 'heavy' else 'L'}-{'R' if roughness == 'rough' else 'S'}-{'Ri' if hardness == 'rigid' else 'So'}"


def main() -> None:
    output = Path(__file__).with_name("provisional_slot_validation_oof.png")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    labels = [f"{short_name(reference)}  ->  {short_name(distractor)}" for reference, distractor, *_ in ROWS]
    scores = [score * 100 for *_, score, _ in ROWS]
    y = list(range(len(ROWS)))
    colors = [COLORS[changed] for _, _, changed, *_ in ROWS]
    bars = ax.barh(y, scores, color=colors, height=0.64)

    ax.axvline(50, color="#555555", lw=1.6, ls="--", zorder=0)
    ax.text(50.8, len(ROWS) - 0.38, "Chance = 50%", color="#555555", va="bottom", fontsize=11)

    for bar, (_, _, changed, n, score, margin) in zip(bars, ROWS):
        x = bar.get_width()
        label = f"{score * 100:.1f}%   n={n}   margin={margin:.2f}   [{changed}]"
        ax.text(min(x + 1.2, 101.0), bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontsize=11)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xticks(range(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    ax.set_xlabel("5-fold out-of-fold reference-to-candidate matching accuracy")
    fig.text(
        0.125,
        0.968,
        "Provisional Composable Tactile-Slot Expression (v0)",
        fontsize=20,
        fontweight="medium",
    )
    fig.text(
        0.125,
        0.932,
        "39 valid episodes; 9 invalid three-object probes excluded. This is a diagnostic of the current public-probe expression, not a benchmark result.",
        fontsize=11,
        color="#444444",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)

    handles = [Line2D([0], [0], color=color, lw=8, label=label) for label, color in COLORS.items()]
    ax.legend(handles=handles, title="Changed slot(s)", ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    fig.text(0.125, 0.035, "Abbreviations: H/L = heavy/light; R/S = rough/smooth; Ri/So = rigid/soft.", fontsize=10, color="#555555")

    fig.subplots_adjust(left=0.27, right=0.98, top=0.89, bottom=0.18)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(output)


if __name__ == "__main__":
    main()
