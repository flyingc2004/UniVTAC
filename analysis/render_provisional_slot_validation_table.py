from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

ROWS = [
    ("Weight", "heavy_rough_soft → light_rough_soft", 2, "50.0%", "0.25"),
    ("Weight", "heavy_smooth_rigid → light_smooth_rigid", 4, "50.0%", "0.25"),
    ("Weight", "light_smooth_rigid → heavy_smooth_rigid", 4, "75.0%", "0.20"),
    ("Roughness", "light_smooth_soft → light_rough_soft", 4, "50.0%", "0.47"),
    ("Hardness", "light_rough_rigid → light_rough_soft", 2, "100.0%", "0.10"),
    ("Weight + roughness", "light_rough_rigid → heavy_smooth_rigid", 4, "50.0%", "9.10"),
    ("Weight + roughness", "light_smooth_rigid → heavy_rough_rigid", 4, "0.0%", "5.21"),
    ("Weight + hardness", "light_smooth_soft → heavy_smooth_rigid", 7, "85.7%", "0.71"),
    ("All three slots", "heavy_smooth_rigid → light_rough_soft", 1, "0.0%", "0.44"),
    ("All three slots", "light_smooth_rigid → heavy_rough_soft", 3, "66.7%", "0.25"),
    ("All three slots", "light_smooth_soft → heavy_rough_rigid", 4, "75.0%", "0.76"),
]

COLORS = {
    "Weight": "#DDEBF7",
    "Roughness": "#FFF2CC",
    "Hardness": "#D9EAD3",
    "Weight + roughness": "#E4DFEC",
    "Weight + hardness": "#FCE4D6",
    "All three slots": "#F4CCCC",
}


def main() -> None:
    output = Path(__file__).with_name("provisional_slot_validation_table.png")
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    headers = ["Changed slot(s)", "Reference → distractor", "n", "5-fold OOF accuracy", "Mean margin"]
    table = ax.table(
        cellText=[[slot, pair, str(n), accuracy, margin] for slot, pair, n, accuracy, margin in ROWS],
        colLabels=headers,
        cellLoc="left",
        colLoc="center",
        colWidths=[0.19, 0.46, 0.07, 0.17, 0.11],
        bbox=[0.035, 0.20, 0.93, 0.68],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.55)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#1F4E78")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("center")
        else:
            slot = ROWS[row - 1][0]
            cell.set_facecolor(COLORS[slot] if column == 0 else "#FFFFFF")
            if column in {2, 3, 4}:
                cell.get_text().set_ha("center")
            else:
                cell.get_text().set_ha("left")
            if column == 3 and ROWS[row - 1][3] in {"0.0%", "50.0%"}:
                cell.get_text().set_color("#9C0006")
                cell.get_text().set_weight("bold")

    fig.text(
        0.035,
        0.94,
        "Provisional Composable Tactile-Slot Validation",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.905,
        "5-fold out-of-fold reference-to-candidate matching. Rows are ordered by the changed physical slot(s).",
        fontsize=12,
        color="#444444",
    )
    fig.text(0.035, 0.135, "结论", fontsize=15, fontweight="bold", fontproperties=FONT, color="#9C0006")
    fig.text(
        0.035,
        0.102,
        "当前 v0 触觉槽位表达在 39 条有效轨迹上的 5-fold OOF 准确率为 59.0%，未能稳定超过随机二选一（50%）。",
        fontsize=12,
        fontproperties=FONT,
        color="#222222",
    )
    fig.text(
        0.035,
        0.072,
        "重量与粗糙度单槽位尚不可稳定区分；硬度出现初步信号但样本不足。高 margin 的错误案例说明当前特征仍受抓取状态耦合影响，",
        fontsize=12,
        fontproperties=FONT,
        color="#222222",
    )
    fig.text(
        0.035,
        0.042,
        "因此暂不能作为可靠的可组合触觉 memory；下一步应设计属性解耦的受控 probe，再重新验证。",
        fontsize=12,
        fontproperties=FONT,
        color="#222222",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(output)


if __name__ == "__main__":
    main()
