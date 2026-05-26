import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 0. Save path
# =========================
SAVE_DIR = Path(r"F:\cor\RUN8\down_8_NEW\figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. Global style
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.2,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =========================
# 2. Panel A structural data from RUN8 dev164 threshold summary
# =========================
thresholds = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])

dice_scores = np.array([
    0.463915,
    0.464358,
    0.462621,
    0.455906,
    0.438967,
    0.436653,
    0.434773,
    0.432111,
    0.427994,
    0.418726
])

rve_errors = np.array([
    5.126765,
    0.374187,
    -5.206708,
    -13.231123,
    -22.266829,
    -25.825536,
    -29.078431,
    -32.532508,
    -36.684652,
    -43.302808
])


# =========================
# 3. Panel B downstream dev164 results
# =========================
auc_values = np.array([0.946, 0.945, 0.946, 0.949, 0.954, 0.953, 0.952, 0.954, 0.949, 0.945])
acc_values = np.array([0.860, 0.872, 0.878, 0.896, 0.896, 0.896, 0.909, 0.902, 0.896, 0.896])
sen_values = np.array([0.940, 0.940, 0.940, 0.940, 0.940, 0.920, 0.920, 0.920, 0.900, 0.880])
spe_values = np.array([0.825, 0.842, 0.851, 0.877, 0.877, 0.886, 0.904, 0.895, 0.895, 0.904])
delong_values = np.array([0.5454, 0.4924, 0.5728, 0.7140, 0.7890, 0.9075, 0.9841, 0.8492, 0.7591, 0.5969])
rescue_values = np.array([9, 10, 13, 17, 27, 27, 28, 28, 29, 30])

GT_AUC = 0.952
GT_ACC = 0.921
GT_SEN = 0.920
GT_SPE = 0.921
GT_CUTOFF = 0.3536


# =========================
# 4. Colors
# =========================
COLOR_DICE = "#C0392B"
COLOR_RVE = "#2E86C1"
COLOR_STRUCT = "#F39C12"
COLOR_FINAL = "#B23A48"
COLOR_SENS = "#7F8C8D"

COLOR_AUC = "#2C3E50"
COLOR_ACC = "#2E86C1"
COLOR_SEN = "#27AE60"
COLOR_SPE = "#8E44AD"
GRID_COLOR = "#D9D9D9"


# =========================
# 5. Figure layout
# =========================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 6.2))

fig.subplots_adjust(
    left=0.07,
    right=0.96,
    top=0.88,
    bottom=0.18,
    wspace=0.23
)


# =========================
# 6. Panel A: structural calibration
# =========================
axA2 = axA.twinx()

line_dice, = axA.plot(
    thresholds,
    dice_scores,
    marker="o",
    markersize=6.5,
    linewidth=2.5,
    color=COLOR_DICE,
    label="Volumetric DSC"
)

bars = axA2.bar(
    thresholds,
    rve_errors,
    width=0.022,
    color=COLOR_RVE,
    edgecolor=COLOR_RVE,
    linewidth=1.0,
    alpha=0.30,
    label="RVE"
)

axA2.axhline(0, color="black", linestyle="--", linewidth=1.1)

# -------------------------
# Structural volume-fidelity point: threshold 0.55
# -------------------------
idx_055 = np.where(np.isclose(thresholds, 0.55))[0][0]

axA.plot(
    0.55,
    dice_scores[idx_055],
    marker="*",
    markersize=15,
    color="#FFD700",
    markeredgecolor=COLOR_STRUCT,
    markeredgewidth=1.1,
    label="Volume-fidelity candidate"
)

bars[idx_055].set_alpha(0.75)
bars[idx_055].set_color(COLOR_STRUCT)
bars[idx_055].set_edgecolor(COLOR_STRUCT)

axA.text(
    0.555,
    dice_scores[idx_055] - 0.006,
    "0.55\nVolume-fidelity candidate",
    fontsize=8.5,
    color=COLOR_STRUCT,
    ha="left",
    va="top"
)

# -------------------------
# Frozen downstream deployment threshold: 0.80
# (Removed 0.75 from Panel A to keep structural focus clean)
# -------------------------
axA.axvline(
    0.80,
    color=COLOR_FINAL,
    linestyle="-",
    linewidth=1.9,
    alpha=0.90
)

axA.text(
    0.805,
    0.477,
    "0.80 downstream-frozen\nthreshold",
    rotation=90,
    va="top",
    ha="left",
    fontsize=8.5,
    color=COLOR_FINAL
)

axA.set_title("A. Structural threshold calibration", pad=10, fontweight="bold")
axA.set_xlabel("Prediction probability threshold", fontweight="bold")
axA.set_ylabel("Volumetric DSC", color=COLOR_DICE, fontweight="bold")
axA2.set_ylabel("Relative volume error (RVE, %)", color=COLOR_RVE, fontweight="bold")

axA.tick_params(axis="y", labelcolor=COLOR_DICE)
axA2.tick_params(axis="y", labelcolor=COLOR_RVE)

# Updated axis ranges for new RUN8 structural values
axA.set_ylim(0.41, 0.48)
axA2.set_ylim(-50, 10)

axA.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.42, color=GRID_COLOR)
axA.spines["top"].set_visible(False)
axA2.spines["top"].set_visible(False)

handles1, labels1 = axA.get_legend_handles_labels()
handles2, labels2 = axA2.get_legend_handles_labels()

axA.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="lower left",
    frameon=True,
    framealpha=0.94,
    fontsize=8.8,
    borderpad=0.4
)


# =========================
# 7. Panel B: downstream threshold comparison
# =========================
axB.plot(thresholds, auc_values, marker="o", linewidth=2.3, color=COLOR_AUC, label="AUC")
axB.plot(thresholds, acc_values, marker="s", linewidth=2.1, color=COLOR_ACC, label="Accuracy")
axB.plot(thresholds, sen_values, marker="^", linewidth=2.1, color=COLOR_SEN, label="Sensitivity")
axB.plot(thresholds, spe_values, marker="D", linewidth=2.1, color=COLOR_SPE, label="Specificity")

# Highlight 0.75 and 0.80
axB.axvline(0.75, color=COLOR_SENS, linestyle="--", linewidth=1.5, alpha=0.75)
axB.axvline(0.80, color=COLOR_FINAL, linestyle="-", linewidth=1.9, alpha=0.90)

idx_080 = np.where(np.isclose(thresholds, 0.80))[0][0]
axB.scatter(
    [0.80],
    [auc_values[idx_080]],
    s=95,
    color=COLOR_FINAL,
    edgecolor="black",
    zorder=5
)

axB.text(
    0.805,
    auc_values[idx_080] + 0.004,
    "Frozen threshold 0.80",
    fontsize=9,
    color=COLOR_FINAL,
    ha="left",
    va="bottom"
)

axB.text(
    0.74,
    0.825,
    "0.75 sensitivity",
    rotation=90,
    va="bottom",
    ha="right",
    fontsize=8.5,
    color=COLOR_SENS
)


axB.set_title("B. Development-cohort downstream threshold comparison", pad=10, fontweight="bold")
axB.set_xlabel("Prediction probability threshold", fontweight="bold")
axB.set_ylabel("Performance metric score", fontweight="bold")

axB.set_xlim(0.485, 0.965)
axB.set_ylim(0.80, 0.965)
axB.set_xticks(thresholds)
axB.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)

axB.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.42, color=GRID_COLOR)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

axB.legend(
    loc="lower left",
    frameon=True,
    framealpha=0.94,
    fontsize=8.8,
    borderpad=0.45
)

textstr = "\n".join([
    "GT-based reference:",
    f"AUC={GT_AUC:.3f}, ACC={GT_ACC:.3f}",
    f"SEN={GT_SEN:.3f}, SPE={GT_SPE:.3f}",
    "",
    "At threshold 0.80:",
    f"AUC={auc_values[idx_080]:.3f}, ACC={acc_values[idx_080]:.3f}",
    f"SEN={sen_values[idx_080]:.3f}, SPE={spe_values[idx_080]:.3f}",
    f"DeLong p={delong_values[idx_080]:.4f}"
])

axB.text(
    0.98,
    0.05,
    textstr,
    transform=axB.transAxes,
    fontsize=8.7,
    ha="right",
    va="bottom",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="#BDC3C7",
        alpha=0.94
    )
)


# =========================
# 8. Save
# =========================
png_path = SAVE_DIR / "Figure5_threshold_calibration_final1.png"
pdf_path = SAVE_DIR / "Figure5_threshold_calibration_final1.pdf"
svg_path = SAVE_DIR / "Figure5_threshold_calibration_final1.svg"
tif_path = SAVE_DIR / "Figure5_threshold_calibration_final1.tif"

fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(tif_path, bbox_inches="tight", dpi=600)

plt.show()

print("Saved to:")
print(png_path)
print(pdf_path)
print(svg_path)
print(tif_path)