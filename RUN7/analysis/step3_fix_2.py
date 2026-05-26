import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import matplotlib.patches as mpatches

# =========================
# Global settings
# =========================
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Arial"

ANALYSIS_ROOT = Path(r"/RUN7/analysis")
PRED_ROOT = ANALYSIS_ROOT / "3D_Predictions"
OUT_ROOT = ANALYSIS_ROOT / "Qualitative_Results_2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 只保留最终模型
MODEL_TAGS = ["best_model"]   # 如果你最终定稿用的是 swa_offline，就改成 ["swa_offline"]

# 固定主文展示病例，不再自动选
AUTO_SELECT_CASES = False
TARGET_CASES = [
    "0011526400_20231029130246",
    "0004025950_20220804193615",
    "0009497917_20221110203204"
]

# =========================
# Visualization options
# =========================
FLIP_VERTICAL = True
FLIP_HORIZONTAL = False
MARGIN = 120

COLUMN_TITLES = [
    "Raw T2-STIR ROI",
    "Ground-truth contour",
    "AI-predicted contour",
    "Error decomposition (TP/FP/FN)"
]

CASE_LABELS = [
    "Case 1\nHigh-agreement",
    "Case 2\nTypical lesion",
    "Case 3\nError-prone"
]


# =========================
# Windowing: volume-wise
# =========================
def get_volume_window_params(image_3d):
    """
    基于整例 volume 的非零像素计算 window，避免单层 slice windowing 导致困难病例视觉风格异常。
    """
    valid_pixels = image_3d[image_3d > 0]
    if len(valid_pixels) == 0:
        return 0.0, 1.0

    p2, p98 = np.percentile(valid_pixels, (2, 98))
    if p98 - p2 < 1e-5:
        p98 = p2 + 1e-5
    return float(p2), float(p98)


def apply_window_with_params(image_2d, p2, p98):
    image_clipped = np.clip(image_2d, p2, p98)
    image_normalized = (image_clipped - p2) / (p98 - p2) * 255.0
    return image_normalized.astype(np.uint8)


# =========================
# Slice selection
# =========================
def get_best_slice_and_bbox(gt_array, pred_array, margin=120):
    """
    优先选择 GT 面积最大的层面作为展示层面；
    仅当 GT 全空时，才回退到 Pred 面积最大的层面。
    这样能避免 error-prone case 因为大块 FP 把显示切片带偏。
    """
    gt_slice_areas = np.sum(gt_array > 0, axis=(1, 2))
    pred_slice_areas = np.sum(pred_array > 0, axis=(1, 2))

    if np.max(gt_slice_areas) > 0:
        best_z = int(np.argmax(gt_slice_areas))
    elif np.max(pred_slice_areas) > 0:
        best_z = int(np.argmax(pred_slice_areas))
    else:
        best_z = gt_array.shape[0] // 2

    # bbox 在选中的 best_z 上，用 GT 和 Pred 的并集来裁剪
    best_mask = (gt_array[best_z] > 0) | (pred_array[best_z] > 0)

    if not np.any(best_mask):
        h, w = best_mask.shape
        return best_z, 0, h, 0, w

    rows = np.any(best_mask, axis=1)
    cols = np.any(best_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    h, w = best_mask.shape
    y_min = max(0, y_min - margin)
    y_max = min(h, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin)

    return best_z, y_min, y_max, x_min, x_max


# =========================
# Orientation
# =========================
def apply_orientation(img2d, gt2d, pred2d, flip_vertical=True, flip_horizontal=False):
    if flip_vertical:
        img2d = np.flipud(img2d)
        gt2d = np.flipud(gt2d)
        pred2d = np.flipud(pred2d)

    if flip_horizontal:
        img2d = np.fliplr(img2d)
        gt2d = np.fliplr(gt2d)
        pred2d = np.fliplr(pred2d)

    return img2d, gt2d, pred2d


# =========================
# Visualization helpers
# =========================
def create_traffic_light_overlay(img_gray, gt_mask, pred_mask, alpha=0.45):
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = img_color.copy()

    tp = (gt_mask > 0) & (pred_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)
    fn = (gt_mask > 0) & (pred_mask == 0)

    # TP = green, FP = red, FN = cyan
    overlay[tp] = [0, 255, 0]
    overlay[fp] = [255, 0, 0]
    overlay[fn] = [0, 255, 255]

    colored_region = tp | fp | fn
    result = img_color.copy()
    result[colored_region] = cv2.addWeighted(
        overlay, alpha, img_color, 1 - alpha, 0
    )[colored_region]
    return result


def draw_contours(img_gray, mask, color_rgb, thickness=2):
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours, -1, color_rgb, thickness)
    return img_color


# =========================
# Optional auto-pick
# =========================
def auto_pick_cases():
    report_path = ANALYSIS_ROOT / "Clinical_Analysis" / "best_model" / "best_model_Clinical_Metrics_Report.xlsx"
    if not report_path.exists():
        print("⚠️ 找不到 clinical report，回退到手动 TARGET_CASES。")
        return TARGET_CASES

    df = pd.read_excel(report_path, sheet_name="Sorted_by_3D_Dice")
    df = df[(df["Status"] != "Perfect TN")].copy()
    if len(df) < 3:
        print("⚠️ 可用病例不足 3 个，回退到手动 TARGET_CASES。")
        return TARGET_CASES

    df = df.sort_values("3D_Dice", ascending=False).reset_index(drop=True)
    top_case = df.iloc[0]["Patient_ID"]
    mid_case = df.iloc[len(df) // 2]["Patient_ID"]
    low_case = df.iloc[-1]["Patient_ID"]

    return [top_case, mid_case, low_case]


# =========================
# Main figure generation
# =========================
def generate_figure_for_model(model_tag, target_cases):
    pred_dir = PRED_ROOT / model_tag
    output_dir = OUT_ROOT / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3, 4,
        figsize=(16, 12),
        gridspec_kw={"wspace": 0.02, "hspace": 0.08}
    )
    fig.patch.set_facecolor("white")

    for row_idx, patient_id in enumerate(target_cases):
        img_candidates = sorted(pred_dir.rglob(f"{patient_id}_Image.nii.gz"))
        gt_candidates = sorted(pred_dir.rglob(f"{patient_id}_GT.nii.gz"))
        pred_candidates = sorted(pred_dir.rglob(f"{patient_id}_Pred.nii.gz"))

        if not (img_candidates and gt_candidates and pred_candidates):
            raise FileNotFoundError(f"❌ 缺少病例 {patient_id} 的 Image/GT/Pred 文件。")

        img_path = img_candidates[0]
        gt_path = gt_candidates[0]
        pred_path = pred_candidates[0]

        print(f"\n[{model_tag}] patient = {patient_id}")
        print("img_path :", img_path)
        print("gt_path  :", gt_path)
        print("pred_path:", pred_path)

        img_itk = sitk.ReadImage(str(img_path))
        gt_itk = sitk.ReadImage(str(gt_path))
        pred_itk = sitk.ReadImage(str(pred_path))

        print("img size      :", img_itk.GetSize())
        print("img spacing   :", img_itk.GetSpacing())
        print("img components:", img_itk.GetNumberOfComponentsPerPixel())

        img_array = sitk.GetArrayFromImage(img_itk)
        gt_array = sitk.GetArrayFromImage(gt_itk)
        pred_array = sitk.GetArrayFromImage(pred_itk)

        # 关键改动：整例统一 window
        p2, p98 = get_volume_window_params(img_array)

        # 关键改动：优先用 GT 最大层面，而不是 union 最大层面
        best_z, y1, y2, x1, x2 = get_best_slice_and_bbox(gt_array, pred_array, margin=MARGIN)

        slice_img = apply_window_with_params(img_array[best_z], p2, p98)
        slice_gt = gt_array[best_z]
        slice_pred = pred_array[best_z]

        roi_img = slice_img[y1:y2, x1:x2]
        roi_gt = slice_gt[y1:y2, x1:x2]
        roi_pred = slice_pred[y1:y2, x1:x2]

        roi_img, roi_gt, roi_pred = apply_orientation(
            roi_img, roi_gt, roi_pred,
            flip_vertical=FLIP_VERTICAL,
            flip_horizontal=FLIP_HORIZONTAL
        )

        img_gt_contour = draw_contours(roi_img, roi_gt, (0, 255, 0), thickness=2)
        img_pred_contour = draw_contours(roi_img, roi_pred, (255, 0, 0), thickness=2)
        img_overlay = create_traffic_light_overlay(roi_img, roi_gt, roi_pred, alpha=0.45)

        views = [
            cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB),
            img_gt_contour,
            img_pred_contour,
            img_overlay
        ]

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.imshow(views[col_idx])
            ax.axis("off")

            if row_idx == 0:
                ax.set_title(COLUMN_TITLES[col_idx], fontsize=16, fontweight="bold", pad=12)

            if col_idx == 0:
                label_text = CASE_LABELS[row_idx]
                ax.text(
                    -0.08, 0.5, label_text,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90
                )

            if row_idx == 0 and col_idx == 3:
                legend_elements = [
                    mpatches.Patch(color="#00FF00", label="True Positive", alpha=0.6),
                    mpatches.Patch(color="#FF0000", label="False Positive", alpha=0.6),
                    mpatches.Patch(color="#00FFFF", label="False Negative", alpha=0.6)
                ]
                ax.legend(
                    handles=legend_elements,
                    loc="upper right",
                    bbox_to_anchor=(0.99, 0.99),
                    borderaxespad=0.10,
                    prop={"size": 9},
                    framealpha=0.92,
                    facecolor="white",
                    edgecolor="lightgray"
                )

    pdf_path = output_dir / "Figure3_qualitative_results.pdf"
    png_path = output_dir / "Figure3_qualitative_results.png"

    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)

    print(f"🎨 {model_tag} qualitative figure 已生成: {png_path}")


def main():
    if AUTO_SELECT_CASES:
        target_cases = auto_pick_cases()
    else:
        target_cases = TARGET_CASES

    print(f"📌 本次可视化病例: {target_cases}")

    for model_tag in MODEL_TAGS:
        generate_figure_for_model(model_tag, target_cases)

    print("\n🎉 Figure 3 qualitative figure 已全部生成。")


if __name__ == "__main__":
    main()