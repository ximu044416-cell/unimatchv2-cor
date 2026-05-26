import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

ANALYSIS_ROOT = Path(r"/RUN7/analysis")
PRED_ROOT = ANALYSIS_ROOT / "3D_Predictions"
OUT_ROOT = ANALYSIS_ROOT / "Qualitative_Results"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_TAGS = ["best_model", "swa_offline"]

# 如果你想手动指定病例，就把 AUTO_SELECT_CASES 改成 False，并填 TARGET_CASES
AUTO_SELECT_CASES = True
TARGET_CASES = [
    "0011526400_20231029130246",
    "0004025950_20220804193615",
    "0009497917_20221110203204"
]


def window_image_2_98(image):
    valid_pixels = image[image > 0]
    if len(valid_pixels) == 0:
        return np.zeros_like(image, dtype=np.uint8)

    p2, p98 = np.percentile(valid_pixels, (2, 98))
    if p98 - p2 < 1e-5:
        p98 = p2 + 1e-5

    image_clipped = np.clip(image, p2, p98)
    image_normalized = (image_clipped - p2) / (p98 - p2) * 255.0
    return image_normalized.astype(np.uint8)


def get_best_slice_and_bbox(gt_array, pred_array, margin=120):
    union_mask = (gt_array > 0) | (pred_array > 0)
    slice_areas = np.sum(union_mask, axis=(1, 2))
    best_z = np.argmax(slice_areas)

    best_mask = union_mask[best_z]
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


def create_traffic_light_overlay(img_gray, gt_mask, pred_mask, alpha=0.45):
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = img_color.copy()

    tp = (gt_mask > 0) & (pred_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)
    fn = (gt_mask > 0) & (pred_mask == 0)

    overlay[tp] = [0, 255, 0]
    overlay[fp] = [255, 0, 0]
    overlay[fn] = [0, 255, 255]

    colored_region = tp | fp | fn
    result = img_color.copy()
    result[colored_region] = cv2.addWeighted(overlay, alpha, img_color, 1 - alpha, 0)[colored_region]
    return result


def draw_contours(img_gray, mask, color_rgb):
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours, -1, color_rgb, 2)
    return img_color


def auto_pick_cases():
    """
    从 best_model 的 clinical report 自动选：
    1 个高 Dice
    1 个中位 Dice
    1 个低 Dice（但不是 Perfect TN）
    """
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


def generate_figure_for_model(model_tag, target_cases):
    pred_dir = PRED_ROOT / model_tag
    output_dir = OUT_ROOT / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    col_titles = ["Raw STIR MRI (ROI)", "Ground Truth Contour", "AI Prediction Contour", "TP/FP/FN Evaluation"]

    for row_idx, patient_id in enumerate(target_cases):
        img_path = list(pred_dir.rglob(f"{patient_id}_Image.nii.gz"))[0]
        gt_path = list(pred_dir.rglob(f"{patient_id}_GT.nii.gz"))[0]
        pred_path = list(pred_dir.rglob(f"{patient_id}_Pred.nii.gz"))[0]

        img_array = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        gt_array = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        pred_array = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

        best_z, y1, y2, x1, x2 = get_best_slice_and_bbox(gt_array, pred_array, margin=120)

        slice_img = window_image_2_98(img_array[best_z])
        slice_gt = gt_array[best_z]
        slice_pred = pred_array[best_z]

        roi_img = slice_img[y1:y2, x1:x2]
        roi_gt = slice_gt[y1:y2, x1:x2]
        roi_pred = slice_pred[y1:y2, x1:x2]

        img_gt_contour = draw_contours(roi_img, roi_gt, (0, 255, 0))
        img_pred_contour = draw_contours(roi_img, roi_pred, (255, 0, 0))
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
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=14, fontweight='bold', pad=10)

            if col_idx == 0:
                case_labels = ["Case 1", "Case 2", "Case 3"]
                ax.text(
                    -0.1, 0.5, f"{case_labels[row_idx]}\n{patient_id}",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight='bold',
                    va='center',
                    ha='right',
                    rotation=90
                )

            if row_idx == 0 and col_idx == 3:
                import matplotlib.patches as mpatches
                legend_elements = [
                    mpatches.Patch(color='#00FF00', label='True Positive', alpha=0.6),
                    mpatches.Patch(color='#FF0000', label='False Positive', alpha=0.6),
                    mpatches.Patch(color='#00FFFF', label='False Negative', alpha=0.6)
                ]
                ax.legend(handles=legend_elements, loc='upper right', prop={'size': 9}, framealpha=0.8)

    pdf_path = output_dir / f"{model_tag}_Qualitative_Results.pdf"
    png_path = output_dir / f"{model_tag}_Qualitative_Results.png"
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
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

    print("\n🎉 双模型 qualitative figure 已全部生成。")


if __name__ == "__main__":
    main()