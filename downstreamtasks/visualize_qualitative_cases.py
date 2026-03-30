import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# 设置高分辨率和美观字体
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'


def window_image_2_98(image):
    """医学图像专用的 2%~98% 极值灰度拉伸，纯净提亮 STIR 序列"""
    # 🔥 导师级修复：只针对有信号的真实组织计算分位数，彻底排除纯黑背景的干扰！
    valid_pixels = image[image > 0]

    if len(valid_pixels) == 0:  # 防御性编程：如果全黑则直接返回
        return np.zeros_like(image, dtype=np.uint8)

    p2, p98 = np.percentile(valid_pixels, (2, 98))

    # 防止除以 0 (当图像亮度极其单一时)
    if p98 - p2 < 1e-5:
        p98 = p2 + 1e-5

    image_clipped = np.clip(image, p2, p98)
    image_normalized = (image_clipped - p2) / (p98 - p2) * 255.0
    return image_normalized.astype(np.uint8)


def get_best_slice_and_bbox(gt_array, pred_array, margin=120):
    """自动寻找病灶最大的一层切片，并计算带 120 像素边距的 ROI 边界框"""
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
    """生成顶刊灵魂：交通灯红绿青重叠图"""
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = img_color.copy()

    tp = (gt_mask > 0) & (pred_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)
    fn = (gt_mask > 0) & (pred_mask == 0)

    # 填充颜色：TP纯绿，FP警告红，FN亮青色
    overlay[tp] = [0, 255, 0]
    overlay[fp] = [255, 0, 0]
    overlay[fn] = [0, 255, 255]

    colored_region = tp | fp | fn
    result = img_color.copy()
    result[colored_region] = cv2.addWeighted(overlay, alpha, img_color, 1 - alpha, 0)[colored_region]

    return result


def draw_contours(img_gray, mask, color_rgb):
    """画空心轮廓边缘线"""
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours, -1, color_rgb, 2)
    return img_color


def generate_publication_figure():
    # 注意根据你的实际路径调整
    PRED_DIR = Path(r"F:\Dinov2_data\3D_Predictions")
    OUTPUT_DIR = Path(r"F:\Dinov2_data\Clinical_Analysis")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 锁定猎物
    target_cases = [
        "0011526400_20231029130246",  # Easy / Top Performer
        "0004025950_20220804193615",  # Normal / Excellent Case
        "0009497917_20221110203204"  # Hard / Huge FP Outlier
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), gridspec_kw={'wspace': 0.02, 'hspace': 0.05})

    col_titles = ["Raw STIR MRI (ROI)", "Ground Truth Contour", "AI Prediction Contour", "TP/FP/FN Evaluation"]

    for row_idx, patient_id in enumerate(target_cases):
        # 🔥 修复：使用 rglob 在子文件夹中寻找对应文件
        img_path = list(PRED_DIR.rglob(f"{patient_id}_Image.nii.gz"))[0]
        gt_path = list(PRED_DIR.rglob(f"{patient_id}_GT.nii.gz"))[0]
        pred_path = list(PRED_DIR.rglob(f"{patient_id}_Pred.nii.gz"))[0]

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
                case_labels = ["Case 1 (Top Performer)", "Case 2 (Typical Edema)", "Case 3 (FP Analysis)"]
                ax.text(-0.1, 0.5, case_labels[row_idx], transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

            if row_idx == 0 and col_idx == 3:
                import matplotlib.patches as mpatches
                legend_elements = [
                    mpatches.Patch(color='#00FF00', label='True Positive', alpha=0.6),
                    mpatches.Patch(color='#FF0000', label='False Positive', alpha=0.6),
                    mpatches.Patch(color='#00FFFF', label='False Negative', alpha=0.6)
                ]
                ax.legend(handles=legend_elements, loc='upper right', prop={'size': 9}, framealpha=0.8)

    pdf_path = OUTPUT_DIR / "Qualitative_Results.pdf"
    png_path = OUTPUT_DIR / "Qualitative_Results.png"
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.1, dpi=300)

    print(f"\n🎨 顶刊级视觉大图已生成完毕！")
    print(f"👉 请打开查看: {png_path}")


if __name__ == "__main__":
    generate_publication_figure()