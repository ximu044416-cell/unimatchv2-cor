import os
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

# =========================================================
# 路径
# =========================================================
PROJECT_ROOT = Path(r"F:\cor\RUN8")
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_2 import config
from models.dinov2_unet import DINOUNet
from data.dataset_2 import UniMatchDataset2

DEVICE = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

WEIGHT_PATH = Path(r"F:\cor\RUN8\logs\UniMatch_Cor_RUN8_new\best_model.pth")
OUT_ROOT = Path(r"F:\cor\RUN8\down_8_NEW\down_test")
OUT_3D_DIR = OUT_ROOT / "reconstructed_test40"
MANIFEST_PATH = OUT_ROOT / "test40_manifest.csv"

# 主分析 + 敏感性分析
THRESHOLDS = [0.80, 0.75]

# =========================================================
# TTA 推理
# =========================================================
@torch.no_grad()
def sliding_window_inference_tta(model, image, window_size=(518, 518), overlap=0.5):
    b, c, h, w = image.shape
    tile_h, tile_w = window_size
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))

    def get_gaussian(size, sigma_scale=1.0 / 8):
        k_h, k_w = size
        center_coords = [k_h // 2, k_w // 2]
        sigmas = [k_h * sigma_scale, k_w * sigma_scale]
        y, x = np.ogrid[:k_h, :k_w]
        h_norm = (y - center_coords[0]) / sigmas[0]
        w_norm = (x - center_coords[1]) / sigmas[1]
        g = np.exp(-(h_norm ** 2 + w_norm ** 2) / 2)
        return torch.from_numpy(g).float().to(image.device)

    gaussian_weight = get_gaussian(window_size)
    output_sum = torch.zeros((b, config.NUM_CLASSES, h, w), device=image.device)
    weight_sum = torch.zeros((b, config.NUM_CLASSES, h, w), device=image.device)

    h_steps = math.ceil((h - tile_h) / stride_h) + 1
    w_steps = math.ceil((w - tile_w) / stride_w) + 1

    def predict_patch_tta(patch):
        pred = torch.softmax(model(patch), dim=1)

        pred_h = torch.softmax(model(torch.flip(patch, [3])), dim=1)
        pred += torch.flip(pred_h, [3])

        pred_v = torch.softmax(model(torch.flip(patch, [2])), dim=1)
        pred += torch.flip(pred_v, [2])

        return pred / 3.0

    for i in range(h_steps):
        for j in range(w_steps):
            h_start = min(i * stride_h, h - tile_h)
            w_start = min(j * stride_w, w - tile_w)
            h_end = h_start + tile_h
            w_end = w_start + tile_w

            patch = image[:, :, h_start:h_end, w_start:w_end]
            prob_patch = predict_patch_tta(patch)

            output_sum[:, :, h_start:h_end, w_start:w_end] += prob_patch * gaussian_weight
            weight_sum[:, :, h_start:h_end, w_start:w_end] += gaussian_weight

    return output_sum / weight_sum


def collect_test40_patients():
    test_dir = Path(config.TEST_DIR)
    if not test_dir.exists():
        raise FileNotFoundError(f"❌ config.TEST_DIR 不存在：{test_dir}")

    patient_dict = {}
    manifest_rows = []

    for case_dir in sorted(test_dir.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith("."):
            continue

        patient_id = case_dir.name
        slice_list = sorted(case_dir.glob("*_data.npy"))
        if len(slice_list) == 0:
            continue

        patient_dict[patient_id] = {
            "split": "test40",
            "slice_list": [p.absolute() for p in slice_list]
        }

        manifest_rows.append({
            "Patient_ID": patient_id,
            "Source_Split": "test40",
            "Num_Slices": len(slice_list)
        })

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["Patient_ID"])
    return patient_dict, manifest_df


def reconstruct_test40():
    OUT_3D_DIR.mkdir(parents=True, exist_ok=True)

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到权重：{WEIGHT_PATH}")

    patient_dict, manifest_df = collect_test40_patients()
    manifest_df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("🚀 RUN8 test40 最终冻结重建启动")
    print(f"📥 权重: {WEIGHT_PATH}")
    print(f"📂 输出: {OUT_3D_DIR}")
    print(f"📋 test40 患者数: {len(patient_dict)}")
    print(f"🎯 阈值: {THRESHOLDS}")
    print(f"📄 manifest: {MANIFEST_PATH}")
    print("=" * 80)

    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE), strict=True)
    model.eval()

    with torch.no_grad():
        for patient_id, meta in tqdm(patient_dict.items(), desc="test40 重建进度"):
            slice_list = meta["slice_list"]

            patient_output_dir = OUT_3D_DIR / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)

            patient_dataset = UniMatchDataset2(slice_list, mode="val")
            patient_loader = DataLoader(patient_dataset, batch_size=1, shuffle=False, num_workers=0)

            patient_images = []
            patient_gts = []
            patient_preds_dict = {t: [] for t in THRESHOLDS}

            for data_tuple, slice_path in zip(patient_loader, slice_list):
                img_tensor, _ = data_tuple
                img_tensor = img_tensor.to(DEVICE)

                orig_img = np.load(slice_path)
                orig_h, orig_w = orig_img.shape[:2]

                # 与 dev164 一致：取第1通道作为 radiomics image
                if orig_img.ndim == 3:
                    patient_images.append(orig_img[..., 0].astype(np.float32))
                else:
                    patient_images.append(orig_img.astype(np.float32))

                with autocast("cuda" if DEVICE.type == "cuda" else "cpu", enabled=(DEVICE.type == "cuda")):
                    prob_map = sliding_window_inference_tta(
                        model,
                        img_tensor,
                        window_size=(config.IMG_SIZE, config.IMG_SIZE),
                        overlap=0.5
                    )

                pred_prob = prob_map[:, 1:2, :, :]
                prob_map_np = pred_prob.cpu().numpy().astype(np.float32)[0, 0]

                pad_h = max(0, config.IMG_SIZE - orig_h)
                pad_w = max(0, config.IMG_SIZE - orig_w)
                if pad_h > 0 or pad_w > 0:
                    pad_top = pad_h // 2
                    pad_left = pad_w // 2
                    prob_map_np = prob_map_np[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]

                label_path = str(slice_path).replace("_data.npy", "_label.npy")
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"❌ 缺少 GT label: {label_path}")

                gt_mask_np = np.load(label_path).astype(np.uint8)
                patient_gts.append(gt_mask_np)

                for t in THRESHOLDS:
                    mask_t = (prob_map_np > t).astype(np.uint8)
                    patient_preds_dict[t].append(mask_t)

            vol_img = np.stack(patient_images, axis=0)
            vol_gt = np.stack(patient_gts, axis=0)

            sitk_img = sitk.GetImageFromArray(vol_img)
            sitk_gt = sitk.GetImageFromArray(vol_gt)
            sitk_img.SetSpacing((1.0, 1.0, 1.0))
            sitk_gt.SetSpacing((1.0, 1.0, 1.0))

            sitk.WriteImage(sitk_img, str(patient_output_dir / f"{patient_id}_Image.nii.gz"))
            sitk.WriteImage(sitk_gt, str(patient_output_dir / f"{patient_id}_GT.nii.gz"))

            with open(patient_output_dir / "meta.txt", "w", encoding="utf-8") as f:
                f.write(f"Patient_ID: {patient_id}\n")
                f.write("Source_Split: test40\n")
                f.write(f"Num_Slices: {len(slice_list)}\n")
                f.write(f"Thresholds: {THRESHOLDS}\n")

            for t in THRESHOLDS:
                t_str = f"{int(round(t * 100)):03d}"
                vol_pred_t = np.stack(patient_preds_dict[t], axis=0)
                sitk_pred_t = sitk.GetImageFromArray(vol_pred_t)
                sitk_pred_t.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(sitk_pred_t, str(patient_output_dir / f"{patient_id}_Pred_{t_str}.nii.gz"))

    print("\n🎉 test40 最终冻结重建完成。")


if __name__ == "__main__":
    reconstruct_test40()