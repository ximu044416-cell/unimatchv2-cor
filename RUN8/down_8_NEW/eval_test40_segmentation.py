import os
import sys
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score

# =========================
# 0. 固定路径
# =========================
PROJECT_ROOT = Path(r"F:\cor\RUN8")
OUT_ROOT = Path(r"F:\cor\RUN8\down_8_NEW\down_1_test_dice")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = Path(r"F:\cor\RUN8\logs\UniMatch_Cor_RUN8_new\best_model.pth")

# 把 RUN8 项目根目录加入 import 路径
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_2 import config
from models.dinov2_unet import DINOUNet

# =========================
# 1. 运行参数
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 0

# 这里只做“固定阈值独立评估”，不做 test 上调参
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# 是否保存二值预测 mask（按阈值分别保存）
SAVE_BINARY_MASKS = True

# 是否保存前景概率图（只保存一份，和阈值无关）
SAVE_PROB_MAPS = False

# =========================
# 2. 测试集 Dataset
# =========================
class RUN8TestDataset(Dataset):
    def __init__(self, test_dir: Path):
        self.test_dir = Path(test_dir)
        self.img_size = config.IMG_SIZE

        self.clahe_transform = A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0)

        self.val_aug = A.Compose([
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            )
        ], additional_targets={'medsam_prior': 'mask'})

        self.norm = A.Compose([
            A.Normalize(
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD,
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ])

        self.items = []
        self._collect_items()

    def _collect_items(self):
        case_dirs = sorted([p for p in self.test_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
        if len(case_dirs) == 0:
            raise RuntimeError(f"test 目录为空：{self.test_dir}")

        for case_dir in case_dirs:
            data_files = sorted(case_dir.glob("*_data.npy"))
            for data_path in data_files:
                label_path = data_path.parent / data_path.name.replace("_data.npy", "_label.npy")
                if not label_path.exists():
                    raise FileNotFoundError(f"缺少 label 文件：{label_path}")

                self.items.append({
                    "case_id": case_dir.name,
                    "slice_id": data_path.stem.replace("_data", ""),
                    "data_path": data_path,
                    "label_path": label_path
                })

        if len(self.items) == 0:
            raise RuntimeError("没有在 test 中找到任何 *_data.npy")

    def __len__(self):
        return len(self.items)

    def _load_image_and_prior(self, data_path: Path):
        img_npy = np.load(data_path).astype(np.float32)
        h, w = img_npy.shape[:2]

        # 和训练一致：先 min-max，再 CLAHE
        img_min, img_max = img_npy.min(), img_npy.max()
        if img_max > img_min:
            img_npy = (img_npy - img_min) / (img_max - img_min)

        img_uint8 = (img_npy * 255.0).astype(np.uint8)
        img_uint8 = self.clahe_transform(image=img_uint8)["image"]
        img_npy = (img_uint8 / 255.0).astype(np.float32)

        prior_path = data_path.parent / data_path.name.replace("_data.npy", "_medsam2_prior.npy")
        if prior_path.exists():
            prob_map = np.load(prior_path).astype(np.float32)
            if prob_map.shape != (h, w):
                print(f"⚠️ prior 尺寸异常，已置零：{prior_path.name}, {prob_map.shape} vs {(h, w)}")
                prob_map = np.zeros((h, w), dtype=np.float32)
        else:
            prob_map = np.zeros((h, w), dtype=np.float32)

        return img_npy, prob_map

    def __getitem__(self, idx):
        item = self.items[idx]

        img_npy, prior_npy = self._load_image_and_prior(item["data_path"])
        mask_npy = np.load(item["label_path"]).astype(np.int64)

        val_res = self.val_aug(
            image=img_npy,
            mask=mask_npy,
            medsam_prior=prior_npy
        )

        norm_res = self.norm(image=val_res["image"], mask=val_res["mask"])
        img_tensor = norm_res["image"]  # (3,H,W)
        mask_tensor = norm_res["mask"].long()

        prior_tensor = torch.tensor(val_res["medsam_prior"], dtype=torch.float32).unsqueeze(0)
        fused_img = torch.cat([img_tensor, prior_tensor], dim=0)  # (4,H,W)

        return {
            "image": fused_img,
            "mask": mask_tensor,
            "case_id": item["case_id"],
            "slice_id": item["slice_id"]
        }

# =========================
# 3. 推理与指标
# =========================
def get_gaussian(size, sigma_scale=1.0 / 8, device="cpu"):
    k_h, k_w = size
    center_coords = [k_h // 2, k_w // 2]
    sigmas = [k_h * sigma_scale, k_w * sigma_scale]

    y, x = np.ogrid[:k_h, :k_w]
    h_norm = (y - center_coords[0]) / sigmas[0]
    w_norm = (x - center_coords[1]) / sigmas[1]
    g = np.exp(-(h_norm ** 2 + w_norm ** 2) / 2)

    return torch.from_numpy(g).float().to(device)

@torch.no_grad()
def sliding_window_inference_tta(model, image, window_size=(518, 518), overlap=0.5):
    b, c, h, w = image.shape
    tile_h, tile_w = window_size
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))

    gaussian_weight = get_gaussian(window_size, device=image.device)
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

def calc_metrics(pred_mask, gt_mask, pred_prob, smooth=1e-5):
    pred = pred_mask.float()
    gt = gt_mask.float()

    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)

    gt_flat = gt.view(-1).cpu().numpy()
    prob_flat = pred_prob.view(-1).cpu().numpy()
    if len(np.unique(gt_flat)) == 1:
        auc = np.nan
    else:
        auc = roc_auc_score(gt_flat, prob_flat)

    return dice, recall, precision, auc

# =========================
# 4. 主流程
# =========================
def main():
    print("=" * 80)
    print("RUN8 strict hold-out 40 测试集独立分割评估")
    print("=" * 80)
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"TEST_DIR      = {config.TEST_DIR}")
    print(f"BEST_MODEL    = {BEST_MODEL_PATH}")
    print(f"OUTPUT_DIR    = {OUT_ROOT}")
    print(f"THRESHOLDS    = {THRESHOLDS}")
    print(f"DEVICE        = {DEVICE}")
    print("=" * 80)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 best_model.pth: {BEST_MODEL_PATH}")

    dataset = RUN8TestDataset(config.TEST_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 输出目录
    (OUT_ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    if SAVE_BINARY_MASKS:
        (OUT_ROOT / "pred_masks").mkdir(parents=True, exist_ok=True)
    if SAVE_PROB_MAPS:
        (OUT_ROOT / "prob_maps").mkdir(parents=True, exist_ok=True)

    per_slice_records = {thr: [] for thr in THRESHOLDS}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test40 inference"):
            image = batch["image"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            case_id = batch["case_id"][0]
            slice_id = batch["slice_id"][0]

            with autocast("cuda" if DEVICE == "cuda" else "cpu", enabled=(DEVICE == "cuda")):
                prob_map = sliding_window_inference_tta(
                    model,
                    image,
                    window_size=(config.IMG_SIZE, config.IMG_SIZE),
                    overlap=0.5
                )

            pred_prob = prob_map[:, 1, :, :]  # (1,H,W)

            if SAVE_PROB_MAPS:
                case_dir = OUT_ROOT / "prob_maps" / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                np.save(case_dir / f"{slice_id}_prob.npy", pred_prob[0].cpu().numpy().astype(np.float32))

            gt_binary = (mask == 1).float()
            gt_positive_pixels = int(gt_binary.sum().item())
            gt_empty = int(gt_positive_pixels == 0)

            for thr in THRESHOLDS:
                pred_mask = (pred_prob > thr).float()

                dice, recall, precision, auc = calc_metrics(pred_mask, gt_binary, pred_prob)

                pred_positive_pixels = int(pred_mask.sum().item())
                pred_empty = int(pred_positive_pixels == 0)

                per_slice_records[thr].append({
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "threshold": thr,
                    "dice": dice,
                    "recall": recall,
                    "precision": precision,
                    "auc": auc,
                    "gt_positive_pixels": gt_positive_pixels,
                    "pred_positive_pixels": pred_positive_pixels,
                    "gt_empty": gt_empty,
                    "pred_empty": pred_empty
                })

                if SAVE_BINARY_MASKS:
                    thr_tag = f"thr_{int(round(thr * 100)):03d}"
                    case_dir = OUT_ROOT / "pred_masks" / thr_tag / case_id
                    case_dir.mkdir(parents=True, exist_ok=True)
                    np.save(
                        case_dir / f"{slice_id}_pred.npy",
                        pred_mask[0].cpu().numpy().astype(np.uint8)
                    )

    # =========================
    # 5. 保存 per-slice / per-case / summary
    # =========================
    summary_rows = []

    for thr in THRESHOLDS:
        thr_tag = f"thr_{int(round(thr * 100)):03d}"
        df_slice = pd.DataFrame(per_slice_records[thr])

        # per-slice
        slice_csv = OUT_ROOT / "metrics" / f"per_slice_metrics_{thr_tag}.csv"
        df_slice.to_csv(slice_csv, index=False, encoding="utf-8-sig")

        # per-case：对每个病例按 slice 求平均
        df_case = (
            df_slice.groupby("case_id", as_index=False)
            .agg({
                "dice": "mean",
                "recall": "mean",
                "precision": "mean",
                "auc": "mean",
                "gt_positive_pixels": "sum",
                "pred_positive_pixels": "sum",
                "gt_empty": "sum",
                "pred_empty": "sum"
            })
            .rename(columns={
                "dice": "case_mean_dice",
                "recall": "case_mean_recall",
                "precision": "case_mean_precision",
                "auc": "case_mean_auc",
                "gt_empty": "num_gt_empty_slices",
                "pred_empty": "num_pred_empty_slices"
            })
        )
        case_csv = OUT_ROOT / "metrics" / f"per_case_metrics_{thr_tag}.csv"
        df_case.to_csv(case_csv, index=False, encoding="utf-8-sig")

        # summary
        summary = {
            "threshold": thr,
            "n_cases": int(df_case.shape[0]),
            "n_slices": int(df_slice.shape[0]),

            "slice_mean_dice": float(df_slice["dice"].mean()),
            "slice_mean_recall": float(df_slice["recall"].mean()),
            "slice_mean_precision": float(df_slice["precision"].mean()),
            "slice_mean_auc": float(df_slice["auc"].dropna().mean()) if df_slice["auc"].notna().any() else np.nan,

            "case_mean_dice": float(df_case["case_mean_dice"].mean()),
            "case_mean_recall": float(df_case["case_mean_recall"].mean()),
            "case_mean_precision": float(df_case["case_mean_precision"].mean()),
            "case_mean_auc": float(df_case["case_mean_auc"].dropna().mean()) if df_case["case_mean_auc"].notna().any() else np.nan,

            "slice_pred_empty_rate": float(df_slice["pred_empty"].mean()),
            "slice_gt_empty_rate": float(df_slice["gt_empty"].mean()),

            "num_slices_pred_empty": int(df_slice["pred_empty"].sum()),
            "num_slices_gt_empty": int(df_slice["gt_empty"].sum()),
            "total_pred_positive_pixels": int(df_slice["pred_positive_pixels"].sum()),
            "total_gt_positive_pixels": int(df_slice["gt_positive_pixels"].sum()),
        }
        summary_rows.append(summary)

        # 单独 txt
        txt_path = OUT_ROOT / "metrics" / f"summary_{thr_tag}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Threshold: {thr}\n")
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_ROOT / "metrics" / "summary_by_threshold.csv"
    summary_json = OUT_ROOT / "metrics" / "summary_by_threshold.json"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("\n✅ 独立测试集分割评估完成。")
    print(f"📄 summary csv: {summary_csv}")
    print(f"📄 summary json: {summary_json}")
    print("📄 per-slice / per-case 指标均已写入 metrics 文件夹。")


if __name__ == "__main__":
    main()