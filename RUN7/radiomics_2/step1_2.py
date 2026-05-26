import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from torch.utils.data import DataLoader
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_2 import config
from models.dinov2_unet import DINOUNet
from data.dataset_2 import UniMatchDataset2

# =========================
# 固定 RUN7 最佳权重
# =========================
WEIGHT_PATH = r"/RUN7/logs/UniMatch_Cor_Run7_NegPrior/best_model.pth"

# =========================
# 新输出目录：radiomics_2
# =========================
OUTPUT_ROOT = Path(r"/RUN7/radiomics_2")

# =========================
# 参考 train/test 花名册
# =========================
REFERENCE_DIR = Path(r"F:\downstreamtasks\radiomics")

# =========================
# 扫描阈值
# =========================
THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]


def save_nifti(arr, out_path, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(out_path))


def reconstruct_run7_prob_and_thresholds():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 输出目录
    (OUTPUT_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "test").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "thresholds").mkdir(parents=True, exist_ok=True)

    for thr in THRESHOLDS:
        (OUTPUT_ROOT / "thresholds" / f"thr_{int(thr*100):03d}" / "test").mkdir(parents=True, exist_ok=True)

    print("🚀 启动 RUN7 Step1（深度学习环境）")
    print(f"📥 权重: {WEIGHT_PATH}")
    print(f"📂 输出根目录: {OUTPUT_ROOT}")

    train_ref_path = REFERENCE_DIR / "train"
    test_ref_path = REFERENCE_DIR / "test"

    if not train_ref_path.exists() or not test_ref_path.exists():
        raise FileNotFoundError(f"找不到参考目录: {train_ref_path} 或 {test_ref_path}")

    train_roster = set(os.listdir(train_ref_path))
    test_roster = set(os.listdir(test_ref_path))
    print(f"📋 train roster: {len(train_roster)} | test roster: {len(test_roster)}")

    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device), strict=True)
    model.eval()

    target_dirs = [Path(config.TRAIN_DIR), Path(config.UNLABEL_DIR), Path(config.VAL_DIR)]
    patient_dict = {}

    for split_dir in target_dirs:
        if not split_dir.exists():
            print(f"⚠️ 找不到数据目录: {split_dir}，跳过")
            continue

        for case_dir in split_dir.iterdir():
            if case_dir.is_dir() and not case_dir.name.startswith('.'):
                patient_id = case_dir.name
                if patient_id not in train_roster and patient_id not in test_roster:
                    continue

                slices = sorted([p.absolute() for p in case_dir.glob("*_data.npy")])
                if len(slices) > 0:
                    patient_dict[patient_id] = slices

    print(f"📊 准备重建 {len(patient_dict)} 名在册患者")

    summary_rows = []

    with torch.no_grad():
        for patient_id, slice_list in tqdm(patient_dict.items(), desc="全队列重建进度"):
            split_folder = "train" if patient_id in train_roster else "test"
            patient_output_dir = OUTPUT_ROOT / split_folder / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)

            patient_dataset = UniMatchDataset2(slice_list, mode='val')
            patient_loader = DataLoader(patient_dataset, batch_size=1, shuffle=False, num_workers=0)

            patient_images = []
            patient_probs = []
            patient_gts = []

            for data_tuple, slice_path_str in zip(patient_loader, slice_list):
                img_tensor, _ = data_tuple

                orig_img = np.load(slice_path_str)
                orig_h, orig_w = orig_img.shape[:2]
                patient_images.append(orig_img[..., 0].astype(np.float32))

                tensor_h, tensor_w = img_tensor.shape[-2:]
                img_tensor = img_tensor.to(device)

                resized = False
                if tensor_h != config.IMG_SIZE or tensor_w != config.IMG_SIZE:
                    img_tensor = F.interpolate(
                        img_tensor,
                        size=(config.IMG_SIZE, config.IMG_SIZE),
                        mode='bilinear',
                        align_corners=False
                    )
                    resized = True

                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1:2, :, :]

                if resized:
                    probs = F.interpolate(probs, size=(tensor_h, tensor_w), mode='bilinear', align_corners=False)

                prob_np = probs.cpu().numpy()[0, 0].astype(np.float32)

                # 去 padding
                pad_h = max(0, config.IMG_SIZE - orig_h)
                pad_w = max(0, config.IMG_SIZE - orig_w)
                if pad_h > 0 or pad_w > 0:
                    pad_top = pad_h // 2 if pad_h > 0 else 0
                    pad_left = pad_w // 2 if pad_w > 0 else 0
                    prob_np = prob_np[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]

                label_path = str(slice_path_str).replace('_data.npy', '_label.npy')
                if os.path.exists(label_path):
                    gt_mask_np = np.load(label_path).astype(np.uint8)
                else:
                    gt_mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)

                assert prob_np.shape == gt_mask_np.shape == (orig_h, orig_w), \
                    f"尺寸异常！概率:{prob_np.shape}, GT:{gt_mask_np.shape}"

                patient_probs.append(prob_np)
                patient_gts.append(gt_mask_np)

            vol_img = np.stack(patient_images, axis=0)
            vol_prob = np.stack(patient_probs, axis=0)
            vol_gt = np.stack(patient_gts, axis=0).astype(np.uint8)

            # 保存 train/test 基础体积
            save_nifti(vol_img, patient_output_dir / f"{patient_id}_Image.nii.gz")
            save_nifti(vol_gt, patient_output_dir / f"{patient_id}_GT.nii.gz")

            # test 额外保存概率图 + 多阈值 mask
            if split_folder == "test":
                save_nifti(vol_prob, patient_output_dir / f"{patient_id}_Prob.nii.gz")

                for thr in THRESHOLDS:
                    pred_bin = (vol_prob > thr).astype(np.uint8)
                    thr_dir = OUTPUT_ROOT / "thresholds" / f"thr_{int(thr*100):03d}" / "test" / patient_id
                    thr_dir.mkdir(parents=True, exist_ok=True)
                    save_nifti(pred_bin, thr_dir / f"{patient_id}_Pred.nii.gz")

                    summary_rows.append({
                        "Patient_ID": patient_id,
                        "Threshold": thr,
                        "Pred_Volume": int(pred_bin.sum()),
                        "GT_Volume": int(vol_gt.sum()),
                        "Is_ZeroMask": int(pred_bin.sum() == 0),
                        "Label": int(vol_gt.sum() > 0)
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ROOT / "threshold_volume_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n🎉 Step1 完成，结果已保存到: {OUTPUT_ROOT}")
    print(f"📄 阈值体积统计表: {OUTPUT_ROOT / 'threshold_volume_summary.csv'}")


if __name__ == "__main__":
    reconstruct_run7_prob_and_thresholds()