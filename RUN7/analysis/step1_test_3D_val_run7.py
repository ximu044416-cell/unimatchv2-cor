import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_2 import config
from models.dinov2_unet import DINOUNet
from data.dataset_2 import UniMatchDataset2


MODEL_RUNS = {
    "best_model": r"F:\cor\RUN7\logs\UniMatch_Cor_Run7_NegPrior\best_model.pth",
    "swa_offline": r"F:\cor\RUN7\logs\UniMatch_Cor_Run7_NegPrior\best_model_SWA_offline_380_400_best_420_440.pth",
}

ANALYSIS_ROOT = Path(r"/RUN7/analysis")
OUTPUT_3D_ROOT = ANALYSIS_ROOT / "3D_Predictions"
OUTPUT_3D_ROOT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.70


def reconstruct_for_one_weight(model_tag, weight_path):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    output_dir = OUTPUT_3D_ROOT / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 启动 3D 重建: {model_tag}")
    print(f"📥 加载权重: {weight_path}")
    print(f"📂 输出目录: {output_dir}")

    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    model.eval()

    val_dir_path = Path(config.VAL_DIR)
    patient_dict = {}

    for case_dir in val_dir_path.iterdir():
        if case_dir.is_dir() and not case_dir.name.startswith('.'):
            patient_id = case_dir.name
            slices = sorted([p.absolute() for p in case_dir.glob("*_data.npy")])
            if len(slices) > 0:
                patient_dict[patient_id] = slices

    print(f"📊 检测到 {len(patient_dict)} 名验证集患者。")

    with torch.no_grad():
        for patient_id, slice_list in patient_dict.items():
            print(f"🩺 正在处理患者: [{patient_id}] ({len(slice_list)} slices)")

            patient_dataset = UniMatchDataset2(slice_list, mode='val')
            patient_loader = DataLoader(patient_dataset, batch_size=1, shuffle=False, num_workers=0)

            patient_images = []
            patient_preds = []
            patient_gts = []

            for data_tuple, slice_path_str in tqdm(
                zip(patient_loader, slice_list),
                total=len(slice_list),
                desc=f"   - 推理与坐标还原",
                leave=False
            ):
                img_tensor, _ = data_tuple

                orig_img = np.load(slice_path_str)
                orig_h, orig_w = orig_img.shape[:2]
                patient_images.append(orig_img[..., 0].astype(np.float32))

                tensor_h, tensor_w = img_tensor.shape[-2:]
                img_tensor = img_tensor.to(device)

                if tensor_h != config.IMG_SIZE or tensor_w != config.IMG_SIZE:
                    img_tensor = F.interpolate(
                        img_tensor,
                        size=(config.IMG_SIZE, config.IMG_SIZE),
                        mode='bilinear',
                        align_corners=False
                    )

                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1:2, :, :]
                pred_mask = (probs > THRESHOLD).float()

                if tensor_h != config.IMG_SIZE or tensor_w != config.IMG_SIZE:
                    pred_mask = F.interpolate(pred_mask, size=(tensor_h, tensor_w), mode='nearest')

                pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)[0, 0]

                pad_h = max(0, config.IMG_SIZE - orig_h)
                pad_w = max(0, config.IMG_SIZE - orig_w)

                if pad_h > 0 or pad_w > 0:
                    pad_top = pad_h // 2 if pad_h > 0 else 0
                    pad_left = pad_w // 2 if pad_w > 0 else 0
                    pred_mask_np = pred_mask_np[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]

                label_path = str(slice_path_str).replace('_data.npy', '_label.npy')
                if os.path.exists(label_path):
                    gt_mask_np = np.load(label_path).astype(np.uint8)
                else:
                    gt_mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)

                assert pred_mask_np.shape == gt_mask_np.shape == (orig_h, orig_w), \
                    f"尺寸异常！预测:{pred_mask_np.shape}, GT:{gt_mask_np.shape}"

                patient_preds.append(pred_mask_np)
                patient_gts.append(gt_mask_np)

            vol_img = np.stack(patient_images, axis=0)
            vol_pred = np.stack(patient_preds, axis=0)
            vol_gt = np.stack(patient_gts, axis=0)

            sitk_img = sitk.GetImageFromArray(vol_img)
            sitk_pred = sitk.GetImageFromArray(vol_pred)
            sitk_gt = sitk.GetImageFromArray(vol_gt)

            sitk_img.SetSpacing((1.0, 1.0, 1.0))
            sitk_pred.SetSpacing((1.0, 1.0, 1.0))
            sitk_gt.SetSpacing((1.0, 1.0, 1.0))

            img_file = output_dir / f"{patient_id}_Image.nii.gz"
            pred_file = output_dir / f"{patient_id}_Pred.nii.gz"
            gt_file = output_dir / f"{patient_id}_GT.nii.gz"

            sitk.WriteImage(sitk_img, str(img_file))
            sitk.WriteImage(sitk_pred, str(pred_file))
            sitk.WriteImage(sitk_gt, str(gt_file))

    print(f"✅ {model_tag} 的 3D 重建完成。输出: {output_dir}")


def main():
    for model_tag, weight_path in MODEL_RUNS.items():
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过 {model_tag}，因为找不到权重: {weight_path}")
            continue
        reconstruct_for_one_weight(model_tag, weight_path)

    print("\n🎉 所有模型的 3D 重建已完成。")


if __name__ == "__main__":
    main()