import os
import torch
import numpy as np
import cv2
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor


class MedSAM2PriorGenerator3D:
    def __init__(self, checkpoint_path, model_cfg="sam2.1_hiera_b+.yaml", device="cuda"):
        self.device = device
        print(f"📥 正在加载 MedSAM-2 3D 引擎: {checkpoint_path}")

        torch.cuda.empty_cache()
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=self.device)
        print("✅ 3D 引擎加载完毕！准备开启流式记忆。")

    def extract_slice_info(self, filename):
        """
        🔥 排雷 1 & 3：兼容性极强的正则提取
        同时提取 Patient ID 和 Slice Number，应对各种扁平命名法
        假设文件名格式如：patient001_slice_015_data.npy 或 case_02_slice3_data.npy
        """
        # 匹配 "slice" 加上可选的下划线，再接数字
        match = re.search(r'(.*?)_?slice_?(\d+)', filename)
        if match:
            patient_id = match.group(1)
            slice_num = int(match.group(2))
            return patient_id, slice_num
        return "unknown_patient", 0

    def get_auto_bbox(self, img_uint8):
        """自动寻找 518x518 里的真实 MRI 边界，规避冗余的外部 TXT 坐标映射"""
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return np.array([x, y, x + w, y + h], dtype=np.float32)
        else:
            margin = 40
            return np.array([margin, margin, 518 - margin, 518 - margin], dtype=np.float32)

    @torch.inference_mode()
    def process_dataset(self, data_dir):
        data_path = Path(data_dir)

        # 获取所有 npy 文件
        all_files = list(data_path.rglob("*_data.npy"))
        if not all_files:
            print(f"⚠️ {data_dir} 中没有找到 *_data.npy 文件。")
            return

        # 🔥 排雷 1：动态内存分组，解决扁平文件夹问题
        patient_groups = defaultdict(list)
        for f in all_files:
            pid, snum = self.extract_slice_info(f.name)
            patient_groups[pid].append((snum, f))

        print(f"📊 在 {data_path.name} 中检测到 {len(patient_groups)} 个独立患者序列。")

        # 按患者级别进行 3D 流式处理
        for pid, slice_tuples in tqdm(patient_groups.items(), desc=f"Processing Patients"):
            # 严格按 Z 轴物理序号排序
            slice_tuples.sort(key=lambda x: x[0])
            slice_files = [t[1] for t in slice_tuples]

            with tempfile.TemporaryDirectory() as temp_video_dir:
                frames_uint8 = []

                # 1. 准备当前患者的视频帧
                for idx, f in enumerate(slice_files):
                    img_3c = np.load(f)
                    water_channel = img_3c[..., 0]
                    img_gray_rgb = np.stack([water_channel] * 3, axis=-1)
                    img_uint8 = (img_gray_rgb * 255.0).clip(0, 255).astype(np.uint8)

                    frames_uint8.append(img_uint8)
                    cv2.imwrite(os.path.join(temp_video_dir, f"{idx:05d}.jpg"),
                                cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

                # 2. 启动单患者的 3D 推理
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    inference_state = self.predictor.init_state(video_path=temp_video_dir)

                    middle_frame_idx = len(frames_uint8) // 2
                    middle_img = frames_uint8[middle_frame_idx]
                    best_box = self.get_auto_bbox(middle_img)

                    # 注入先验框
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=middle_frame_idx,
                        obj_id=1,
                        box=best_box
                    )

                    # 时空传播与软先验榨取
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                            inference_state):
                        logits_tensor = out_mask_logits[0]
                        soft_prob_map = torch.sigmoid(logits_tensor).squeeze().cpu().numpy().astype(np.float32)

                        original_file = slice_files[out_frame_idx]
                        save_path = original_file.parent / original_file.name.replace("_data.npy", "_medsam2_prior.npy")
                        np.save(save_path, soft_prob_map)

                    # 🔥 排雷 2：强制重置状态并熔断清理显存，彻底封杀 OOM 黑洞
                    self.predictor.reset_state(inference_state)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 配置区
    MEDSAM2_CKPT = r"F:\cor\checkpoints\MedSAM2_latest.pt"
    MODEL_CFG = "configs/sam2.1_hiera_t512.yaml"

    DATA_DIRS = [
        r"F:\Dinov2_data\final_slices_YOLO_cor\train",
        r"F:\Dinov2_data\final_slices_YOLO_cor\val",
        r"F:\Dinov2_data\final_slices_YOLO_cor\unlabel"
    ]

    generator = MedSAM2PriorGenerator3D(checkpoint_path=MEDSAM2_CKPT, model_cfg=MODEL_CFG)

    for folder in DATA_DIRS:
        generator.process_dataset(folder)

    print("\n🎉 全部排雷完毕！3D 软先验已生成，随时可以向 DINOv2 投喂数据！")