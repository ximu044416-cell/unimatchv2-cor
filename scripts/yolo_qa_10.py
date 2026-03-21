import os
import cv2
import numpy as np
import SimpleITK as sitk
import random

# ==========================================
# 1. 核心配置区 (请核对您的文件名)
# ==========================================
# 原始对齐好的 NIfTI 数据根目录 (包含 204 个病人)
INPUT_DIR = r"F:\Dinov2_data\ALL_COR_FINAL"

# 包含绝对坐标的 YOLO TXT 文件夹
YOLO_TXT_DIR = r"F:\YOLO\Inference_Annotated_2"

# QA 图片输出的新主目录
OUTPUT_DIR = r"F:\Dinov2_data\ALL_COR_YOLO"

# 你的 NIfTI 序列文件名
T1_NAME = "T1.nii.gz"
WATER_NAME = "Water.nii.gz"  # 或 STIR.nii.gz
FAT_NAME = "Fat.nii.gz"

# 🚨 请确认你文件夹里手工勾画的标签文件名！
LABEL_NAME = "label.nii.gz"

# 测试采样的病人数量
SAMPLE_SIZE = 10

# 外扩边距 (向外扩展包住健康骨髓)
MARGIN = 35


# ==========================================

def normalize_for_vis(slice_array):
    """仅用于可视化：将 NIfTI 切片转为 0-255 灰度图"""
    p1 = np.percentile(slice_array, 1)
    p99 = np.percentile(slice_array, 99)
    if p99 - p1 < 1e-3:
        return np.zeros_like(slice_array, dtype=np.uint8)
    norm = np.clip(slice_array, p1, p99)
    norm = (norm - p1) / (p99 - p1) * 255.0
    return norm.astype(np.uint8)


def get_valid_boxes(patient_txt_dir):
    """读取病人目录下所有非空的 TXT 文件，获取有效坐标"""
    valid_boxes = {}
    if not os.path.exists(patient_txt_dir):
        return valid_boxes

    for txt_file in os.listdir(patient_txt_dir):
        if not txt_file.endswith(".txt"): continue

        try:
            # 提取文件名中的 slice 数字
            slice_idx = int(txt_file.split('_')[-1].split('.')[0])
        except:
            continue

        txt_path = os.path.join(patient_txt_dir, txt_file)
        if os.path.getsize(txt_path) > 0:
            with open(txt_path, 'r') as f:
                content = f.read().strip()
                if content:
                    coords = list(map(int, content.split(',')))
                    valid_boxes[slice_idx] = coords
    return valid_boxes


if __name__ == '__main__':
    print(f"🚀 启动 QA 专属 10 人抽样切割 (仅生成 RGB 和 SEG 图片)...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_patients = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
    sample_patients = random.sample(all_patients, min(SAMPLE_SIZE, len(all_patients)))

    print(f"🎯 抽中的病人 ID: {sample_patients}")

    for patient_id in sample_patients:
        print(f"🔄 正在处理病人: {patient_id} ...")

        # 建立输出子目录
        p_out_dir = os.path.join(OUTPUT_DIR, patient_id)
        rgb_dir = os.path.join(p_out_dir, "rgb")
        seg_dir = os.path.join(p_out_dir, "seg")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

        patient_in_dir = os.path.join(INPUT_DIR, patient_id)
        patient_txt_dir = os.path.join(YOLO_TXT_DIR, patient_id)

        valid_boxes = get_valid_boxes(patient_txt_dir)
        if not valid_boxes:
            print(f"⚠️ 找不到 YOLO 坐标，跳过。")
            continue

        # 读取该病人的所有 3D NIfTI 序列
        try:
            t1_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_in_dir, T1_NAME)))
            water_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_in_dir, WATER_NAME)))
            fat_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_in_dir, FAT_NAME)))
        except Exception as e:
            print(f"⚠️ 序列缺失，跳过: {e}")
            continue

        # 读取标签 (如果没有则全 0)
        label_path = os.path.join(patient_in_dir, LABEL_NAME)
        if os.path.exists(label_path):
            label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        else:
            label_arr = np.zeros_like(t1_arr, dtype=np.uint8)

        num_slices, img_h, img_w = t1_arr.shape

        # 逐层遍历
        for z in range(num_slices):
            # 这只是个 QA 脚本，如果这层没有 YOLO 框，我们就不测这层，直接跳过
            if z not in valid_boxes:
                continue

            # 1. 获取绝对坐标并外扩
            xmin, ymin, xmax, ymax = valid_boxes[z]
            xmin_m = max(0, xmin - MARGIN)
            ymin_m = max(0, ymin - MARGIN)
            xmax_m = min(img_w - 1, xmax + MARGIN)
            ymax_m = min(img_h - 1, ymax + MARGIN)

            # 2. 核心：切下 4 块矩阵！
            c_t1 = t1_arr[z, ymin_m:ymax_m, xmin_m:xmax_m]
            c_water = water_arr[z, ymin_m:ymax_m, xmin_m:xmax_m]
            c_fat = fat_arr[z, ymin_m:ymax_m, xmin_m:xmax_m]
            c_label = label_arr[z, ymin_m:ymax_m, xmin_m:xmax_m]

            base_name = f"{patient_id}_slice_{z:03d}"

            # 3. 准备可视化图像 (转 0-255)
            vis_t1 = normalize_for_vis(c_t1)
            vis_water = normalize_for_vis(c_water)
            vis_fat = normalize_for_vis(c_fat)

            # ==========================================
            # 🔥 生成 RGB 堆叠图 (查错位)
            # ==========================================
            # OpenCV 默认是 BGR 格式，所以传入顺序为 [Fat, Water, T1]
            # 这样 T1 就是红色通道，Water 是绿色，Fat 是蓝色
            rgb_img = cv2.merge([vis_fat, vis_water, vis_t1])
            cv2.imwrite(os.path.join(rgb_dir, f"{base_name}_rgb.jpg"), rgb_img)

            # ==========================================
            # 🔥 生成 SEG 叠加图 (查标签漂移)
            # ==========================================
            # 用切好的 T1 做底图
            seg_img = cv2.cvtColor(vis_t1, cv2.COLOR_GRAY2BGR)

            # 如果这层有标签，就叠加上去
            if np.any(c_label > 0):
                red_mask = np.zeros_like(seg_img)
                red_mask[c_label > 0] = [0, 0, 255]  # 红色
                # 0.5 的透明度叠加
                seg_img = cv2.addWeighted(seg_img, 1.0, red_mask, 0.5, 0)

            cv2.imwrite(os.path.join(seg_dir, f"{base_name}_seg.jpg"), seg_img)

    print("🎉 极速 QA 图片生成完毕！请前往 ALL_COR_YOLO 目录查阅！")