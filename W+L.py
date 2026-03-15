import os
import nibabel as nib
import numpy as np
import cv2


def process_mri_and_labels(base_dir, output_base_dir, slice_axis=2, thickness=1):
    """
    遍历文件夹处理MRI和Label，修复了高亮区域数值溢出导致的等高线伪影
    """
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    folder_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folder_names.sort()

    total_folders = len(folder_names)
    print(f"总共找到 {total_folders} 个文件夹，开始处理...\n" + "=" * 40)

    for index, folder_name in enumerate(folder_names, 1):
        folder_path = os.path.join(base_dir, folder_name)

        water_path = None
        label_path = None
        for ext in ['.nii', '.nii.gz']:
            if os.path.exists(os.path.join(folder_path, f'water{ext}')):
                water_path = os.path.join(folder_path, f'water{ext}')
            if os.path.exists(os.path.join(folder_path, f'label{ext}')):
                label_path = os.path.join(folder_path, f'label{ext}')

        if not water_path or not label_path:
            continue

        print(f"[{index}/{total_folders}] 正在处理: {folder_name} ...")

        try:
            water_nii = nib.load(water_path)
            label_nii = nib.load(label_path)
            water_data = water_nii.get_fdata()
            label_data = label_nii.get_fdata()
        except Exception as e:
            print(f"  [错误] 读取文件失败: {e}")
            continue

        if water_data.shape != label_data.shape:
            continue

        output_folder = os.path.join(output_base_dir, folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        num_slices = water_data.shape[slice_axis]

        # 【核心修复区】：获取整个3D图像的最大和最小值，用于安全的全局亮度映射
        vol_min = water_data.min()
        vol_max = water_data.max()

        for i in range(num_slices):
            if slice_axis == 0:
                img_slice = water_data[i, :, :]
                lbl_slice = label_data[i, :, :]
            elif slice_axis == 1:
                img_slice = water_data[:, i, :]
                lbl_slice = label_data[:, i, :]
            else:
                img_slice = water_data[:, :, i]
                lbl_slice = label_data[:, :, i]

            # 【核心修复区】：安全转换到 0-255，杜绝亮斑变成黑圈的现象
            if vol_max > vol_min:
                # 重新映射到 0-255 的浮点数范围
                img_normalized = (img_slice - vol_min) / (vol_max - vol_min) * 255.0
                # 使用 np.clip 掐断任何超出 0-255 的极值，然后安全转为 uint8
                img_slice_8bit = np.clip(img_normalized, 0, 255).astype(np.uint8)
            else:
                img_slice_8bit = np.zeros_like(img_slice, dtype=np.uint8)

            bgr_img = cv2.cvtColor(img_slice_8bit, cv2.COLOR_GRAY2BGR)

            # 画 Label
            if np.any(lbl_slice > 0):
                binary_mask = (lbl_slice > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(bgr_img, contours, -1, (0, 0, 255), thickness)

            save_name = f"{folder_name}_slice_{i:02d}.png"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, bgr_img)

    print("=" * 40 + "\n处理完成！异常描边已被修复。")


# ================= 运行区 =================
base_directory = r"F:\Dinov2_data\ALL_COR_N4"
output_directory = r"F:\W+L"

process_mri_and_labels(base_directory, output_directory, slice_axis=2, thickness=1)