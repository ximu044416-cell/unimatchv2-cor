import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(r"F:\Dinov2_data\final_slices_cor")


def main():
    max_h = 0
    max_w = 0

    # 遍历 train, val, unlabel 三个大本营
    for sub_dir in ["train", "val", "unlabel"]:
        folder_path = BASE_DIR / sub_dir
        if not folder_path.exists():
            continue

        print(f"🔍 正在扫描 {sub_dir} 文件夹...")
        # 遍历每个病人
        for case_dir in tqdm(list(folder_path.iterdir())):
            if not case_dir.is_dir():
                continue

            # 因为同一个病人 3D 框出来的所有切片尺寸都一样，我们只读他第一张 data.npy 就行，极速！
            npy_files = list(case_dir.glob("*_data.npy"))
            if npy_files:
                sample_file = npy_files[0]
                arr = np.load(sample_file)
                h, w = arr.shape[0], arr.shape[1]

                if h > max_h: max_h = h
                if w > max_w: max_w = w

    print("\n" + "=" * 50)
    print(f"📊 扫描完毕！全军 204 例数据中：")
    print(f"   - 最大高度 (Max Height): {max_h}")
    print(f"   - 最大宽度 (Max Width): {max_w}")
    print("=" * 50)
    print("👉 请把这两个数字告诉我，我来为你指定最完美的 DINOv2 黄金尺寸档位！")


if __name__ == "__main__":
    main()