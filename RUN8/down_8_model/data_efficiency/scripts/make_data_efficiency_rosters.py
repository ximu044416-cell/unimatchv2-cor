import random
from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# 路径
# =========================================================
RUN8_ROOT = Path(r"F:\cor\RUN8")
DATA_ROOT = RUN8_ROOT / "data" / "final_slices_YOLO_cor"
TRAIN_DIR = DATA_ROOT / "train"

OUT_DIR = RUN8_ROOT / "down_8_model" / "data_efficiency" / "rosters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def case_has_foreground(case_dir: Path) -> bool:
    """
    判断一个病例是否有前景标签。
    如果 label 文件不存在或读取失败，则保守视为 False。
    """
    label_files = sorted(case_dir.glob("*_label.npy"))
    if not label_files:
        return False

    for lp in label_files:
        try:
            arr = np.load(lp)
            if np.any(arr > 0):
                return True
        except Exception:
            continue

    return False


def write_roster(names, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(str(name).strip() + "\n")


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"❌ 找不到 train 目录: {TRAIN_DIR}")

    case_dirs = sorted([p for p in TRAIN_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")])

    if len(case_dirs) == 0:
        raise RuntimeError(f"❌ train 目录下没有病例文件夹: {TRAIN_DIR}")

    print("=" * 80)
    print("🚀 生成 data efficiency 固定病例名单")
    print(f"TRAIN_DIR = {TRAIN_DIR}")
    print(f"病例数 = {len(case_dirs)}")
    print("=" * 80)

    records = []
    for cd in case_dirs:
        fg = case_has_foreground(cd)
        n_slices = len(list(cd.glob("*_data.npy")))
        records.append({
            "Patient_ID": cd.name,
            "Has_Foreground": int(fg),
            "Num_Slices": n_slices
        })

    df = pd.DataFrame(records)

    pos_cases = df[df["Has_Foreground"] == 1]["Patient_ID"].tolist()
    neg_cases = df[df["Has_Foreground"] == 0]["Patient_ID"].tolist()

    rng = random.Random(SEED)
    rng.shuffle(pos_cases)
    rng.shuffle(neg_cases)

    total_n = len(df)
    pos_ratio = len(pos_cases) / total_n if total_n > 0 else 0

    print(f"前景病例数: {len(pos_cases)}")
    print(f"非前景/空标签病例数: {len(neg_cases)}")
    print(f"前景比例: {pos_ratio:.3f}")

    rosters = {}

    for n in [11, 22, 55]:
        if n > total_n:
            raise ValueError(f"❌ 请求 {n} 个病例，但 train 只有 {total_n} 个病例")

        if len(pos_cases) > 0 and len(neg_cases) > 0:
            n_pos = int(round(n * pos_ratio))
            n_pos = max(1, min(n_pos, len(pos_cases)))
            n_neg = n - n_pos

            if n_neg > len(neg_cases):
                n_neg = len(neg_cases)
                n_pos = n - n_neg

            selected = pos_cases[:n_pos] + neg_cases[:n_neg]
        else:
            all_cases = pos_cases + neg_cases
            selected = all_cases[:n]

        selected = sorted(selected)
        rosters[n] = selected

        out_txt = OUT_DIR / f"train_{n}_roster.txt"
        write_roster(selected, out_txt)

        print(f"✅ train_{n}_roster.txt | n={len(selected)} | {out_txt}")

    # 55 例确保就是全部 train 病例
    all_train_names = sorted(df["Patient_ID"].tolist())
    write_roster(all_train_names, OUT_DIR / "train_55_roster.txt")
    rosters[55] = all_train_names

    # 保存总表
    df["In_11"] = df["Patient_ID"].isin(rosters[11]).astype(int)
    df["In_22"] = df["Patient_ID"].isin(rosters[22]).astype(int)
    df["In_55"] = df["Patient_ID"].isin(rosters[55]).astype(int)
    df.to_csv(OUT_DIR / "data_efficiency_roster_summary.csv", index=False, encoding="utf-8-sig")

    # 检查嵌套关系
    set11 = set(rosters[11])
    set22 = set(rosters[22])
    set55 = set(rosters[55])

    assert set11.issubset(set22), "❌ train_11 不是 train_22 的子集"
    assert set22.issubset(set55), "❌ train_22 不是 train_55 的子集"

    print("\n🎉 roster 生成完成")
    print(f"📂 输出目录: {OUT_DIR}")
    print("✅ 已保证 11 ⊂ 22 ⊂ 55")


if __name__ == "__main__":
    main()