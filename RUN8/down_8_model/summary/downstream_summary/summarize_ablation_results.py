import re
from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 基础路径
# =========================================================
ROOT = Path(r"F:\cor\RUN8\down_8_model")
OUT_DIR = ROOT / "summary" / "segmentation_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 主模型结果：你最终模型真实验证 Dice
# =========================================================
FINAL_MODEL_ROW = {
    "Group": "Module ablation",
    "Experiment": "Final model",
    "Setting": "DINOv2 + prior input + negative-prior penalty",
    "Mode": "semi",
    "N_labeled": 55,
    "Best_Dice": 0.7386,
    "Best_Threshold": "",
    "Best_Epoch": "",
    "Recall": "",
    "Precision": "",
    "AUC": "",
    "Source": "Final main model"
}


# =========================================================
# 日志解析：适用于 resnet50 / no_prior / prior_no_penalty
# =========================================================
def parse_train_log(log_path: Path):
    """
    从 train_log.txt 中解析所有 Epoch 行，并返回 Dice 最大的一行。
    支持格式：
    Epoch 010 | ... Best(Th=0.65) Dice 0.6625, Rec 0.7855, Prec 0.7337, AUC 0.9648
    """
    if not log_path.exists():
        print(f"⚠️ 找不到日志文件: {log_path}")
        return None

    text = log_path.read_text(encoding="utf-8", errors="ignore")

    pattern = re.compile(
        r"Epoch\s+(\d+).*?"
        r"Best\(Th=([0-9.]+)\)\s+Dice\s+([0-9.]+),\s+"
        r"Rec\s+([0-9.]+),\s+"
        r"Prec\s+([0-9.]+),\s+"
        r"AUC\s+([0-9.]+)",
        re.IGNORECASE
    )

    rows = []
    for m in pattern.finditer(text):
        rows.append({
            "Best_Epoch": int(m.group(1)),
            "Best_Threshold": float(m.group(2)),
            "Best_Dice": float(m.group(3)),
            "Recall": float(m.group(4)),
            "Precision": float(m.group(5)),
            "AUC": float(m.group(6)),
        })

    if not rows:
        print(f"⚠️ 未能从日志解析 Epoch 结果: {log_path}")
        return None

    df = pd.DataFrame(rows)
    best_row = df.loc[df["Best_Dice"].idxmax()].to_dict()
    return best_row


# =========================================================
# 读取 data_efficiency 的 best_epoch_summary.csv
# 如果没有，就从 epoch_metrics.csv 中取 val_dice 最大行
# =========================================================
def parse_data_efficiency_folder(folder: Path):
    best_csv = folder / "best_epoch_summary.csv"
    epoch_csv = folder / "epoch_metrics.csv"

    if best_csv.exists():
        df = pd.read_csv(best_csv)
        if len(df) > 0:
            row = df.iloc[0].to_dict()
            return {
                "Best_Epoch": int(row.get("epoch", "")) if not pd.isna(row.get("epoch", np.nan)) else "",
                "Best_Threshold": row.get("best_threshold", ""),
                "Best_Dice": float(row.get("val_dice", np.nan)),
                "Recall": float(row.get("val_recall", np.nan)),
                "Precision": float(row.get("val_precision", np.nan)),
                "AUC": float(row.get("val_auc", np.nan)),
            }

    if epoch_csv.exists():
        df = pd.read_csv(epoch_csv)
        if len(df) == 0:
            return None

        idx = df["val_dice"].idxmax()
        row = df.loc[idx].to_dict()

        return {
            "Best_Epoch": int(row.get("epoch", "")) if not pd.isna(row.get("epoch", np.nan)) else "",
            "Best_Threshold": row.get("best_threshold", ""),
            "Best_Dice": float(row.get("val_dice", np.nan)),
            "Recall": float(row.get("val_recall", np.nan)),
            "Precision": float(row.get("val_precision", np.nan)),
            "AUC": float(row.get("val_auc", np.nan)),
        }

    print(f"⚠️ 找不到 best_epoch_summary.csv 或 epoch_metrics.csv: {folder}")
    return None


# =========================================================
# 模块消融
# =========================================================
def collect_module_ablation():
    experiments = [
        {
            "Experiment": "ResNet50 baseline",
            "Setting": "ResNet50-UNet + prior input + negative-prior penalty",
            "Folder": ROOT / "resnet50_baseline",
            "Log": ROOT / "resnet50_baseline" / "logs" / "train_log.txt",
        },
        {
            "Experiment": "Prior without penalty",
            "Setting": "DINOv2 + prior input, without negative-prior penalty",
            "Folder": ROOT / "prior_no_penalty",
            "Log": ROOT / "prior_no_penalty" / "logs" / "train_log.txt",
        },
        {
            "Experiment": "No prior",
            "Setting": "DINOv2 using only 3 MRI channels, without prior input",
            "Folder": ROOT / "no_prior",
            "Log": ROOT / "no_prior" / "logs" / "train_log.txt",
        },
    ]

    rows = [FINAL_MODEL_ROW.copy()]

    for exp in experiments:
        parsed = parse_train_log(exp["Log"])
        if parsed is None:
            rows.append({
                "Group": "Module ablation",
                "Experiment": exp["Experiment"],
                "Setting": exp["Setting"],
                "Mode": "",
                "N_labeled": "",
                "Best_Dice": np.nan,
                "Best_Threshold": "",
                "Best_Epoch": "",
                "Recall": "",
                "Precision": "",
                "AUC": "",
                "Source": str(exp["Log"]),
            })
            continue

        rows.append({
            "Group": "Module ablation",
            "Experiment": exp["Experiment"],
            "Setting": exp["Setting"],
            "Mode": "semi",
            "N_labeled": 55,
            "Best_Dice": parsed["Best_Dice"],
            "Best_Threshold": parsed["Best_Threshold"],
            "Best_Epoch": parsed["Best_Epoch"],
            "Recall": parsed["Recall"],
            "Precision": parsed["Precision"],
            "AUC": parsed["AUC"],
            "Source": str(exp["Log"]),
        })

    return pd.DataFrame(rows)


# =========================================================
# 数据敏感性
# =========================================================
def collect_data_efficiency():
    rows = []

    # 这 5 组是你实际重新跑的
    experiments = [
        ("Semi-11", "semi", 11, ROOT / "data_efficiency" / "semi_11"),
        ("Sup-11", "sup", 11, ROOT / "data_efficiency" / "sup_11"),
        ("Semi-22", "semi", 22, ROOT / "data_efficiency" / "semi_22"),
        ("Sup-22", "sup", 22, ROOT / "data_efficiency" / "sup_22"),
        ("Sup-55", "sup", 55, ROOT / "data_efficiency" / "sup_55"),
    ]

    for name, mode, n_labeled, folder in experiments:
        parsed = parse_data_efficiency_folder(folder)
        if parsed is None:
            rows.append({
                "Group": "Data efficiency",
                "Experiment": name,
                "Setting": f"{mode}, {n_labeled} labeled patients",
                "Mode": mode,
                "N_labeled": n_labeled,
                "Best_Dice": np.nan,
                "Best_Threshold": "",
                "Best_Epoch": "",
                "Recall": "",
                "Precision": "",
                "AUC": "",
                "Source": str(folder),
            })
            continue

        rows.append({
            "Group": "Data efficiency",
            "Experiment": name,
            "Setting": f"{mode}, {n_labeled} labeled patients",
            "Mode": mode,
            "N_labeled": n_labeled,
            "Best_Dice": parsed["Best_Dice"],
            "Best_Threshold": parsed["Best_Threshold"],
            "Best_Epoch": parsed["Best_Epoch"],
            "Recall": parsed["Recall"],
            "Precision": parsed["Precision"],
            "AUC": parsed["AUC"],
            "Source": str(folder),
        })

    # Semi-55 直接使用最终主模型结果
    rows.append({
        "Group": "Data efficiency",
        "Experiment": "Semi-55 / Final model",
        "Setting": "semi, 55 labeled patients; final model",
        "Mode": "semi",
        "N_labeled": 55,
        "Best_Dice": 0.7386,
        "Best_Threshold": "",
        "Best_Epoch": "",
        "Recall": "",
        "Precision": "",
        "AUC": "",
        "Source": "Final main model",
    })

    return pd.DataFrame(rows)


# =========================================================
# 排序与保存
# =========================================================
def main():
    print("=" * 80)
    print("🚀 汇总消融实验结果")
    print(f"ROOT    = {ROOT}")
    print(f"OUT_DIR = {OUT_DIR}")
    print("=" * 80)

    df_module = collect_module_ablation()
    df_data = collect_data_efficiency()

    # 排序
    module_order = {
        "No prior": 1,
        "Prior without penalty": 2,
        "ResNet50 baseline": 3,
        "Final model": 4,
    }

    data_order = {
        ("sup", 11): 1,
        ("semi", 11): 2,
        ("sup", 22): 3,
        ("semi", 22): 4,
        ("sup", 55): 5,
        ("semi", 55): 6,
    }

    df_module["SortKey"] = df_module["Experiment"].map(module_order).fillna(99)

    df_data["SortKey"] = df_data.apply(
        lambda r: data_order.get((r["Mode"], int(r["N_labeled"])), 99),
        axis=1
    )

    df_module = df_module.sort_values("SortKey").drop(columns=["SortKey"])
    df_data = df_data.sort_values("SortKey").drop(columns=["SortKey"])

    df_all = pd.concat([df_module, df_data], ignore_index=True)

    # 保留 4 位小数
    for col in ["Best_Dice", "Recall", "Precision", "AUC"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
        df_module[col] = pd.to_numeric(df_module[col], errors="coerce")
        df_data[col] = pd.to_numeric(df_data[col], errors="coerce")

    # 保存
    module_path = OUT_DIR / "Module_Ablation_Summary.csv"
    data_path = OUT_DIR / "Data_Efficiency_Summary.csv"
    all_path = OUT_DIR / "Ablation_Final_Summary.csv"

    df_module.to_csv(module_path, index=False, encoding="utf-8-sig")
    df_data.to_csv(data_path, index=False, encoding="utf-8-sig")
    df_all.to_csv(all_path, index=False, encoding="utf-8-sig")

    # 同时输出一个 Excel，方便你直接看
    xlsx_path = OUT_DIR / "Ablation_Final_Summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_module.to_excel(writer, sheet_name="Module_Ablation", index=False)
        df_data.to_excel(writer, sheet_name="Data_Efficiency", index=False)
        df_all.to_excel(writer, sheet_name="All", index=False)

    print("\n✅ 汇总完成")
    print(f"📄 Module ablation: {module_path}")
    print(f"📄 Data efficiency : {data_path}")
    print(f"📄 All summary     : {all_path}")
    print(f"📘 Excel summary   : {xlsx_path}")

    print("\n📌 模块消融结果预览:")
    print(df_module[["Experiment", "Best_Dice", "Best_Threshold", "Best_Epoch", "Recall", "Precision", "AUC"]])

    print("\n📌 数据敏感性结果预览:")
    print(df_data[["Experiment", "Best_Dice", "Best_Threshold", "Best_Epoch", "Recall", "Precision", "AUC"]])


if __name__ == "__main__":
    main()