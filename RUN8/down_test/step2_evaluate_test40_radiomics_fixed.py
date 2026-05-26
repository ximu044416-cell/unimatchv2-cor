import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.stats as st
import statsmodels.api as sm

from radiomics import featureextractor
from scipy import stats
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

# =========================================================
# 路径
# =========================================================
BASE_ROOT = Path(r"F:\cor\RUN8\down_8_NEW")

# test40 最终输出
TEST_ROOT = BASE_ROOT / "down_test"
RECON_DIR = TEST_ROOT / "reconstructed_test40"
FEATURE_DIR = TEST_ROOT / "features_test40"
RESULT_DIR = TEST_ROOT / "final_eval_test40"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# dev164 已冻结模型工件
DEV_ROOT = BASE_ROOT / "down_2_threshold"
DEV_CLINICAL_FILE = DEV_ROOT / "clinical" / "clinical_info_dev164.xlsx"
DEV_GT_FEATURES_FILE = DEV_ROOT / "features_dev164" / "GT_Features_Dev164.csv"
DEV_MODEL_DIR = DEV_ROOT / "models_dev164"
YAML_PATH = DEV_ROOT / "config" / "radiomics_features.yaml"

# test40 临床表（你已经整理好的）
TEST_CLINICAL_FILE = DEV_ROOT / "clinical" / "clinical_info_test.xlsx"

# 主分析 + 敏感性分析
THRESHOLD_TAGS = ["080", "075"]


# =========================================================
# DeLong
# =========================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, label_0_count):
    m = label_1_count
    n = label_0_count
    b = predictions_sorted_transposed
    tx = np.empty([np.shape(b)[0], m], dtype=float)
    ty = np.empty([np.shape(b)[0], n], dtype=float)
    tz = np.empty([np.shape(b)[0], m + n], dtype=float)
    for r in range(np.shape(b)[0]):
        tz[r, :] = compute_midrank(b[r, :])
        tx[r, :] = compute_midrank(b[r, :m])
        ty[r, :] = compute_midrank(b[r, m:])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 2 * (1 - st.norm.cdf(np.abs(z)))[0][0]


def delong_roc_test(y_true, y_prob1, y_prob2):
    y_true = np.asarray(y_true)
    y_prob1 = np.asarray(y_prob1)
    y_prob2 = np.asarray(y_prob2)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    preds = np.array([y_prob1, y_prob2])
    preds_sorted = np.hstack([preds[:, pos_idx], preds[:, neg_idx]])
    aucs, sigma = fastDeLong(preds_sorted, len(pos_idx), len(neg_idx))
    p_value = calc_pvalue(aucs, sigma)
    return p_value


# =========================================================
# 工具函数
# =========================================================
def apply_pure_zscore(sitk_image):
    img_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    valid_pixels = img_array[img_array > 0]

    if len(valid_pixels) == 0:
        return sitk_image

    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)
    if std_val < 1e-5:
        std_val = 1e-5

    img_normalized = (img_array - mean_val) / std_val
    img_normalized[img_array <= 0] = img_normalized.min() - 1.0

    new_sitk_image = sitk.GetImageFromArray(img_normalized)
    new_sitk_image.CopyInformation(sitk_image)
    return new_sitk_image


def calc_metrics(y_t, prob, locked_cutoff):
    pred_label = (prob >= locked_cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_t, pred_label, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_t, pred_label)
    auc_val = roc_auc_score(y_t, prob)
    return auc_val, acc, sens, spec, pred_label


def collect_test40_ids():
    patient_dirs = sorted([p for p in RECON_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if len(patient_dirs) == 0:
        raise RuntimeError(f"❌ reconstructed_test40 为空：{RECON_DIR}")
    return [p.name for p in patient_dirs]


def load_and_prepare_clinical(clinical_path):
    if not clinical_path.exists():
        raise FileNotFoundError(f"❌ 找不到临床表：{clinical_path}")

    df = pd.read_excel(clinical_path)
    if "label" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"label": "Label"})

    if "Patient_ID" not in df.columns:
        raise KeyError(f"❌ {clinical_path.name} 缺少 Patient_ID 列")

    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    for col in ["CRP", "ESR", "HLA-B27", "Disease_Duration_Category", "sparcc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def extract_features_for_mask(mask_tag, feature_cols_all, out_csv):
    extractor = featureextractor.RadiomicsFeatureExtractor(str(YAML_PATH))
    patient_dirs = sorted([p for p in RECON_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")])

    rows = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        img_path = patient_dir / f"{patient_id}_Image.nii.gz"

        if mask_tag == "GT":
            mask_path = patient_dir / f"{patient_id}_GT.nii.gz"
        else:
            mask_path = patient_dir / f"{patient_id}_Pred_{mask_tag}.nii.gz"

        row = {
            "Patient_ID": patient_id,
            "ExtractionFailed": 0,
            "MaskEmpty": 0
        }

        # 先全部补 0，保证列齐
        for feat in feature_cols_all:
            row[feat] = 0.0

        try:
            if not img_path.exists() or not mask_path.exists():
                row["ExtractionFailed"] = 1
                rows.append(row)
                continue

            sitk_img = sitk.ReadImage(str(img_path))
            sitk_mask = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(sitk_mask)

            if np.sum(mask_array) == 0:
                row["MaskEmpty"] = 1
                rows.append(row)
                continue

            sitk_img_norm = apply_pure_zscore(sitk_img)
            features = extractor.execute(sitk_img_norm, sitk_mask)

            for feat in feature_cols_all:
                if feat in features:
                    row[feat] = features[feat]

            rows.append(row)

        except Exception:
            row["ExtractionFailed"] = 1
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def compute_radscore(df_features, feature_cols_all, scaler, selected_features, lasso_weights_dict):
    intercept = float(lasso_weights_dict["intercept"])
    coef_dict = lasso_weights_dict["coefs"]

    X = df_features[feature_cols_all].copy().fillna(0.0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols_all)

    radscore = np.full(len(df_features), intercept, dtype=float)
    for feat in selected_features:
        radscore += X_scaled[feat].values * float(coef_dict[feat])

    return radscore, X_scaled


def build_dev_gt_reference():
    # -----------------------------
    # 读工件
    # -----------------------------
    if not DEV_GT_FEATURES_FILE.exists():
        raise FileNotFoundError(f"❌ 找不到 {DEV_GT_FEATURES_FILE}")
    if not DEV_CLINICAL_FILE.exists():
        raise FileNotFoundError(f"❌ 找不到 {DEV_CLINICAL_FILE}")

    imputer_dict = joblib.load(DEV_MODEL_DIR / "imputer_dict.pkl")
    scaler = joblib.load(DEV_MODEL_DIR / "train_scaler.pkl")
    selected_features = joblib.load(DEV_MODEL_DIR / "selected_features.pkl")
    lasso_weights_dict = joblib.load(DEV_MODEL_DIR / "lasso_weights_dict.pkl")
    optimal_cutoff = joblib.load(DEV_MODEL_DIR / "optimal_cutoff.pkl")["optimal_cutoff"]

    # -----------------------------
    # 读 dev GT
    # -----------------------------
    df_dev_gt = pd.read_csv(DEV_GT_FEATURES_FILE)
    df_dev_gt["Patient_ID"] = df_dev_gt["Patient_ID"].astype(str).str.strip()

    df_dev_clin = load_and_prepare_clinical(DEV_CLINICAL_FILE)
    for col, val in imputer_dict.items():
        if col in df_dev_clin.columns:
            df_dev_clin[col] = df_dev_clin[col].fillna(val)

    feature_cols_all = [c for c in df_dev_gt.columns if c != "Patient_ID"]

    df_dev = pd.merge(df_dev_gt, df_dev_clin, on="Patient_ID", how="inner")
    if len(df_dev) == 0:
        raise RuntimeError("❌ dev GT merge 后为空")

    radscore_dev, _ = compute_radscore(
        df_dev,
        feature_cols_all,
        scaler,
        selected_features,
        lasso_weights_dict
    )
    df_dev["Rad_score"] = radscore_dev

    # dev GT 上的健康组最小 Rad-score，供 rescue
    healthy_min_radscore = df_dev.loc[df_dev["Label"] == 0, "Rad_score"].min()
    if pd.isna(healthy_min_radscore):
        healthy_min_radscore = df_dev["Rad_score"].min()

    clinical_vars = ["ESR", "Disease_Duration_Category", "Rad_score"]
    missing_vars = [v for v in clinical_vars if v not in df_dev.columns]
    if len(missing_vars) > 0:
        raise KeyError(f"❌ dev164 临床表缺少列：{missing_vars}")

    X_clinical = sm.add_constant(df_dev[clinical_vars], has_constant="add")[["const"] + clinical_vars]
    y_dev = df_dev["Label"].values

    logit_result = sm.Logit(y_dev, X_clinical).fit(disp=False)

    return {
        "imputer_dict": imputer_dict,
        "scaler": scaler,
        "selected_features": selected_features,
        "lasso_weights_dict": lasso_weights_dict,
        "optimal_cutoff": optimal_cutoff,
        "healthy_min_radscore": healthy_min_radscore,
        "feature_cols_all": feature_cols_all,
        "logit_result": logit_result
    }


def evaluate_one_path(path_name, df_feat, df_test_clin, artifacts):
    feature_cols_all = artifacts["feature_cols_all"]
    scaler = artifacts["scaler"]
    selected_features = artifacts["selected_features"]
    lasso_weights_dict = artifacts["lasso_weights_dict"]
    healthy_min_radscore = artifacts["healthy_min_radscore"]
    optimal_cutoff = artifacts["optimal_cutoff"]
    logit_result = artifacts["logit_result"]

    # 对齐到 test 临床表全病例
    df_eval = pd.merge(
        df_test_clin[["Patient_ID", "Label", "ESR", "Disease_Duration_Category"] + ([c for c in ["sparcc"] if c in df_test_clin.columns])],
        df_feat,
        on="Patient_ID",
        how="left"
    )

    # 特征未提取到的情况
    if "ExtractionFailed" not in df_eval.columns:
        df_eval["ExtractionFailed"] = 1
    if "MaskEmpty" not in df_eval.columns:
        df_eval["MaskEmpty"] = 1

    for feat in feature_cols_all:
        if feat not in df_eval.columns:
            df_eval[feat] = 0.0

    failed_idx = df_eval["ExtractionFailed"].fillna(1).astype(int).astype(bool)

    df_eval[feature_cols_all] = df_eval[feature_cols_all].fillna(0.0)
    radscore, _ = compute_radscore(
        df_eval,
        feature_cols_all,
        scaler,
        selected_features,
        lasso_weights_dict
    )
    df_eval["Rad_score"] = radscore

    all_zero_idx = (df_eval[feature_cols_all].abs().sum(axis=1) == 0.0)
    rescue_idx = failed_idx | all_zero_idx
    df_eval["Rescued"] = rescue_idx.astype(int)
    df_eval.loc[rescue_idx, "Rad_score"] = healthy_min_radscore

    clinical_vars = ["ESR", "Disease_Duration_Category", "Rad_score"]
    X_eval = sm.add_constant(df_eval[clinical_vars], has_constant="add")[["const"] + clinical_vars]
    prob = logit_result.predict(X_eval)

    y_test = df_eval["Label"].values
    auc_val, acc, sens, spec, pred_label = calc_metrics(y_test, prob, optimal_cutoff)

    df_eval["Pred_Prob"] = prob
    df_eval["Pred_Label"] = pred_label
    df_eval["Failed_Extraction"] = failed_idx.astype(int)
    df_eval["All_Zero_Features"] = all_zero_idx.astype(int)

    # sparcc 相关性（如果有）
    if "sparcc" in df_eval.columns and df_eval["sparcc"].notna().sum() >= 3:
        corr_res = stats.spearmanr(df_eval["Rad_score"], df_eval["sparcc"], nan_policy="omit")
        sp_r = float(corr_res.statistic) if corr_res.statistic is not None else np.nan
        sp_p = float(corr_res.pvalue) if corr_res.pvalue is not None else np.nan
    else:
        sp_r, sp_p = np.nan, np.nan

    return {
        "table": df_eval,
        "prob": np.asarray(prob),
        "summary": {
            "Path": path_name,
            "AUC": auc_val,
            "ACC": acc,
            "SEN": sens,
            "SPE": spec,
            "Rescue_N": int(rescue_idx.sum()),
            "Failed_Extraction_N": int(failed_idx.sum()),
            "All_Zero_Features_N": int(all_zero_idx.sum()),
            "Spearman_RadScore_vs_SPARCC_r": sp_r,
            "Spearman_RadScore_vs_SPARCC_p": sp_p
        }
    }


def main():
    print("=" * 80)
    print("🚀 RUN8 test40 最终冻结组学评估")
    print("=" * 80)
    print(f"RECON_DIR          = {RECON_DIR}")
    print(f"TEST_CLINICAL_FILE = {TEST_CLINICAL_FILE}")
    print(f"DEV_ROOT           = {DEV_ROOT}")
    print(f"YAML_PATH          = {YAML_PATH}")
    print("=" * 80)

    if not RECON_DIR.exists():
        raise FileNotFoundError(f"❌ reconstructed_test40 不存在：{RECON_DIR}")
    if not TEST_CLINICAL_FILE.exists():
        raise FileNotFoundError(f"❌ 找不到 test 临床表：{TEST_CLINICAL_FILE}")
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到 radiomics yaml：{YAML_PATH}")

    test_ids = collect_test40_ids()
    df_test_clin = load_and_prepare_clinical(TEST_CLINICAL_FILE)
    df_test_clin = df_test_clin[df_test_clin["Patient_ID"].isin(test_ids)].copy()

    # 使用 dev164 保存的填补器填补 test 临床
    artifacts = build_dev_gt_reference()
    for col, val in artifacts["imputer_dict"].items():
        if col in df_test_clin.columns:
            df_test_clin[col] = df_test_clin[col].fillna(val)

    if len(df_test_clin) == 0:
        raise RuntimeError("❌ test 临床表过滤到 test40 后为空")
    if len(df_test_clin) != len(test_ids):
        print(f"⚠️ 提醒：clinical_info_test.xlsx 中匹配到 {len(df_test_clin)} 例，reconstructed_test40 中有 {len(test_ids)} 例。")

    feature_cols_all = artifacts["feature_cols_all"]

    # =====================================================
    # 1. 提取 test40 的 GT / Pred_080 / Pred_075 特征
    # =====================================================
    gt_csv = FEATURE_DIR / "Test40_GT_Features.csv"
    p080_csv = FEATURE_DIR / "Test40_Pred_080_Features.csv"
    p075_csv = FEATURE_DIR / "Test40_Pred_075_Features.csv"

    print("\n📊 正在提取 test40 GT 特征...")
    df_gt_feat = extract_features_for_mask("GT", feature_cols_all, gt_csv)

    print("\n📊 正在提取 test40 Pred_080 特征...")
    df_080_feat = extract_features_for_mask("080", feature_cols_all, p080_csv)

    print("\n📊 正在提取 test40 Pred_075 特征...")
    df_075_feat = extract_features_for_mask("075", feature_cols_all, p075_csv)

    # =====================================================
    # 2. 最终冻结评估
    # =====================================================
    gt_res = evaluate_one_path("GT", df_gt_feat, df_test_clin, artifacts)
    p080_res = evaluate_one_path("Pred_080", df_080_feat, df_test_clin, artifacts)
    p075_res = evaluate_one_path("Pred_075", df_075_feat, df_test_clin, artifacts)

    # DeLong
    y_test = gt_res["table"]["Label"].values
    delong_080_vs_gt = delong_roc_test(y_test, gt_res["prob"], p080_res["prob"])
    delong_075_vs_gt = delong_roc_test(y_test, gt_res["prob"], p075_res["prob"])

    gt_res["summary"]["DeLong_P_vs_GT"] = np.nan
    p080_res["summary"]["DeLong_P_vs_GT"] = delong_080_vs_gt
    p075_res["summary"]["DeLong_P_vs_GT"] = delong_075_vs_gt

    # =====================================================
    # 3. 保存结果
    # =====================================================
    gt_res["table"].to_csv(RESULT_DIR / "Test40_GT_Final_Table.csv", index=False, encoding="utf-8-sig")
    p080_res["table"].to_csv(RESULT_DIR / "Test40_Pred_080_Final_Table.csv", index=False, encoding="utf-8-sig")
    p075_res["table"].to_csv(RESULT_DIR / "Test40_Pred_075_Final_Table.csv", index=False, encoding="utf-8-sig")

    df_summary = pd.DataFrame([
        gt_res["summary"],
        p080_res["summary"],
        p075_res["summary"]
    ])
    summary_csv = RESULT_DIR / "Test40_Final_Summary.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(RESULT_DIR / "Test40_Final_Summary.txt", "w", encoding="utf-8") as f:
        f.write("RUN8 test40 final frozen evaluation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Locked cutoff (from dev164): {artifacts['optimal_cutoff']:.6f}\n")
        f.write(f"Healthy minimum Rad-score  : {artifacts['healthy_min_radscore']:.6f}\n")
        f.write("\n")
        for row in [gt_res["summary"], p080_res["summary"], p075_res["summary"]]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n🎉 test40 最终冻结评估完成。")
    print(f"📄 Summary CSV: {summary_csv}")
    print(f"📁 Features dir: {FEATURE_DIR}")
    print(f"📁 Results dir : {RESULT_DIR}")


if __name__ == "__main__":
    main()