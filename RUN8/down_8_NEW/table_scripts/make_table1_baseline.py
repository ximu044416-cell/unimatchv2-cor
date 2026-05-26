import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =========================
# 0. Paths
# =========================
DEV_FILE = Path(r"F:\cor\RUN8\down_8_NEW\down_2_threshold\clinical\clinical_info_dev164.xlsx")
TEST_FILE = Path(r"F:\cor\RUN8\down_8_NEW\down_2_threshold\clinical\clinical_info_test.xlsx")

OUT_DIR = Path(r"F:\cor\RUN8\down_8_NEW\tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table1_Baseline_Characteristics_204_164_40.csv"
OUT_XLSX = OUT_DIR / "Table1_Baseline_Characteristics_204_164_40.xlsx"


# =========================
# 1. Helpers
# =========================
def fmt_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_mean_sd(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return f"{x.mean():.1f} ± {x.std(ddof=1):.1f}"


def fmt_median_iqr(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    q1 = np.percentile(x, 25)
    q2 = np.percentile(x, 50)
    q3 = np.percentile(x, 75)
    return f"{q2:.1f} ({q1:.1f}–{q3:.1f})"


def fmt_count_pct(df, col, value=1):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    n = len(s)
    c = int((s == value).sum())
    pct = c / n * 100 if n > 0 else np.nan
    return f"{c}/{n} ({pct:.1f}%)"


def p_continuous(dev, test, col, method="mannwhitney"):
    x = pd.to_numeric(dev[col], errors="coerce").dropna()
    y = pd.to_numeric(test[col], errors="coerce").dropna()

    if method == "t":
        return stats.ttest_ind(x, y, equal_var=False).pvalue
    else:
        return stats.mannwhitneyu(x, y, alternative="two-sided").pvalue


def p_categorical(dev, test, col):
    x = pd.to_numeric(dev[col], errors="coerce").dropna().astype(int)
    y = pd.to_numeric(test[col], errors="coerce").dropna().astype(int)

    cats = sorted(set(x.tolist() + y.tolist()))
    table = np.array([
        [(x == c).sum() for c in cats],
        [(y == c).sum() for c in cats]
    ])

    if table.shape == (2, 2):
        chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
        if (exp < 5).any():
            p = stats.fisher_exact(table)[1]
        return p
    else:
        chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
        return p


# =========================
# 2. Main
# =========================
def main():
    if not DEV_FILE.exists():
        raise FileNotFoundError(f"Cannot find dev file: {DEV_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Cannot find test file: {TEST_FILE}")

    dev = pd.read_excel(DEV_FILE)
    test = pd.read_excel(TEST_FILE)

    dev["Cohort"] = "Development"
    test["Cohort"] = "Independent test"

    all_df = pd.concat([dev, test], axis=0, ignore_index=True)

    print(f"Overall n = {len(all_df)}")
    print(f"Development n = {len(dev)}")
    print(f"Independent test n = {len(test)}")

    rows = []

    rows.append({
        "Variable": "Age, years",
        "Overall cohort (n=204)": fmt_mean_sd(all_df["Age"]),
        "Development cohort (n=164)": fmt_mean_sd(dev["Age"]),
        "Independent test cohort (n=40)": fmt_mean_sd(test["Age"]),
        "P value": fmt_p(p_continuous(dev, test, "Age", method="t")),
        "Test": "Welch t test"
    })

    rows.append({
        "Variable": "Gender = 1, n/N (%)",
        "Overall cohort (n=204)": fmt_count_pct(all_df, "Gender", 1),
        "Development cohort (n=164)": fmt_count_pct(dev, "Gender", 1),
        "Independent test cohort (n=40)": fmt_count_pct(test, "Gender", 1),
        "P value": fmt_p(p_categorical(dev, test, "Gender")),
        "Test": "Chi-square or Fisher exact test"
    })

    rows.append({
        "Variable": "Label = 1, n/N (%)",
        "Overall cohort (n=204)": fmt_count_pct(all_df, "Label", 1),
        "Development cohort (n=164)": fmt_count_pct(dev, "Label", 1),
        "Independent test cohort (n=40)": fmt_count_pct(test, "Label", 1),
        "P value": fmt_p(p_categorical(dev, test, "Label")),
        "Test": "Chi-square or Fisher exact test"
    })

    rows.append({
        "Variable": "CRP, median (IQR)",
        "Overall cohort (n=204)": fmt_median_iqr(all_df["CRP"]),
        "Development cohort (n=164)": fmt_median_iqr(dev["CRP"]),
        "Independent test cohort (n=40)": fmt_median_iqr(test["CRP"]),
        "P value": fmt_p(p_continuous(dev, test, "CRP", method="mannwhitney")),
        "Test": "Mann–Whitney U test"
    })

    rows.append({
        "Variable": "ESR, median (IQR)",
        "Overall cohort (n=204)": fmt_median_iqr(all_df["ESR"]),
        "Development cohort (n=164)": fmt_median_iqr(dev["ESR"]),
        "Independent test cohort (n=40)": fmt_median_iqr(test["ESR"]),
        "P value": fmt_p(p_continuous(dev, test, "ESR", method="mannwhitney")),
        "Test": "Mann–Whitney U test"
    })

    rows.append({
        "Variable": "HLA-B27 positive, n/N (%)",
        "Overall cohort (n=204)": fmt_count_pct(all_df, "HLA-B27", 1),
        "Development cohort (n=164)": fmt_count_pct(dev, "HLA-B27", 1),
        "Independent test cohort (n=40)": fmt_count_pct(test, "HLA-B27", 1),
        "P value": fmt_p(p_categorical(dev, test, "HLA-B27")),
        "Test": "Chi-square or Fisher exact test"
    })

    rows.append({
        "Variable": "Disease duration category = 1, n/N (%)",
        "Overall cohort (n=204)": fmt_count_pct(all_df, "Disease_Duration_Category", 1),
        "Development cohort (n=164)": fmt_count_pct(dev, "Disease_Duration_Category", 1),
        "Independent test cohort (n=40)": fmt_count_pct(test, "Disease_Duration_Category", 1),
        "P value": fmt_p(p_categorical(dev, test, "Disease_Duration_Category")),
        "Test": "Chi-square or Fisher exact test"
    })

    rows.append({
        "Variable": "SPARCC score, median (IQR)",
        "Overall cohort (n=204)": fmt_median_iqr(all_df["sparcc"]),
        "Development cohort (n=164)": fmt_median_iqr(dev["sparcc"]),
        "Independent test cohort (n=40)": fmt_median_iqr(test["sparcc"]),
        "P value": fmt_p(p_continuous(dev, test, "sparcc", method="mannwhitney")),
        "Test": "Mann–Whitney U test"
    })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    table.to_excel(OUT_XLSX, index=False)

    print("Saved:")
    print(OUT_CSV)
    print(OUT_XLSX)
    print(table)


if __name__ == "__main__":
    main()