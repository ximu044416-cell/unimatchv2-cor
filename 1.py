from pathlib import Path
import shutil
import csv
import math

ROOT = Path(r"F:\Dinov2_data\ALL_COR_YOLO")

# False = 如果已经存在 label2.nii.gz，就跳过，避免覆盖你已经修改过的版本
# True = 强制用 label.nii.gz 覆盖 label2.nii.gz，不建议平时打开
OVERWRITE = False

LOG_PATH = ROOT / "copy_label2_log.csv"

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


def close_enough_tuple(a, b, tol=1e-5):
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def check_geometry(image_path, label_path):
    """
    检查 water.nii.gz 和 label/label2.nii.gz 的几何信息是否一致。
    只检查，不修改。
    """
    if not HAS_SITK:
        return "not_checked_no_SimpleITK"

    try:
        img = sitk.ReadImage(str(image_path))
        lab = sitk.ReadImage(str(label_path))

        same_size = img.GetSize() == lab.GetSize()
        same_spacing = close_enough_tuple(img.GetSpacing(), lab.GetSpacing())
        same_origin = close_enough_tuple(img.GetOrigin(), lab.GetOrigin())
        same_direction = close_enough_tuple(img.GetDirection(), lab.GetDirection())

        if same_size and same_spacing and same_origin and same_direction:
            return "geometry_ok"

        problems = []
        if not same_size:
            problems.append("size")
        if not same_spacing:
            problems.append("spacing")
        if not same_origin:
            problems.append("origin")
        if not same_direction:
            problems.append("direction")

        return "geometry_mismatch_" + "_".join(problems)

    except Exception as e:
        return f"geometry_check_failed: {e}"


def main():
    label_files = sorted(ROOT.rglob("label.nii.gz"))

    rows = []
    copied = 0
    skipped = 0
    mismatch = 0

    for label_path in label_files:
        case_dir = label_path.parent
        label2_path = case_dir / "label2.nii.gz"
        water_path = case_dir / "water.nii.gz"

        if label2_path.exists() and not OVERWRITE:
            status = "skipped_label2_exists"
            skipped += 1
        else:
            shutil.copy2(label_path, label2_path)
            status = "copied"
            copied += 1

        if water_path.exists():
            geo_status = check_geometry(water_path, label2_path)
        else:
            geo_status = "water_missing"

        if geo_status.startswith("geometry_mismatch"):
            mismatch += 1

        rows.append({
            "case_dir": str(case_dir),
            "label": str(label_path),
            "label2": str(label2_path),
            "status": status,
            "geometry_status_vs_water": geo_status,
        })

    with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_dir",
                "label",
                "label2",
                "status",
                "geometry_status_vs_water",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Root: {ROOT}")
    print(f"Found label.nii.gz: {len(label_files)}")
    print(f"Copied: {copied}")
    print(f"Skipped existing label2: {skipped}")
    print(f"Geometry mismatch vs water: {mismatch}")
    print(f"Log saved to: {LOG_PATH}")

    if not HAS_SITK:
        print("\nNote: SimpleITK is not installed, so geometry was not checked.")
        print("You can install it with: pip install SimpleITK")


if __name__ == "__main__":
    main()