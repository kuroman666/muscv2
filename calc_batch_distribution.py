import os
import pandas as pd

# ==========================================
# 1. 路徑設定（與前面腳本相同）
# ==========================================
MASTER_GT_CSV = "master_gt.csv"
OUTPUT_CSV = "batch_distribution.csv"
RAW_DIRS = {
    "W639": "/path/to/W639_raw",
    "W642": "/path/to/W642_raw",
    "W646": "/path/to/W646_raw",
}
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ==========================================
# 2. 計算邏輯
# ==========================================
def count_images(folder: str) -> int:
    """計算資料夾內（含子資料夾）所有圖片總張數"""
    if not os.path.exists(folder):
        print(f"⚠️  [警告] 找不到資料夾：{folder}")
        return 0
    count = 0
    for _, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                count += 1
    return count


def make_row(batch_name: str, total: int, defect: int) -> dict:
    normal = max(total - defect, 0)
    defect_pct = (defect / total) if total > 0 else 0.0
    normal_pct = (normal / total) if total > 0 else 0.0
    ratio_str = f"{normal / defect:.1f} : 1" if defect > 0 else "N/A"

    return {
        "Batch": batch_name,
        "總張數": total,
        "正常張數": normal,
        "瑕疵張數": defect,
        "正常比例": f"{normal_pct:.2%}",
        "瑕疵比例": f"{defect_pct:.2%}",
        "正常:瑕疵比": ratio_str,
    }


def main():
    # 讀取 master_gt.csv，只計算工程師確認為真瑕疵的張數 (engineer_confirmed == 1)
    df = pd.read_csv(MASTER_GT_CSV)
    df["engineer_confirmed"] = pd.to_numeric(df["engineer_confirmed"], errors="coerce").fillna(0).astype(int)
    valid_defects = df[df["engineer_confirmed"] == 1]

    rows = []
    sum_total, sum_defect = 0, 0

    for batch, folder in RAW_DIRS.items():
        total = count_images(folder)
        defect = len(valid_defects[valid_defects["batch"] == batch])
        rows.append(make_row(batch, total, defect))
        sum_total += total
        sum_defect += defect

    # 加上三批總計
    rows.append(make_row("ALL (總計)", sum_total, sum_defect))

    result_df = pd.DataFrame(rows)
    print("=" * 72)
    print("📊 各批次正常與瑕疵分布統計")
    print("=" * 72)
    print(result_df.to_string(index=False))

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 72)
    print(f"✅ 已存成：{OUTPUT_CSV}（可直接用 Excel 打開貼進投影片）")


if __name__ == "__main__":
    main()