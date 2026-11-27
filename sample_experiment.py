import os
import random
import shutil
import pandas as pd

# ==========================================
# 1. 基本路徑設定（請改成你實際的路徑）
# ==========================================
MASTER_GT_CSV = "master_gt.csv"
RAW_DIRS = {
    "W639": "/path/to/W639_raw",
    "W642": "/path/to/W642_raw",
    "W646": "/path/to/W646_raw",
}

RANDOM_SEED = 42  # 固定隨機種子，確保每次抽樣結果可重現
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ==========================================
# 2. 選擇實驗模式（改這裡的字串就能快速切換實驗）
#    - defect 填 "all" 代表把該批所有最新確認瑕疵全拿出來
# ==========================================
ACTIVE_MODE = "100_normal_each"

EXPERIMENT_PRESETS = {
    # 模式 1：原本的 1:1 (15 瑕疵 : 15 正常)
    "1_to_1": {
        "W639": {"defect": 5, "normal": 5},
        "W642": {"defect": 5, "normal": 5},
        "W646": {"defect": 5, "normal": 5},
    },
    # 模式 2：7:3 比例 (35 正常 : 15 瑕疵 -> 70% : 30%)
    "7_to_3": {
        "W639": {"defect": 5, "normal": 12},
        "W642": {"defect": 5, "normal": 12},
        "W646": {"defect": 5, "normal": 11},
    },
    # 模式 3：正常樣本放大 20 倍 (各批抽 100 張正常 + 各 5 張瑕疵 = 300 : 15)
    "100_normal_each": {
        "W639": {"defect": 5, "normal": 100},
        "W642": {"defect": 5, "normal": 100},
        "W646": {"defect": 5, "normal": 100},
    },
    # 模式 4：Dataset 2 跨批混合實驗（拿全部最新確認瑕疵 + 各批抽 100 張正常，張數可自調）
    "dataset2_all_defects": {
        "W639": {"defect": "all", "normal": 100},
        "W642": {"defect": "all", "normal": 100},
        "W646": {"defect": "all", "normal": 100},
    },
}

# 輸出資料夾名稱（會自動依模式命名，不怕蓋掉舊實驗）
OUTPUT_DIR = f"exp_{ACTIVE_MODE}"


# ==========================================
# 3. 核心邏輯
# ==========================================
def scan_raw_folder(folder: str) -> dict:
    """掃描 raw_dir，回傳 {無副檔名檔名: 完整檔案路徑}"""
    stem_to_path = {}
    if not os.path.exists(folder):
        print(f"⚠️  [警告] 找不到資料夾：{folder}")
        return stem_to_path

    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                stem = os.path.splitext(fname)[0]
                stem_to_path[stem] = os.path.join(root, fname)
    return stem_to_path


def uniform_sample(pool: list, k) -> list:
    """均勻隨機抽樣 (Uniform Sampling without replacement)"""
    if k == "all" or k >= len(pool):
        return list(pool)
    return random.sample(pool, k)


def main():
    random.seed(RANDOM_SEED)

    if ACTIVE_MODE not in EXPERIMENT_PRESETS:
        raise ValueError(f"找不到模式 '{ACTIVE_MODE}'，請檢查 EXPERIMENT_PRESETS。")

    plan = EXPERIMENT_PRESETS[ACTIVE_MODE]

    # 建立輸出資料夾（若已存在先清空 images，避免混到上次的圖）
    img_out_dir = os.path.join(OUTPUT_DIR, "images")
    if os.path.exists(img_out_dir):
        shutil.rmtree(img_out_dir)
    os.makedirs(img_out_dir, exist_ok=True)

    # 讀取 master_gt.csv，只取工程師確認為真瑕疵的清單 (engineer_confirmed == 1)
    gt_df = pd.read_csv(MASTER_GT_CSV)
    gt_df["engineer_confirmed"] = pd.to_numeric(gt_df["engineer_confirmed"], errors="coerce").fillna(0).astype(int)
    confirmed_df = gt_df[gt_df["engineer_confirmed"] == 1]

    gt_rows = []
    print("=" * 65)
    print(f"🚀 開始抽樣實驗：{ACTIVE_MODE} (Seed = {RANDOM_SEED})")
    print("=" * 65)

    for batch, counts in plan.items():
        raw_map = scan_raw_folder(RAW_DIRS[batch])
        if not raw_map:
            continue

        # 該批次已確認的瑕疵檔名集合
        batch_defect_stems = set(confirmed_df[confirmed_df["batch"] == batch]["filename"].astype(str))

        # 切分「瑕疵池」與「正常池」（排序後再抽，確保跨平台結果一致）
        defect_pool = sorted([s for s in raw_map.keys() if s in batch_defect_stems])
        normal_pool = sorted([s for s in raw_map.keys() if s not in batch_defect_stems])

        # 檢查是否有瑕疵在 raw_dir 找不到
        missing_defects = batch_defect_stems - set(defect_pool)
        if missing_defects:
            print(f"⚠️  [{batch}] 有 {len(missing_defects)} 張 master_gt 瑕疵圖在 raw_dir 找不到原檔")

        # Uniform Sampling 抽樣
        sampled_defects = uniform_sample(defect_pool, counts["defect"])
        sampled_normals = uniform_sample(normal_pool, counts["normal"])

        # 複製圖片並記錄進 dataset1_gt.csv
        for stem, label in [(s, 1) for s in sampled_defects] + [(s, 0) for s in sampled_normals]:
            src_path = raw_map[stem]
            orig_fname = os.path.basename(src_path)
            dst_path = os.path.join(img_out_dir, orig_fname)
            shutil.copy2(src_path, dst_path)

            gt_rows.append({
                "batch": batch,
                "filename": orig_fname,
                "gt_label": label,
            })

        print(f"📦 【{batch}】原圖共 {len(raw_map)} 張（瑕疵池 {len(defect_pool)} / 正常池 {len(normal_pool)}）")
        print(f"    -> 抽出：瑕疵 {len(sampled_defects)} 張 + 正常 {len(sampled_normals)} 張")

    # 輸出 dataset1_gt.csv
    out_csv_path = os.path.join(OUTPUT_DIR, "dataset1_gt.csv")
    out_df = pd.DataFrame(gt_rows)
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    total_d = int((out_df["gt_label"] == 1).sum())
    total_n = int((out_df["gt_label"] == 0).sum())
    print("\n" + "=" * 65)
    print(f"✅ 抽樣完成！共 {len(out_df)} 張（瑕疵 {total_d} 張 / 正常 {total_n} 張）")
    print(f"📁 圖片已複製至：{img_out_dir}")
    print(f"📄 答案卷已產出：{out_csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()