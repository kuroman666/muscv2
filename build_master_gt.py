import os
import re
import pandas as pd

# ==========================================
# 1. 設定各批次的資料夾路徑（請改成你電腦/伺服器上的實際路徑）
#    - raw_dir: 該批次最原始的圖片資料夾（用來核對檔名防呆，若暫時沒有可填 None）
#    - foundad_dir: FoundAD 抓出來的瑕疵圖片資料夾
#    - muscv2_dir: MuSc-V2 抓出來的瑕疵圖片資料夾（含 Rank 前綴沒關係）
# ==========================================
BATCH_CONFIG = {
    "W639": {
        "raw_dir": "/path/to/W639_raw",
        "foundad_dir": "/path/to/W639_foundad_defects",
        "muscv2_dir": "/path/to/W639_muscv2_defects",
    },
    "W642": {
        "raw_dir": "/path/to/W642_raw",
        "foundad_dir": "/path/to/W642_foundad_defects",
        "muscv2_dir": "/path/to/W642_muscv2_defects",
    },
    "W646": {
        "raw_dir": "/path/to/W646_raw",
        "foundad_dir": "/path/to/W646_foundad_defects",
        "muscv2_dir": "/path/to/W646_muscv2_defects",
    },
}

OUTPUT_CSV = "master_gt.csv"
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ==========================================
# 2. 核心工具函式
# ==========================================
def list_image_files(folder: str) -> list:
    """掃描資料夾內所有圖片（含子資料夾），回傳檔名清單"""
    if not folder or not os.path.exists(folder):
        return []
    files = []
    for root, _, filenames in os.walk(folder):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                files.append(fname)
    return files


def extract_clean_stem(filename: str) -> str:
    """剝掉 MuSc-V2 的 Rank 前綴、子資料夾記號與副檔名"""
    base = os.path.basename(filename)
    # 剝掉開頭的 "Rank001_Score0.8720_Box01_"
    cleaned = re.sub(r"^Rank\d+_Score[\d\.]+_Box\d+_", "", base)
    # 如果原本有子資料夾被換成 "__"，只取最後一段
    if "__" in cleaned:
        cleaned = cleaned.split("__")[-1]
    # 去掉副檔名
    return os.path.splitext(cleaned)[0]


def map_to_original_names(defect_files: list, raw_stems: set, batch: str, model_name: str) -> set:
    """將模型輸出的瑕疵檔名還原成原檔名，並與 raw_dir 核對防呆"""
    matched_stems = set()
    unmatched_files = []

    for fname in defect_files:
        clean_stem = extract_clean_stem(fname)

        # 如果沒提供 raw_dir，就直接相信剝完前綴的檔名
        if not raw_stems:
            matched_stems.add(clean_stem)
            continue

        # 情況 1：剝完前綴後，跟原圖檔名完全一致（最常見）
        if clean_stem in raw_stems:
            matched_stems.add(clean_stem)
            continue

        # 情況 2：保險機制（包含比對），防止 FoundAD 或人工加了其他前後綴
        candidates = [raw for raw in raw_stems if raw in fname]
        if len(candidates) == 1:
            matched_stems.add(candidates[0])
        elif len(candidates) > 1:
            # 若有多個候選，取最長的那個原檔名（最精確）
            best_match = max(candidates, key=len)
            matched_stems.add(best_match)
        else:
            # 真的完全對不上，記錄起來印警告
            unmatched_files.append(fname)
            matched_stems.add(clean_stem)  # 仍先保留，避免漏算

    if unmatched_files:
        print(f"⚠️  [警告] {batch} - {model_name} 有 {len(unmatched_files)} 張圖在原始資料夾找不到對應原檔名：")
        for uf in unmatched_files:
            print(f"    -> {uf}")

    return matched_stems


# ==========================================
# 3. 主程式：比對、扣除 Overlap、輸出 CSV 與統計
# ==========================================
def main():
    all_rows = []
    print("=" * 60)
    print("開始掃描並比對各批次瑕疵資料夾...")
    print("=" * 60)

    for batch, paths in BATCH_CONFIG.items():
        # 讀取原始資料夾檔名（當作字典）
        raw_files = list_image_files(paths.get("raw_dir"))
        raw_stems = {os.path.splitext(f)[0] for f in raw_files}

        # 讀取兩個模型的瑕疵資料夾
        foundad_files = list_image_files(paths.get("foundad_dir"))
        muscv2_files = list_image_files(paths.get("muscv2_dir"))

        # 還原原檔名並檢查有無對不上的檔案
        foundad_set = map_to_original_names(foundad_files, raw_stems, batch, "FoundAD")
        muscv2_set = map_to_original_names(muscv2_files, raw_stems, batch, "MuSc-V2")

        # 用集合 (set) 自動取聯集與交集（自動扣除 Overlap）
        all_defects = sorted(foundad_set | muscv2_set)
        overlap_set = foundad_set & muscv2_set
        f_only_set = foundad_set - muscv2_set
        m_only_set = muscv2_set - foundad_set

        # 整理進總表
        for stem in all_defects:
            all_rows.append({
                "batch": batch,
                "filename": stem,
                "foundad_hit": 1 if stem in foundad_set else 0,
                "muscv2_hit": 1 if stem in muscv2_set else 0,
                "engineer_confirmed": 1,  # 預設皆為真瑕疵；之後工程師退回只要把這欄改成 0
            })

        # 印出該批次統計
        total_gt = len(all_defects)
        f_recall = len(foundad_set) / total_gt if total_gt > 0 else 0
        m_recall = len(muscv2_set) / total_gt if total_gt > 0 else 0

        print(f"\n📊 【{batch} 統計結果】")
        print(f"  - 實際瑕疵母體總數（已扣 Overlap）: {total_gt} 張")
        print(f"  - FoundAD 獨有: {len(f_only_set)} 張 | MuSc-V2 獨有: {len(m_only_set)} 張 | 重疊 (Overlap): {len(overlap_set)} 張")
        print(f"  - FoundAD Recall: {f_recall:.2%} ({len(foundad_set)}/{total_gt})")
        print(f"  - MuSc-V2 Recall: {m_recall:.2%} ({len(muscv2_set)}/{total_gt})")

    # 輸出 master_gt.csv
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 60)
    print(f"✅ 已成功輸出總表：{OUTPUT_CSV}（共 {len(df)} 筆不重複瑕疵）")
    print("💡 之後工程師若退回部分圖片，只要打開 CSV 把該列的 engineer_confirmed 改成 0 即可。")


if __name__ == "__main__":
    main()