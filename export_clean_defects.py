import os
import shutil
import pandas as pd

MASTER_GT_CSV = "master_gt.csv"
OUTPUT_DIR = "Dataset2_clean_defects"
RAW_DIRS = {
    "W639": "/path/to/W639_raw",
    "W642": "/path/to/W642_raw",
    "W646": "/path/to/W646_raw",
}
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

df = pd.read_csv(MASTER_GT_CSV)
df["engineer_confirmed"] = pd.to_numeric(df["engineer_confirmed"], errors="coerce").fillna(0).astype(int)
valid_df = df[df["engineer_confirmed"] == 1]

for batch, raw_dir in RAW_DIRS.items():
    if not os.path.exists(raw_dir):
        print(f"⚠️ 找不到原始資料夾：{raw_dir}")
        continue

    # 建立 {乾淨原檔名: 原始圖檔完整路徑}
    raw_map = {}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                raw_map[os.path.splitext(f)[0]] = os.path.join(root, f)

    out_batch_dir = os.path.join(OUTPUT_DIR, batch)
    os.makedirs(out_batch_dir, exist_ok=True)

    stems = valid_df[valid_df["batch"] == batch]["filename"].astype(str).tolist()
    copied, missing = 0, []
    for s in stems:
        if s in raw_map:
            src = raw_map[s]
            shutil.copy2(src, os.path.join(out_batch_dir, os.path.basename(src)))
            copied += 1
        else:
            missing.append(s)

    print(f"✅ [{batch}] 已複製 {copied}/{len(stems)} 張乾淨原圖至 {out_batch_dir}")
    if missing:
        print(f"   ⚠️ 找不到原圖 ({len(missing)} 張)：{missing[:5]}")