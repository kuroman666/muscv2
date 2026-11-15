#!/usr/bin/env bash
# 檔案位置: MuSc-V2/run_outside_muscv2.sh
# 用途: MuSc-V2 廠外公開資料集 (MVTec AD / MVTec AD 2) Training-Free & Zero-Shot 無 GT 模擬抓漏與拆分評估腳本
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./output"
DATASETS_DIR="./data"
DINO_CKPT_PATH="auto"                 # 自動從 ~/.cache/torch/hub 或 ./hub_cache 搜尋 DINOv2/v3 權重
BACKBONE_NAME="dinov2_vitl14"

if [ $# -ge 1 ]; then
    TARGET_CAT_DIRS=("$@")
else
    TARGET_CAT_DIRS=(
        "${DATASETS_DIR}/mvtec/capsule"
        # "${DATASETS_DIR}/mvtec-2/can"
        "${DATASETS_DIR}/mvtec/bottle"
        "${DATASETS_DIR}/mvtec/cable"
        "${DATASETS_DIR}/mvtec/grid"
        "${DATASETS_DIR}/mvtec/tile"
        "${DATASETS_DIR}/mvtec/transistor"
    )
fi

POOL_MODE="escape"
DEFECT_LIMIT=8
REBUILD_POOL=""                       # 若想強制重建模擬盲池，設為 "--rebuild_pool"
FORCE_INFER=""                        # 若想強制重切測試塊並重跑推論，設為 "--force_infer"
RUN_SPLIT_EVAL=1

# --- MuSc-V2 Zero-Shot 互評超參數 ---
R_LIST="1,3,5"
FEATURE_LAYERS=""                     # 留空自動配置 (24層ViT-L預設 5,11,17,23；12層ViT-B預設 2,5,8,11)
MUTUAL_RATIO=0.30
RSCON_WEIGHT=0.15
MAX_PEER_TILES=400

# --- 動態門檻與防誤報超參數 ---
THRESH_MODE="hybrid"
SIGMA_K=3.5
THRESHOLD=0.25
MIN_THRESHOLD=0
AUTO_PERCENTILE=95.0
TOP_K=0
CORE_RATIO=0.55
RAW_SCORE_MODE=""                     # 若想使用未濾波原生分數，設為 "--raw_score_mode"

WAFER_EDGE_MODE="auto"
WAFER_EDGE_MARGIN=0.04
BORDER_RATIO=0.02
SPECKLE_FILTER_RATIO=0.008
MIN_AREA_RATIO=0.00012

# --- 智慧解析度防呆切塊與縫合參數 ---
MIN_TILING_RES=1536                   # 大圖防呆門檻：僅當長邊 >= 1536px 才切塊；廠外公開集 (700~1024px) 自動整張推論
TILE_SIZE=512
TILE_OVERLAP=64
INFER_BATCH_SIZE=16
BG_STD_THRESH=1.5
DISABLE_TILING=""                     # 若想強制關閉切塊，設為 "--disable_tiling"

# --- Step 2 評估結果拆分參數 ---
NUM_PACKS=3
SPLIT_COUNTS=""
SPLIT_RATIOS="1:1:1"
SPLIT_ORDER="stratified"
RECOMPUTE_PACK_THRESH=""
USE_SYMLINK="--symlink"
# ===============================================================

BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${OUTPUT_ROOT}"
BATCH_SUMMARY_TXT="${OUTPUT_ROOT}/multi_dataset_batch_summary_muscv2_${BATCH_TIMESTAMP}.txt"
BATCH_SUMMARY_LATEST="${OUTPUT_ROOT}/latest_multi_dataset_summary_muscv2.txt"
BATCH_SUMMARY_CSV="${OUTPUT_ROOT}/latest_multi_dataset_summary_muscv2.csv"

echo "==========================================================================================================" > "${BATCH_SUMMARY_TXT}"
echo " 📊 [MuSc-V2 (Zero-Shot) 多資料集批次推論總覽報表] 啟動時間: ${BATCH_TIMESTAMP}" >> "${BATCH_SUMMARY_TXT}"
echo "==========================================================================================================" >> "${BATCH_SUMMARY_TXT}"
echo "Dataset_Key,Category,Total_Imgs,Normal_Imgs,Escape_Imgs,Eff_Threshold,Recall_Pct,FPR_Pct,Normal_Mean,Normal_Std,Infer_Dir,Report_CSV" > "${BATCH_SUMMARY_CSV}"

prepare_public_sim_pool() {
    local SRC_CAT_DIR="$1"
    local DST_POOL_DIR="$2"
    local MODE="$3"
    local LIMIT="$4"
    local FORCE_REBUILD="$5"

    python - "${SRC_CAT_DIR}" "${DST_POOL_DIR}" "${MODE}" "${LIMIT}" "${FORCE_REBUILD}" << 'EOF'
import os
import sys
import stat
import shutil

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
src_cat_dir, dst_pool_dir, mode, defect_limit, force_rebuild = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    sys.argv[5],
)

if os.path.isdir(dst_pool_dir) and not force_rebuild:
    existing = [f for f in os.listdir(dst_pool_dir) if f.lower().endswith(IMG_EXTS)]
    if existing:
        print(f"[⚡ 模擬無GT盲池快取命中] 直接沿用既有 {len(existing)} 張盲池圖片 -> {os.path.abspath(dst_pool_dir)}")
        sys.exit(0)

test_dir = None
for cand in ["test", "test_public", "val", "validation"]:
    cand_path = os.path.join(src_cat_dir, cand)
    if os.path.isdir(cand_path) and os.path.isdir(os.path.join(cand_path, "good")):
        test_dir = cand_path
        break

if test_dir is None:
    raise FileNotFoundError(
        f"在 {src_cat_dir} 下找不到包含 good/ 的測試資料夾 (已檢查 test, test_public, validation)！"
    )

if os.path.exists(dst_pool_dir):
    shutil.rmtree(dst_pool_dir)
os.makedirs(dst_pool_dir, exist_ok=True)

def copy_unlocked(src, dst):
    shutil.copyfile(src, dst)
    try:
        os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass

good_dir = os.path.join(test_dir, "good")
good_count = 0
for root, _, files in os.walk(good_dir):
    for f in sorted(files):
        if f.lower().endswith(IMG_EXTS) and not f.startswith("."):
            copy_unlocked(os.path.join(root, f), os.path.join(dst_pool_dir, f"clean_{f}"))
            good_count += 1

if good_count < 8:
    for backup_sub in ["validation/good", "train/good"]:
        b_dir = os.path.join(src_cat_dir, backup_sub)
        if os.path.isdir(b_dir):
            for f in sorted(os.listdir(b_dir)):
                if f.lower().endswith(IMG_EXTS) and not f.startswith("."):
                    dst_f = os.path.join(dst_pool_dir, f"clean_val_{f}")
                    if not os.path.exists(dst_f):
                        copy_unlocked(os.path.join(b_dir, f), dst_f)
                        good_count += 1
                        if good_count >= 25:
                            break
        if good_count >= 25:
            break

defect_count = 0
if mode == "escape" and defect_limit > 0:
    for sub in sorted(os.listdir(test_dir)):
        if sub == "good":
            continue
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for root, _, files in os.walk(sub_dir):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS) and not f.startswith(".") and defect_count < defect_limit:
                    copy_unlocked(os.path.join(root, f), os.path.join(dst_pool_dir, f"escape_{sub}_{f}"))
                    defect_count += 1

print(
    f"[✓ 模擬無GT盲池建立完成] 來源: {test_dir} | 模式: {mode} | "
    f"共 {good_count + defect_count} 張圖 (正常: {good_count} 張, 隱藏瑕疵: {defect_count} 張) -> {os.path.abspath(dst_pool_dir)}"
)
EOF
}

run_single_public_category() {
    local CAT_DIR="$1"
    if [ ! -d "${CAT_DIR}" ]; then
        echo "❌ [路徑錯誤] 找不到類別資料夾: ${CAT_DIR}"
        exit 1
    fi

    local CAT_NAME
    CAT_NAME=$(basename "${CAT_DIR}")
    local PARENT_DATASET
    PARENT_DATASET=$(basename "$(dirname "${CAT_DIR}")")
    local DS_PREFIX="${PARENT_DATASET//-/_}"
    local UNIQUE_TAG="${DS_PREFIX}_${CAT_NAME}"

    local SIM_POOL_DIR=""
    local DATA_NAME=""
    local SAVE_BASE_DIR=""

    if [ "${POOL_MODE}" = "pure_normal" ]; then
        SIM_POOL_DIR="${DATASETS_DIR}/sim_pure_normal_${UNIQUE_TAG}_pool"
        DATA_NAME="${DS_PREFIX}_pure_${CAT_NAME}_muscv2"
        SAVE_BASE_DIR="${OUTPUT_ROOT}/3_pure_normal_${UNIQUE_TAG}_muscv2"
    else
        SIM_POOL_DIR="${DATASETS_DIR}/sim_fab_${UNIQUE_TAG}_pool"
        DATA_NAME="${DS_PREFIX}_sim_${CAT_NAME}_muscv2"
        SAVE_BASE_DIR="${OUTPUT_ROOT}/2_fab_no_gt_${UNIQUE_TAG}_muscv2"
    fi

    echo ""
    echo "==========================================================================="
    echo " 🚀 [廠外公開資料集無 GT 測試啟動 (MuSc-V2 Zero-Shot)] 資料集: ${PARENT_DATASET} | 類別: ${CAT_NAME}"
    echo "    DATA_NAME: ${DATA_NAME} | 獨立輸出目錄: ${SAVE_BASE_DIR}"
    echo "==========================================================================="

    # Step 0: 建立或沿用模擬盲池
    prepare_public_sim_pool "${CAT_DIR}" "${SIM_POOL_DIR}" "${POOL_MODE}" "${DEFECT_LIMIT}" "${REBUILD_POOL}"

    # Step 1: 執行 1_infer_muscv2_no_gt.py (Training-Free 直接互評推論)
    echo "---------------------------------------------------------------------------"
    echo " [Step 1/2] 執行 1_infer_muscv2_no_gt.py -> Zero-Shot 互評推理與防誤報紅框定位"
    echo "---------------------------------------------------------------------------"
    local LAYER_ARG=()
    if [ -n "${FEATURE_LAYERS}" ]; then
        LAYER_ARG=(--feature_layers "${FEATURE_LAYERS}")
    fi

    python 1_infer_muscv2_no_gt.py \
        --pool_dir "${SIM_POOL_DIR}" \
        --ckpt "${DINO_CKPT_PATH}" \
        --backbone "${BACKBONE_NAME}" \
        --out_dir "${SAVE_BASE_DIR}" \
        --datasets_dir "${DATASETS_DIR}" \
        --data_name "${DATA_NAME}" \
        --category "${CAT_NAME}" \
        --r_list "${R_LIST}" \
        "${LAYER_ARG[@]}" \
        --mutual_ratio "${MUTUAL_RATIO}" \
        --rscon_weight "${RSCON_WEIGHT}" \
        --max_peer_tiles "${MAX_PEER_TILES}" \
        --thresh_mode "${THRESH_MODE}" \
        --threshold "${THRESHOLD}" \
        --min_threshold "${MIN_THRESHOLD}" \
        --auto_percentile "${AUTO_PERCENTILE}" \
        --top_k "${TOP_K}" \
        --sigma_k "${SIGMA_K}" \
        --core_ratio "${CORE_RATIO}" \
        --border_ratio "${BORDER_RATIO}" \
        --wafer_edge_mode "${WAFER_EDGE_MODE}" \
        --wafer_edge_margin "${WAFER_EDGE_MARGIN}" \
        --speckle_filter_ratio "${SPECKLE_FILTER_RATIO}" \
        --min_area_ratio "${MIN_AREA_RATIO}" \
        --min_tiling_res "${MIN_TILING_RES}" \
        --tile_size "${TILE_SIZE}" \
        --tile_overlap "${TILE_OVERLAP}" \
        --infer_batch_size "${INFER_BATCH_SIZE}" \
        --bg_std_thresh "${BG_STD_THRESH}" \
        ${FORCE_INFER} ${RAW_SCORE_MODE} ${DISABLE_TILING}

    # Step 2: 執行 2_split_eval_results.py (評估結果拆分為多包獨立 Eval 報表)
    if [ "${RUN_SPLIT_EVAL}" -eq 1 ] && [ -f "2_split_eval_results.py" ]; then
        echo "---------------------------------------------------------------------------"
        echo " [Step 2/2] 執行 2_split_eval_results.py -> 拆分為 ${NUM_PACKS} 份獨立 Eval 報表"
        echo "---------------------------------------------------------------------------"
        local SPLIT_ARGS=()
        if [ -n "${SPLIT_COUNTS}" ]; then
            SPLIT_ARGS+=(--counts "${SPLIT_COUNTS}")
        elif [ -n "${SPLIT_RATIOS}" ]; then
            if [ "${SPLIT_RATIOS}" = "1:1:1" ] && [ "${NUM_PACKS}" -ne 3 ]; then
                local AUTO_RATIOS
                AUTO_RATIOS=$(printf "1:%.0s" $(seq 1 "${NUM_PACKS}") | sed 's/:$//')
                SPLIT_ARGS+=(--ratios "${AUTO_RATIOS}")
            else
                SPLIT_ARGS+=(--ratios "${SPLIT_RATIOS}")
            fi
        fi

        python 2_split_eval_results.py \
            --src_dir "${SAVE_BASE_DIR}" \
            --num_packs "${NUM_PACKS}" \
            --split_order "${SPLIT_ORDER}" \
            --sigma_k "${SIGMA_K}" \
            --auto_percentile "${AUTO_PERCENTILE}" \
            --min_threshold "${MIN_THRESHOLD}" \
            --threshold "${THRESHOLD}" \
            --top_k "${TOP_K}" \
            "${SPLIT_ARGS[@]}" \
            ${RECOMPUTE_PACK_THRESH} ${USE_SYMLINK}
    fi

    # 終端機列印該類別統計總結 (計算 Recall / FPR / 分數分佈)，並 Append 寫入多資料集總表
    python - "${SAVE_BASE_DIR}" "${CAT_NAME}" "${UNIQUE_TAG}" "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_CSV}" << 'EOF'
import os
import sys
import re
import csv
import glob
import numpy as np

save_base, cat_name, unique_tag, batch_txt, batch_csv = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)

all_csvs = sorted(glob.glob(os.path.join(save_base, "*", "escape_ranking_report.csv")))
csv_files = [p for p in all_csvs if not os.path.basename(os.path.dirname(p)).startswith("eval_pack_")]
if not csv_files:
    sys.exit(0)

latest_csv = csv_files[-1]
infer_dir = os.path.dirname(latest_csv)
latest_txt = latest_csv.replace(".csv", ".txt")

eff_thresh = 0.25
score_mode_str = "FILTERED_ROBUST"
if os.path.exists(latest_txt):
    with open(latest_txt, "r", encoding="utf-8") as tf:
        txt_head = tf.read(1500)
        m = re.search(r"Effective Threshold:\s*([0-9.]+)", txt_head)
        if m:
            eff_thresh = float(m.group(1))
        m_sm = re.search(r"Score Mode:\s*([A-Z_]+)", txt_head)
        if m_sm:
            score_mode_str = m_sm.group(1).strip()

records = []
with open(latest_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

total = len(records)
if total == 0:
    sys.exit(0)

normal_recs = [
    r for r in records if not os.path.basename(r["Original_Filename"]).startswith("escape_")
]
escape_defects = [
    r for r in records if os.path.basename(r["Original_Filename"]).startswith("escape_")
]

target_normal = normal_recs if normal_recs else records
norm_scores = [float(r["Max_Score"]) for r in target_normal]

max_s = float(np.max(norm_scores))
p99_s = float(np.percentile(norm_scores, 99))
p95_s = float(np.percentile(norm_scores, 95))
median_s = float(np.median(norm_scores))
mean_s = float(np.mean(norm_scores))
std_s = float(np.std(norm_scores))
m1std = mean_s + 1.0 * std_s
m2std = mean_s + 2.0 * std_s
m3std = mean_s + 3.0 * std_s

fp_normal = [r for r in target_normal if "ANOMALY" in r["Status"]]
caught_escape = [r for r in escape_defects if "ANOMALY" in r["Status"]]
fpr_pct = (len(fp_normal) / len(target_normal) * 100.0) if target_normal else 0.0
recall_pct = (len(caught_escape) / len(escape_defects) * 100.0) if escape_defects else 0.0

summary_block = [
    "=" * 85,
    f" 🎯 [{unique_tag.upper()} (MuSc-V2) 廠外公開資料集測試總結 — 分數分佈與 Threshold 參考 ({score_mode_str})]",
    "-" * 85,
    f"  📂 推論紀錄資料夾             : {os.path.abspath(infer_dir)}",
    f"  📄 最新排名報表位置           : {os.path.abspath(latest_csv)}",
    f"  🖼️ 盲池圖片結構               : 總計 {total} 張 (正常圖: {len(target_normal)} 張 | 隱藏Escape瑕疵: {len(escape_defects)} 張)",
    f"  🎚️ 本次動態對齊後有效門檻      : {eff_thresh:.4f}",
]
if escape_defects:
    summary_block.append(
        f"  🎯 抓漏與誤報表現             : 成功抓出瑕疵 {len(caught_escape)}/{len(escape_defects)} 張 (Recall={recall_pct:.1f}%) | 正常圖誤報 {len(fp_normal)}/{len(target_normal)} 張 (FPR={fpr_pct:.1f}%)"
    )
else:
    summary_block.append(
        f"  🎯 純正常池誤報表現 (FPR)     : 正常圖誤報 {len(fp_normal)}/{len(target_normal)} 張 (FPR={fpr_pct:.2f}%)"
    )
summary_block.extend([
    "-" * 85,
    f"  📊 正常圖分數分佈             : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}",
    f"  📐 正常圖門檻參考值           : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}",
])
if escape_defects:
    esc_scores = [float(r["Max_Score"]) for r in escape_defects]
    summary_block.append(
        f"  🔍 隱藏 Escape 瑕疵圖分數分佈  : Min={float(np.min(esc_scores)):.4f} | Median={float(np.median(esc_scores)):.4f} | Max={float(np.max(esc_scores)):.4f}"
    )
summary_block.append("=" * 85 + "\n")

full_text = "\n".join(summary_block)
print("\n" + full_text)

with open(batch_txt, "a", encoding="utf-8") as bf:
    bf.write(full_text + "\n")

with open(batch_csv, "a", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow([
        unique_tag,
        cat_name,
        total,
        len(target_normal),
        len(escape_defects),
        f"{eff_thresh:.4f}",
        f"{recall_pct:.2f}",
        f"{fpr_pct:.2f}",
        f"{mean_s:.4f}",
        f"{std_s:.4f}",
        os.path.abspath(infer_dir),
        os.path.abspath(latest_csv),
    ])
EOF
}

for cat_path in "${TARGET_CAT_DIRS[@]}"; do
    run_single_public_category "${cat_path}"
done

cp -f "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_LATEST}"
echo ""
echo "=========================================================================================================="
echo " ✅ 全部 ${#TARGET_CAT_DIRS[@]} 個資料集 (MuSc-V2 Zero-Shot) 執行完畢！所有紀錄皆已獨立保存："
echo "    📄 本次多資料集完整彙整報表 (TXT) : ${BATCH_SUMMARY_TXT}"
echo "    📄 最新多資料集完整彙整報表 (TXT) : ${BATCH_SUMMARY_LATEST}"
echo "    📊 最新多資料集完整彙整報表 (CSV) : ${BATCH_SUMMARY_CSV}"
echo "=========================================================================================================="