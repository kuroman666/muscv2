#!/usr/bin/env bash
# 檔案位置: MuSc-V2/run_outside_muscv2.sh
# 用途: MuSc-V2 廠外公開資料集 (MVTec AD / MVTec AD 2 / Mura / AVMap / YDFID-2) Training-Free & Zero-Shot 無 GT 模擬抓漏與拆分評估腳本
# 1. 測試 YDFID-2 單一類別（預設 CL1）： bash run_outside_muscv2.sh
# 2. 測試 YDFID-2 全部 19 個類別： bash run_outside_muscv2.sh ./data/YDFID-2
#                               nohup ./run_outside_muscv2.sh ./data/YDFID-2 > outside_test_YDFID-2.log 2>&1 &
#                               接著用正確檔名即時查看進度：tail -f outside_test_YDFID-2.log
# 3. 跑舊資料集（MVTec / MVTec-2 / Mura / AVMap）（完全相容）： bash run_outside_muscv2.sh ./data/mvtec/capsule ./data/Mura2_wo_bg_w_pb
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./output"
DATASETS_DIR="./data"
DINO_CKPT_PATH="auto"                 # 自動從 ~/.cache/torch/hub 或 ./hub_cache 搜尋 DINOv2/v3 權重
BACKBONE_NAME="dinov2_vitl14"
ENABLE_MEM_MONITOR=1                  # 1=自動背景每秒記錄 RAM & GPU VRAM 至 log；0=關閉

if [ $# -ge 1 ]; then
    TARGET_CAT_DIRS=("$@")
else
    TARGET_CAT_DIRS=(
        # --- 舊資料集範例 (完全相容) ---
        "${DATASETS_DIR}/mvtec/bottle"
        "${DATASETS_DIR}/mvtec/cable"
        "${DATASETS_DIR}/mvtec/capsule"
        "${DATASETS_DIR}/mvtec/grid"
        "${DATASETS_DIR}/mvtec/tile"
        "${DATASETS_DIR}/mvtec/transistor"
        "${DATASETS_DIR}/mvtec-2/can"
        # "${DATASETS_DIR}/mini_AVMap2"
        # "${DATASETS_DIR}/Mura2_wo_bg_w_pb"
        # "${DATASETS_DIR}/Mura2_bg_mask"

        # --- 新資料集 YDFID-2 (可指定單一類別，或直接填入 YDFID-2 自動掃描全部 19 個類別) ---
        #"${DATASETS_DIR}/YDFID-2/CL1"
        # "${DATASETS_DIR}/YDFID-2"
    )
fi

POOL_MODE="escape"
DEFECT_LIMIT=8
REBUILD_POOL=""                       # 若想強制重建模擬盲池，設為 "--rebuild_pool"
FORCE_INFER=""                        # 若想強制重切測試塊並重跑推論，設為 "--force_infer"

# --- MuSc-V2 Zero-Shot 互評超參數 ---
R_LIST="1,3,5"
FEATURE_LAYERS=""                     # 留空自動配置 (24層ViT-L預設 5,11,17,23；12層ViT-B預設 2,5,8,11)
MUTUAL_RATIO=0.30
RSCON_WEIGHT=0.15
MAX_PEER_TILES=400

# --- 動態門檻與防誤報超參數 ---
THRESH_MODE="auto"
SIGMA_K=3.5
THRESHOLD=0
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
MIN_TILING_RES=1536                   # 大圖防呆門檻：僅當長邊 >= 1536px 才切塊；中低解析度公開集自動整張推論
TILE_SIZE=512
TILE_OVERLAP=64
INFER_BATCH_SIZE=16
BG_STD_THRESH=1.5
DISABLE_TILING=""                     # 若想強制關閉切塊，設為 "--disable_tiling"

# ===============================================================

BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${OUTPUT_ROOT}"
BATCH_SUMMARY_TXT="${OUTPUT_ROOT}/multi_dataset_batch_summary_muscv2_${BATCH_TIMESTAMP}.txt"
BATCH_SUMMARY_LATEST="${OUTPUT_ROOT}/latest_multi_dataset_summary_muscv2.txt"
BATCH_SUMMARY_CSV="${OUTPUT_ROOT}/latest_multi_dataset_summary_muscv2.csv"
MEM_MONITOR_LOG="${OUTPUT_ROOT}/mem_monitor_${BATCH_TIMESTAMP}.log"
MEM_MONITOR_LATEST="${OUTPUT_ROOT}/latest_mem_monitor.log"
MEM_MONITOR_PID=""

start_mem_monitor() {
    if [ "${ENABLE_MEM_MONITOR}" -ne 1 ]; then
        return
    fi
    echo "=== MuSc-V2 System RAM & GPU VRAM Monitor Started at $(date '+%Y-%m-%d %H:%M:%S') ===" > "${MEM_MONITOR_LOG}"
    (
        while true; do
            TS=$(date '+%Y-%m-%d %H:%M:%S')
            RAM_INFO=$(free -m | awk '/^Mem:/ {printf "RAM_Used=%d/%d MiB (Avail=%d MiB)", $3, $2, $7} /^Swap:/ {printf " | Swap=%d/%d MiB", $3, $2}')
            if command -v nvidia-smi >/dev/null 2>&1; then
                GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n 1 | awk -F", " '{printf "GPU_Util=%s%% | VRAM=%s/%s MiB (%.1f%%)", $1, $2, $3, ($2/$3)*100}')
            else
                GPU_INFO="GPU=N/A"
            fi
            echo "[${TS}] ${RAM_INFO} | ${GPU_INFO}" >> "${MEM_MONITOR_LOG}"
            sleep 1
        done
    ) &
    MEM_MONITOR_PID=$!
    echo "📊 [系統資源監控已背景啟動] PID=${MEM_MONITOR_PID} -> 即時紀錄檔: ${MEM_MONITOR_LOG}"
}

cleanup_mem_monitor() {
    if [ -n "${MEM_MONITOR_PID}" ] && kill -0 "${MEM_MONITOR_PID}" 2>/dev/null; then
        kill "${MEM_MONITOR_PID}" 2>/dev/null || true
        wait "${MEM_MONITOR_PID}" 2>/dev/null || true
    fi
    if [ -f "${MEM_MONITOR_LOG}" ]; then
        cp -f "${MEM_MONITOR_LOG}" "${MEM_MONITOR_LATEST}"
        python - "${MEM_MONITOR_LOG}" << 'PYEOF'
import re, sys
log_p = sys.argv[1]
ram_used, ram_avail, vram_used = [], [], []
vram_tot = 0
with open(log_p, "r", encoding="utf-8") as f:
    for line in f:
        m_r = re.search(r"RAM_Used=(\d+)/(\d+)\s*MiB\s*\(Avail=(\d+)\s*MiB\)", line)
        if m_r:
            ram_used.append(int(m_r.group(1)))
            ram_avail.append(int(m_r.group(3)))
        m_v = re.search(r"VRAM=(\d+)/(\d+)\s*MiB", line)
        if m_v:
            vram_used.append(int(m_v.group(1)))
            vram_tot = int(m_v.group(2))
if ram_used or vram_used:
    print("\n📈 [整趟執行資源峰值統計 (RAM & GPU VRAM Summary)]")
    if ram_used:
        print(f"   ├─ 系統 RAM 最高已用 : {max(ram_used)} MiB ({max(ram_used)/1024:.2f} GiB) | 最低剩餘可用 (Min Avail): {min(ram_avail)} MiB ({min(ram_avail)/1024:.2f} GiB)")
    if vram_used:
        print(f"   └─ GPU VRAM 最高峰值 : {max(vram_used)} / {vram_tot} MiB ({max(vram_used)/1024:.2f} GiB, {max(vram_used)/max(1,vram_tot)*100:.1f}%)")
PYEOF
    fi
}

trap cleanup_mem_monitor EXIT INT TERM
start_mem_monitor

echo "==========================================================================================================" > "${BATCH_SUMMARY_TXT}"
echo " 📊 [MuSc-V2 (Zero-Shot) 多資料集批次推論總覽報表] 啟動時間: ${BATCH_TIMESTAMP}" >> "${BATCH_SUMMARY_TXT}"
echo "==========================================================================================================" >> "${BATCH_SUMMARY_TXT}"
echo "Dataset_Key,Category,Total_Imgs,Normal_Imgs,Escape_Imgs,Eff_Threshold,I_AUROC,I_AP,F1_Max,Recall_Pct,FPR_Pct,Precision_Pct,F1_Cur_Pct,Normal_Mean,Normal_Std,Infer_Dir,Report_CSV" > "${BATCH_SUMMARY_CSV}"

prepare_public_sim_pool() {
    local SRC_CAT_DIR="$1"
    local DST_POOL_DIR="$2"
    local MODE="$3"
    local LIMIT="$4"
    local FORCE_REBUILD="$5"

    python - "${SRC_CAT_DIR}" "${DST_POOL_DIR}" "${MODE}" "${LIMIT}" "${FORCE_REBUILD}" << 'PYEOF'
import os
import sys
import stat
import shutil

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# 支援舊資料集 (good / ok / normal) 與新資料集 YDFID-2 (defect-free)
GOOD_FOLDER_NAMES = {"good", "defect-free", "defect_free", "ok", "normal"}
IGNORE_DIRS = {"masks", "masks2", "mask", "ground_truth", "groundtruth", "gt", "with_gt", "__macosx"}

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

def find_good_subdir(parent_dir):
    if not os.path.isdir(parent_dir):
        return None
    for d in sorted(os.listdir(parent_dir)):
        if d.lower() in GOOD_FOLDER_NAMES and os.path.isdir(os.path.join(parent_dir, d)):
            return os.path.join(parent_dir, d)
    return None

# 自動相容：
# 1) MVTec / MVTec-2: <src>/test/good/
# 2) YDFID-2: <src>/test/defect-free/ 與 <src>/test/defect/
# 3) Mura / AVMap: <src>/test/<sub_cat>/good/
test_dir = None
good_dir = None
for cand in ["test", "test_public", "val", "validation"]:
    cand_path = os.path.join(src_cat_dir, cand)
    if not os.path.isdir(cand_path):
        continue
    g = find_good_subdir(cand_path)
    if g is not None:
        test_dir = cand_path
        good_dir = g
        break
    sub_entries = [
        os.path.join(cand_path, d)
        for d in sorted(os.listdir(cand_path))
        if os.path.isdir(os.path.join(cand_path, d)) and d.lower() not in IGNORE_DIRS
    ]
    for sub_p in sub_entries:
        g2 = find_good_subdir(sub_p)
        if g2 is not None:
            test_dir = sub_p
            good_dir = g2
            break
    if test_dir is not None:
        break

if test_dir is None or good_dir is None:
    raise FileNotFoundError(
        f"在 {src_cat_dir} 下找不到包含 good/ 或 defect-free/ 的測試資料夾！"
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

good_count = 0
for root, dirs, files in os.walk(good_dir):
    dirs[:] = [d for d in sorted(dirs) if d.lower() not in IGNORE_DIRS]
    for f in sorted(files):
        if f.lower().endswith(IMG_EXTS) and not f.startswith("."):
            copy_unlocked(os.path.join(root, f), os.path.join(dst_pool_dir, f"clean_{f}"))
            good_count += 1

# 若 test 正常樣本少於 8 張，自動從 train/good 或 train/defect-free 補充正常樣本
if good_count < 8:
    rel_sub = os.path.relpath(test_dir, src_cat_dir)
    sub_cat_name = os.path.basename(test_dir) if os.path.dirname(rel_sub) else ""
    backup_parents = ["validation", "train"]
    if sub_cat_name:
        backup_parents = [f"validation/{sub_cat_name}", f"train/{sub_cat_name}"] + backup_parents
    for bp in backup_parents:
        b_dir = find_good_subdir(os.path.join(src_cat_dir, bp))
        if b_dir and os.path.isdir(b_dir):
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
        if sub.lower() in GOOD_FOLDER_NAMES or sub.lower() in IGNORE_DIRS:
            continue
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for root, dirs, files in os.walk(sub_dir):
            dirs[:] = [d for d in sorted(dirs) if d.lower() not in IGNORE_DIRS]
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS) and not f.startswith(".") and defect_count < defect_limit:
                    copy_unlocked(os.path.join(root, f), os.path.join(dst_pool_dir, f"escape_{sub}_{f}"))
                    defect_count += 1

print(
    f"[✓ 模擬無GT盲池建立完成] 來源: {test_dir} | 模式: {mode} | "
    f"共 {good_count + defect_count} 張圖 (正常: {good_count} 張, 隱藏瑕疵: {defect_count} 張) -> {os.path.abspath(dst_pool_dir)}"
)
PYEOF
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

    prepare_public_sim_pool "${CAT_DIR}" "${SIM_POOL_DIR}" "${POOL_MODE}" "${DEFECT_LIMIT}" "${REBUILD_POOL}"

    echo "---------------------------------------------------------------------------"
    echo " [推論執行] 執行 1_infer_muscv2_no_gt.py -> Zero-Shot 互評推理與防誤報紅框定位"
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

python - "${SAVE_BASE_DIR}" "${CAT_NAME}" "${UNIQUE_TAG}" "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_CSV}" << 'PYEOF'
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
        txt_head = tf.read(1800)
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
_all_eval_scores = np.array([float(r["Max_Score"]) for r in records], dtype=np.float64)
_mean_all = float(np.mean(_all_eval_scores))
_std_all = float(np.std(_all_eval_scores))
_th_3sigma = _mean_all + 3.0 * _std_all
_med_all = float(np.median(_all_eval_scores))
_mad_all = float(np.median(np.abs(_all_eval_scores - _med_all)))
_th_modz = _med_all + (3.5 / 0.6745) * _mad_all
if _th_3sigma <= _th_modz:
    _low_th, _low_name = _th_3sigma, "3-Sigma"
    _high_th, _high_name = _th_modz, "Mod-Z(3.5)"
else:
    _low_th, _low_name = _th_modz, "Mod-Z(3.5)"
    _high_th, _high_name = _th_3sigma, "3-Sigma"
_n_all = int(_all_eval_scores.size)
_low_cnt = int(np.sum(_all_eval_scores >= _low_th))
_high_cnt = int(np.sum(_all_eval_scores >= _high_th))
_high_pct = (_high_cnt / max(1, _n_all)) * 100.0
_high_p = float(np.mean(_all_eval_scores < _high_th) * 100.0)
_high_cut_str = (
    f"高於最高分({float(np.max(_all_eval_scores)):.4f})，切出 0 張"
    if _high_cnt == 0
    else f"切在 Rank {_high_cnt}/{_n_all} 之後 (共 {_high_cnt} 張, {_high_pct:.2f}%, 約 P{_high_p:.2f})"
)
m1std = mean_s + 1.0 * std_s
m2std = mean_s + 2.0 * std_s
m3std = mean_s + 3.0 * std_s

fp_normal = [r for r in target_normal if "ANOMALY" in r["Status"]]
caught_escape = [r for r in escape_defects if "ANOMALY" in r["Status"]]
fpr_pct = (len(fp_normal) / len(target_normal) * 100.0) if target_normal else 0.0
recall_pct = (len(caught_escape) / len(escape_defects) * 100.0) if escape_defects else 0.0

auroc_val = ap_val = f1_max_val = best_th_val = 0.0
prec_pct = f1_cur_pct = 0.0

if escape_defects and normal_recs:
    y_true = np.array([1 if os.path.basename(r["Original_Filename"]).startswith("escape_") else 0 for r in records], dtype=np.int32)
    y_score = np.array([float(r["Max_Score"]) for r in records], dtype=np.float64)
    n_pos, n_neg = int(np.sum(y_true == 1)), int(np.sum(y_true == 0))

    order = np.argsort(-y_score, kind="mergesort")
    yt_s, ys_s = y_true[order], y_score[order]
    ranks = np.zeros(len(ys_s), dtype=np.float64)
    i, n = 0, len(ys_s)
    while i < n:
        j = i + 1
        while j < n and ys_s[j] == ys_s[i]:
            j += 1
        ranks[i:j] = 0.5 * ((n - j + 1) + (n - i))
        i = j
    auroc_val = (float(np.sum(ranks[yt_s == 1])) - n_pos * (n_pos + 1) / 2.0) / max(1.0, float(n_pos * n_neg))

    tp_c, fp_c = np.cumsum(yt_s == 1), np.cumsum(yt_s == 0)
    t_idx = np.r_[np.where(np.diff(ys_s))[0], n - 1]
    tps, fps = tp_c[t_idx], fp_c[t_idx]
    precs = tps / np.maximum(1, tps + fps)
    recs = tps / max(1, n_pos)
    ap_val = float(np.sum(np.diff(np.r_[0.0, recs]) * precs))
    f1s = (2.0 * precs * recs) / np.maximum(1e-12, precs + recs)
    b_i = int(np.argmax(f1s))
    f1_max_val = float(f1s[b_i])
    best_th_val = float(ys_s[t_idx[b_i]])

    tp_cur = len(caught_escape)
    fp_cur = len(fp_normal)
    prec_pct = (tp_cur / max(1, tp_cur + fp_cur)) * 100.0
    f1_cur_pct = (2.0 * prec_pct * recall_pct) / max(1e-12, prec_pct + recall_pct)

summary_block = [
    "=" * 95,
    f" 🎯 [{unique_tag.upper()} (MuSc-V2) 廠外公開資料集測試總結 — 分數分佈與量化評估指標 ({score_mode_str})]",
    "-" * 95,
    f"  📂 推論紀錄資料夾             : {os.path.abspath(infer_dir)}",
    f"  📄 最新排名報表位置           : {os.path.abspath(latest_csv)}",
    f"  🖼️ 盲池圖片結構               : 總計 {total} 張 (正常圖: {len(target_normal)} 張 | 隱藏Escape瑕疵: {len(escape_defects)} 張)",
    f"  🎚️ 本次動態對齊後有效門檻      : {eff_thresh:.4f}",
]
if escape_defects:
    summary_block.extend([
        f"  🏆 標準量化評估指標 (Metrics) : I-AUROC={auroc_val:.4f} | I-AP={ap_val:.4f} | F1-max={f1_max_val:.4f} (Best_Th={best_th_val:.4f})",
        f"  🎯 抓漏與誤報表現 (Current)   : Recall={recall_pct:.2f}% ({len(caught_escape)}/{len(escape_defects)}) | FPR={fpr_pct:.2f}% ({len(fp_normal)}/{len(target_normal)}) | Precision={prec_pct:.2f}% | F1={f1_cur_pct:.2f}%",
    ])
else:
    summary_block.append(
        f"  🎯 純正常池誤報表現 (FPR)     : 正常圖誤報 {len(fp_normal)}/{len(target_normal)} 張 (FPR={fpr_pct:.2f}%)"
    )
summary_block.extend([
    "-" * 95,
    f"  📊 正常圖分數分佈             : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}",
    f"  📐 正常圖門檻參考值           :  3-Sigma(Shewhart 1931)={_th_3sigma:.4f} | Mod-Z 3.5(Iglewicz 1993)={_th_modz:.4f}",
    f"  🎯 兩者較小值參考         : [{_low_name}] = {_low_th:.4f} (切出 {_low_cnt}/{_n_all} 張, {_low_cnt/max(1,_n_all)*100.0:.2f}%)",
    f"  📍 兩者較高值切點         : [{_high_name}] = {_high_th:.4f} -> {_high_cut_str}",
])
if escape_defects:
    esc_scores = [float(r["Max_Score"]) for r in escape_defects]
    summary_block.append(
        f"  🔍 隱藏 Escape 瑕疵圖分數分佈  : Min={float(np.min(esc_scores)):.4f} | Median={float(np.median(esc_scores)):.4f} | Max={float(np.max(esc_scores)):.4f}"
    )
summary_block.append("=" * 95 + "\n")

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
        f"{auroc_val:.4f}",
        f"{ap_val:.4f}",
        f"{f1_max_val:.4f}",
        f"{recall_pct:.2f}",
        f"{fpr_pct:.2f}",
        f"{prec_pct:.2f}",
        f"{f1_cur_pct:.2f}",
        f"{mean_s:.4f}",
        f"{std_s:.4f}",
        os.path.abspath(infer_dir),
        os.path.abspath(latest_csv),
    ])
PYEOF
}

# 展開目標清單：若傳入的是整個資料集根目錄 (例如 ./data/YDFID-2 或 ./data/mvtec)，自動展開底下所有類別
EXPANDED_CAT_DIRS=()
for entry_path in "${TARGET_CAT_DIRS[@]}"; do
    if [ ! -d "${entry_path}" ]; then
        echo "❌ [路徑錯誤] 找不到資料夾: ${entry_path}"
        exit 1
    fi
    if [ -d "${entry_path}/test" ] || [ -d "${entry_path}/test_public" ] || [ -d "${entry_path}/validation" ]; then
        EXPANDED_CAT_DIRS+=("${entry_path}")
    else
        for sub_cat in $(find "${entry_path}" -mindepth 1 -maxdepth 1 -type d | sort); do
            if [ -d "${sub_cat}/test" ] || [ -d "${sub_cat}/test_public" ] || [ -d "${sub_cat}/validation" ]; then
                EXPANDED_CAT_DIRS+=("${sub_cat}")
            fi
        done
    fi
done

for cat_path in "${EXPANDED_CAT_DIRS[@]}"; do
    run_single_public_category "${cat_path}"
done

cp -f "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_LATEST}"
echo ""
echo "=========================================================================================================="
echo " ✅ 全部 ${#EXPANDED_CAT_DIRS[@]} 個類別 (MuSc-V2 Zero-Shot) 執行完畢！所有紀錄皆已獨立保存："
echo "    📄 本次多資料集完整彙整報表 (TXT) : ${BATCH_SUMMARY_TXT}"
echo "    📄 最新多資料集完整彙整報表 (TXT) : ${BATCH_SUMMARY_LATEST}"
echo "    📊 最新多資料集完整彙整報表 (CSV) : ${BATCH_SUMMARY_CSV}"
echo "    📈 系統 RAM / VRAM 資源監控日誌  : ${MEM_MONITOR_LOG}"
echo "=========================================================================================================="
