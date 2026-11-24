#!/usr/bin/env bash
# 檔案位置: MuSc-V2/run_inside_muscv2.sh
# 用途: 公司內部實機 (Fab Inside) 完全無 GT 盲池 MuSc-V2 Zero-Shot 一鍵測試腳本 (含自動 RAM/VRAM 背景監控與每資料夾獨立 Mask 設定載入)
# 用法:
# 0) 先花 2 秒測試並儲存該資料夾專屬 Mask 設定檔:
#    bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 0.02
# 1) 動態門檻 (自動 P95 + 穩健統計):
#    bash run_inside_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status
# 2) 多資料夾批次執行 (各資料夾會自動載入各自對應的 mask_<UNIQUE_TAG>.env):
#    bash run_inside_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status /workspace/dataset_use/W642/TMQB28/other_status
# 3) 固定門檻 (例如 0.32):
#    bash run_inside_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 0.32
# 執行期間若想即時偷看資源變化，可隨時執行:
#    tail -f wafer_escape_results/latest_mem_monitor_inside.log
set -e

# ===================== [可自由修改的參數區] =====================
OUTPUT_ROOT="./wafer_escape_results"
DATASETS_DIR="./datasets"
DINO_CKPT_PATH="auto" # 自動從 ~/.cache/torch/hub 或 ./hub_cache 離線搜尋 DINOv2/v3 權重
BACKBONE_NAME="dinov2_vitl14"
ENABLE_MEM_MONITOR=1  # 1=自動背景每秒記錄 RAM & GPU VRAM 至 log；0=關閉

CLI_ARGS=("$@")
CUSTOM_THRESH_PASSED=0

# 若最後一個參數是純數字或小數 (例如 0.25, .32, 1)，自動提取為 THRESHOLD 並切換為 fixed 模式
if [ ${#CLI_ARGS[@]} -ge 1 ]; then
  LAST_IDX=$((${#CLI_ARGS[@]} - 1))
  LAST_ARG="${CLI_ARGS[$LAST_IDX]}"
  if [[ "${LAST_ARG}" =~ ^[0-9]+(\.[0-9]+)?$ || "${LAST_ARG}" =~ ^\.[0-9]+$ ]]; then
    THRESHOLD="${LAST_ARG}"
    CUSTOM_THRESH_PASSED=1
    unset 'CLI_ARGS[$LAST_IDX]'
  fi
fi

# 剩餘的參數即為盲池資料夾清單；若未傳入任何資料夾，則使用預設路徑
if [ ${#CLI_ARGS[@]} -ge 1 ]; then
  TARGET_POOL_DIRS=("${CLI_ARGS[@]}")
else
  TARGET_POOL_DIRS=(
    "${DATASETS_DIR}/fab_wafer_pool"
  )
fi

# --- 階段執行控制開關 ---
FORCE_INFER="" # 若想強制重切測試塊並重跑推論，設為 "--force_infer"

# --- MuSc-V2 Zero-Shot 互評超參數 ---
R_LIST="1,3,5"
FEATURE_LAYERS="" # 留空自動配置 (24層ViT-L預設 5,11,17,23；12層ViT-B預設 2,5,8,11)
MUTUAL_RATIO=0.30
RSCON_WEIGHT=0.15
MAX_PEER_TILES=400

# --- 智慧解析度防呆切塊參數 ---
MIN_TILING_RES=1536 # 大圖防呆門檻：僅當長邊 >= 1536px (如 4096x4096/5000x5000 晶圓) 才切塊
TILE_SIZE=512
TILE_OVERLAP=64
INFER_BATCH_SIZE=64
BG_STD_THRESH=1.5
DISABLE_TILING="" # 若想強制關閉切塊退回整張模式，設為 "--disable_tiling"

# --- 動態門檻與四合一防誤報超參數 ---
if [ "${CUSTOM_THRESH_PASSED}" -eq 1 ]; then
  THRESH_MODE="auto"
else
  THRESH_MODE="${THRESH_MODE:-auto}" # 未指定固定數值時預設使用 hybrid 動態門檻 (亦可改為 fixed / mean_std / percentile)
fi
SIGMA_K=3.5
THRESHOLD="${THRESHOLD:-0.0}"
MIN_THRESHOLD="${MIN_THRESHOLD:-0}"
AUTO_PERCENTILE=95.0
TOP_K=0
CORE_RATIO=0.55
RAW_SCORE_MODE="" # 若想關閉防誤報濾波看原始單點最大值，設為 "--raw_score_mode"

WAFER_EDGE_MARGIN=0.04
BORDER_RATIO=0.02
SPECKLE_FILTER_RATIO=0.008
MIN_AREA_RATIO=0.00012

# --- 推論前圓形 Mask 遮罩預設參數 (找不到專屬設定檔或關閉自動載入時使用) ---
AUTO_LOAD_MASK_CFG=1    # 1=自動讀取 ./wafer_escape_results/mask_configs/mask_<UNIQUE_TAG>.env；0=關閉自動讀取
WAFER_EDGE_MODE="auto"  # 可選: auto, on, off
PRE_MASK=""             # 啟用設為 "--pre_mask"；關閉留空 ""
PRE_MASK_MARGIN=0.02    # 圓形 Mask 半徑內縮比例 (例如 0.02=內縮2%，負值為外擴)
PRE_MASK_FILL=0         # 背景填補像素值 (預設 0=純黑)
MASK_CX=-1              # 手動圓心 X (-1=自動偵測；0~1=寬度比例；>1=像素)
MASK_CY=-1              # 手動圓心 Y (-1=自動偵測；0~1=高度比例；>1=像素)
MASK_R=-1               # 手動半徑 R (-1=自動偵測；0~1=短邊比例；>1=像素)

# ===============================================================

BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${OUTPUT_ROOT}"
BATCH_SUMMARY_TXT="${OUTPUT_ROOT}/inside_muscv2_multi_pool_summary_${BATCH_TIMESTAMP}.txt"
BATCH_SUMMARY_LATEST="${OUTPUT_ROOT}/latest_inside_muscv2_summary.txt"
BATCH_SUMMARY_CSV="${OUTPUT_ROOT}/latest_inside_muscv2_summary.csv"
MEM_MONITOR_LOG="${OUTPUT_ROOT}/mem_monitor_inside_${BATCH_TIMESTAMP}.log"
MEM_MONITOR_LATEST="${OUTPUT_ROOT}/latest_mem_monitor_inside.log"
MEM_MONITOR_PID=""

start_mem_monitor() {
  if [ "${ENABLE_MEM_MONITOR}" -ne 1 ]; then
    return
  fi
  echo "=== MuSc-V2 Inside Fab System RAM & GPU VRAM Monitor Started at $(date '+%Y-%m-%d %H:%M:%S') ===" > "${MEM_MONITOR_LOG}"
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
    python - "${MEM_MONITOR_LOG}" << 'EOF'
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
        print(f" ├─ 系統 RAM 最高已用 : {max(ram_used)} MiB ({max(ram_used)/1024:.2f} GiB) | 最低剩餘可用 (Min Avail): {min(ram_avail)} MiB ({min(ram_avail)/1024:.2f} GiB)")
    if vram_used:
        print(f" └─ GPU VRAM 最高峰值 : {max(vram_used)} / {vram_tot} MiB ({max(vram_used)/1024:.2f} GiB, {max(vram_used)/max(1,vram_tot)*100:.1f}%)")
EOF
  fi
}

trap cleanup_mem_monitor EXIT INT TERM
start_mem_monitor

echo "==========================================================================================================" > "${BATCH_SUMMARY_TXT}"
echo " 🏭 [MuSc-V2 (Zero-Shot) 公司內部無 GT 盲池批次推論總覽報表] 啟動時間: ${BATCH_TIMESTAMP}" >> "${BATCH_SUMMARY_TXT}"
echo "==========================================================================================================" >> "${BATCH_SUMMARY_TXT}"
echo "Pool_Key,Category,Total_Imgs,Flagged_Anomalies,Anomaly_Rate_Pct,Eff_Threshold,Core_Normal_Med,Core_Normal_Mean,Core_Normal_Std,Infer_Dir,Report_CSV" > "${BATCH_SUMMARY_CSV}"

run_single_inside_muscv2_pool() {
  local POOL_DIR="$1"
  if [ ! -d "${POOL_DIR}" ]; then
    echo "❌ [路徑錯誤] 找不到公司內部測試盲池資料夾: ${POOL_DIR}"
    exit 1
  fi

  # 最多往上抓 3 層，遇到 datasets/dataset_use/workspace 停止
  local ABS_POOL_DIR
  ABS_POOL_DIR=$(cd "${POOL_DIR}" && pwd)
  local L1
  L1=$(basename "${ABS_POOL_DIR}")
  local P1
  P1=$(dirname "${ABS_POOL_DIR}")
  local L2
  L2=$(basename "${P1}")
  local P2
  P2=$(dirname "${P1}")
  local L3
  L3=$(basename "${P2}")

  local CAT_NAME="${L1// /_}"
  local UNIQUE_TAG="${L1}"
  if [[ -n "${L2}" && "${L2}" != "datasets" && "${L2}" != "dataset_use" && "${L2}" != "workspace" && "${L2}" != "." && "${L2}" != "/" ]]; then
    UNIQUE_TAG="${L2}_${UNIQUE_TAG}"
    if [[ -n "${L3}" && "${L3}" != "datasets" && "${L3}" != "dataset_use" && "${L3}" != "workspace" && "${L3}" != "." && "${L3}" != "/" ]]; then
      UNIQUE_TAG="${L3}_${UNIQUE_TAG}"
    fi
  fi
  UNIQUE_TAG="${UNIQUE_TAG//-/_}"
  UNIQUE_TAG="${UNIQUE_TAG// /_}"

  local DATA_NAME="fab_inside_muscv2_${UNIQUE_TAG}"
  local SAVE_BASE_DIR="${OUTPUT_ROOT}/inside_no_gt_muscv2_${UNIQUE_TAG}"

  # 每個資料夾獨立初始化 Mask 變數，避免多資料夾批次執行時互相污染
  local CUR_WAFER_EDGE_MODE="${WAFER_EDGE_MODE}"
  local CUR_PRE_MASK="${PRE_MASK}"
  local CUR_PRE_MASK_MARGIN="${PRE_MASK_MARGIN}"
  local CUR_PRE_MASK_FILL="${PRE_MASK_FILL}"
  local CUR_MASK_CX="${MASK_CX}"
  local CUR_MASK_CY="${MASK_CY}"
  local CUR_MASK_R="${MASK_R}"

  local MASK_CFG_FILE="${OUTPUT_ROOT}/mask_configs/mask_${UNIQUE_TAG}.env"
  local MASK_SOURCE_MSG="預設參數 (未找到 ${MASK_CFG_FILE})"
  if [ "${AUTO_LOAD_MASK_CFG}" -eq 1 ] && [ -f "${MASK_CFG_FILE}" ]; then
    source "${MASK_CFG_FILE}"
    MASK_SOURCE_MSG="專屬設定檔 (${MASK_CFG_FILE})"
  elif [ "${AUTO_LOAD_MASK_CFG}" -ne 1 ]; then
    MASK_SOURCE_MSG="腳本手動參數 (AUTO_LOAD_MASK_CFG=0)"
  fi

  echo ""
  echo "==========================================================================="
  echo " 🚀 [公司內部無 GT 實機測試啟動 (MuSc-V2 Zero-Shot)] 盲池路徑: ${POOL_DIR}"
  echo " CATEGORY   : ${CAT_NAME} | UNIQUE_TAG: ${UNIQUE_TAG}"
  echo " THRESH_MODE: ${THRESH_MODE} (THRESHOLD=${THRESHOLD}) | 獨立輸出目錄: ${SAVE_BASE_DIR}"
  echo " MASK 來源  : ${MASK_SOURCE_MSG}"
  echo " MASK 參數  : mode=${CUR_WAFER_EDGE_MODE} | pre_mask=${CUR_PRE_MASK:-off} | margin=${CUR_PRE_MASK_MARGIN} | fill=${CUR_PRE_MASK_FILL} | cx=${CUR_MASK_CX}, cy=${CUR_MASK_CY}, r=${CUR_MASK_R}"
  echo "==========================================================================="

  echo "---------------------------------------------------------------------------"
  echo " [推論執行] 執行 1_infer_muscv2_no_gt.py -> 智慧切塊、Zero-Shot 互評與防誤報紅框定位"
  echo "---------------------------------------------------------------------------"
  local LAYER_ARG=()
  if [ -n "${FEATURE_LAYERS}" ]; then
    LAYER_ARG=(--feature_layers "${FEATURE_LAYERS}")
  fi

  python 1_infer_muscv2_no_gt.py \
    --pool_dir "${POOL_DIR}" \
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
    --wafer_edge_mode "${CUR_WAFER_EDGE_MODE}" \
    --wafer_edge_margin "${WAFER_EDGE_MARGIN}" \
    --speckle_filter_ratio "${SPECKLE_FILTER_RATIO}" \
    --min_area_ratio "${MIN_AREA_RATIO}" \
    --min_tiling_res "${MIN_TILING_RES}" \
    --tile_size "${TILE_SIZE}" \
    --tile_overlap "${TILE_OVERLAP}" \
    --infer_batch_size "${INFER_BATCH_SIZE}" \
    --bg_std_thresh "${BG_STD_THRESH}" \
    --pre_mask_margin "${CUR_PRE_MASK_MARGIN}" \
    --pre_mask_fill "${CUR_PRE_MASK_FILL}" \
    --mask_cx "${CUR_MASK_CX}" \
    --mask_cy "${CUR_MASK_CY}" \
    --mask_r "${CUR_MASK_R}" \
    ${CUR_PRE_MASK} ${FORCE_INFER} ${RAW_SCORE_MODE} ${DISABLE_TILING}

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

flagged_anomalies = [r for r in records if "ANOMALY" in str(r["Status"])]
passed_normals = [r for r in records if "ANOMALY" not in str(r["Status"])]

target_normal = passed_normals if passed_normals else records
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

anom_rate_pct = (len(flagged_anomalies) / total * 100.0) if total else 0.0

summary_block = [
    "=" * 90,
    f" 🏭 [{unique_tag.upper()} 公司內部無 GT 盲池測試總結 (MuSc-V2 Zero-Shot) — 分數分佈與門檻參考 ({score_mode_str})]",
    "-" * 90,
    f" 📂 推論紀錄資料夾 : {os.path.abspath(infer_dir)}",
    f" 📄 最新排名報表位置 : {os.path.abspath(latest_csv)}",
    f" 🖼️ 盲池待測圖片總數 : {total} 張 (Training-Free Zero-Shot，無需 Golden 訓練集)",
    f" 🎚️ 本次有效判定門檻 : {eff_thresh:.4f}",
    f" 🚨 無GT異常檢出統計 : 抓出異常並畫框 {len(flagged_anomalies)}/{total} 張 (檢出率 Anomaly Rate={anom_rate_pct:.2f}%) | 正常略過 {len(passed_normals)} 張",
    "-" * 90,
    f" 📊 盲池正常核心分數分佈 : Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}",
    f" 📐 盲池正常核心門檻參考值 : 3-Sigma(Shewhart 1931)={_th_3sigma:.4f} | Mod-Z 3.5(Iglewicz 1993)={_th_modz:.4f}",
    f" 🎯 兩者較小值參考 : [{_low_name}] = {_low_th:.4f} (切出 {_low_cnt}/{_n_all} 張, {_low_cnt/max(1,_n_all)*100.0:.2f}%)",
    f" 📍 兩者較高值切點 : [{_high_name}] = {_high_th:.4f} -> {_high_cut_str}",
]
if flagged_anomalies:
    anom_scores = [float(r["Max_Score"]) for r in flagged_anomalies]
    top_names = [f"{r['Original_Filename']}({float(r['Max_Score']):.3f})" for r in flagged_anomalies[:5]]
    summary_block.append(
        f" 🔍 檢出異常圖分數分佈 : Min={float(np.min(anom_scores)):.4f} | Median={float(np.median(anom_scores)):.4f} | Max={float(np.max(anom_scores)):.4f}"
    )
    summary_block.append(
        f" 🏆 Top-5 最高分異常樣本 : {', '.join(top_names)}"
    )
summary_block.append("=" * 90 + "\n")

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
        len(flagged_anomalies),
        f"{anom_rate_pct:.2f}",
        f"{eff_thresh:.4f}",
        f"{median_s:.4f}",
        f"{mean_s:.4f}",
        f"{std_s:.4f}",
        os.path.abspath(infer_dir),
        os.path.abspath(latest_csv),
    ])
EOF
}

for pool_path in "${TARGET_POOL_DIRS[@]}"; do
  run_single_inside_muscv2_pool "${pool_path}"
done

cp -f "${BATCH_SUMMARY_TXT}" "${BATCH_SUMMARY_LATEST}"
echo ""
echo "=========================================================================================================="
echo " ✅ 公司內部全部 ${#TARGET_POOL_DIRS[@]} 個無 GT 測試盲池 (MuSc-V2 Zero-Shot) 執行完畢！所有紀錄與紅框圖皆已獨立保存："
echo " 📄 本次完整彙整報表 (TXT) : ${BATCH_SUMMARY_TXT}"
echo " 📄 最新完整彙整報表 (TXT) : ${BATCH_SUMMARY_LATEST}"
echo " 📊 最新完整彙整報表 (CSV) : ${BATCH_SUMMARY_CSV}"
echo " 📈 系統 RAM / VRAM 資源監控日誌 : ${MEM_MONITOR_LOG}"
echo "=========================================================================================================="