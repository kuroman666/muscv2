#!/usr/bin/env bash
# 檔案位置: MuSc-V2/test_mask_muscv2.sh
# 用途: 推論前秒級測試圓形 Mask 大小與圓心位置（不跑模型、不切塊），測完自動存成該資料夾專屬的 mask 設定檔供 run_inside_muscv2.sh 讀取
# 用法:
# 1) 使用預設自動偵測圓心 + 內縮 2% 測試:
#    bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status
# 2) 快速傳參調整 margin (第2個參數):
#    bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 0.03
# 3) 手動指定圓心與半徑 (支援 0~1 比例 或 >1 像素):
#    bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 0.0 0.5 0.5 0.46
#    bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 0.0 2510 2480 2320
set -e

# ===================== [可自由修改的測試參數區] =====================
POOL_DIR="${1:-./datasets/fab_wafer_pool}" # 待測晶圓資料夾路徑
PREVIEW_SAMPLES=5                          # 均勻抽樣張數 (涵蓋資料集前後段)
WAFER_EDGE_MODE="on"                       # 強制開啟圓形偵測 (on / auto)

PRE_MASK_MARGIN="${2:-0.02}" # 半徑內縮比例 (0.02=內縮2%；負值=外擴)
PRE_MASK_FILL=0              # 背景填補像素值 (0=純黑)
MASK_CX="${3:--1}"           # 手動圓心 X (-1=自動；0~1=寬度比例；>1=像素)
MASK_CY="${4:--1}"           # 手動圓心 Y (-1=自動；0~1=高度比例；>1=像素)
MASK_R="${5:--1}"            # 手動半徑 R (-1=自動；0~1=短邊比例；>1=像素)

OUT_DIR="./wafer_escape_results/mask_preview_test"
MASK_CFG_DIR="./wafer_escape_results/mask_configs"
# ===================================================================

if [ ! -d "${POOL_DIR}" ]; then
  echo "❌ [路徑錯誤] 找不到資料夾: ${POOL_DIR}"
  echo "   用法: bash test_mask_muscv2.sh <晶圓資料夾路徑> [MARGIN] [CX] [CY] [R]"
  exit 1
fi

# 計算與 run_inside_muscv2.sh 一致的 UNIQUE_TAG (最多往上抓 3 層，遇到 datasets/dataset_use/workspace 停止)
ABS_POOL_DIR=$(cd "${POOL_DIR}" && pwd)
L1=$(basename "${ABS_POOL_DIR}")
P1=$(dirname "${ABS_POOL_DIR}")
L2=$(basename "${P1}")
P2=$(dirname "${P1}")
L3=$(basename "${P2}")

CAT_NAME="${L1// /_}"
UNIQUE_TAG="${L1}"
if [[ -n "${L2}" && "${L2}" != "datasets" && "${L2}" != "dataset_use" && "${L2}" != "workspace" && "${L2}" != "." && "${L2}" != "/" ]]; then
  UNIQUE_TAG="${L2}_${UNIQUE_TAG}"
  if [[ -n "${L3}" && "${L3}" != "datasets" && "${L3}" != "dataset_use" && "${L3}" != "workspace" && "${L3}" != "." && "${L3}" != "/" ]]; then
    UNIQUE_TAG="${L3}_${UNIQUE_TAG}"
  fi
fi
UNIQUE_TAG="${UNIQUE_TAG//-/_}"
UNIQUE_TAG="${UNIQUE_TAG// /_}"

echo "==========================================================================="
echo " 🔍 [MuSc-V2 圓形 Mask 快速預覽測試]"
echo " 資料夾   : ${POOL_DIR} (UNIQUE_TAG: ${UNIQUE_TAG})"
echo " 抽樣張數 : ${PREVIEW_SAMPLES} 張 | 模式: ${WAFER_EDGE_MODE}"
echo " 參數     : margin=${PRE_MASK_MARGIN} | fill=${PRE_MASK_FILL} | cx=${MASK_CX}, cy=${MASK_CY}, r=${MASK_R}"
echo "==========================================================================="

python 1_infer_muscv2_no_gt.py \
  --pool_dir "${POOL_DIR}" \
  --out_dir "${OUT_DIR}" \
  --wafer_edge_mode "${WAFER_EDGE_MODE}" \
  --pre_mask_margin "${PRE_MASK_MARGIN}" \
  --pre_mask_fill "${PRE_MASK_FILL}" \
  --mask_cx "${MASK_CX}" \
  --mask_cy "${MASK_CY}" \
  --mask_r "${MASK_R}" \
  --preview_mask "${PREVIEW_SAMPLES}"

# ===================== [依資料夾獨立儲存 Mask 設定檔] =====================
MASK_CFG_FILE="${MASK_CFG_DIR}/mask_${UNIQUE_TAG}.env"
mkdir -p "${MASK_CFG_DIR}"

cat << EOF > "${MASK_CFG_FILE}"
# 由 test_mask_muscv2.sh 自動產生 ($(date '+%Y-%m-%d %H:%M:%S'))
# 測試來源資料夾: ${POOL_DIR} (UNIQUE_TAG: ${UNIQUE_TAG})
CUR_WAFER_EDGE_MODE="${WAFER_EDGE_MODE}"
CUR_PRE_MASK="--pre_mask"
CUR_PRE_MASK_MARGIN="${PRE_MASK_MARGIN}"
CUR_PRE_MASK_FILL="${PRE_MASK_FILL}"
CUR_MASK_CX="${MASK_CX}"
CUR_MASK_CY="${MASK_CY}"
CUR_MASK_R="${MASK_R}"
EOF

echo "💾 [已儲存專屬 Mask 設定檔] ${MASK_CFG_FILE}"
echo "   執行 run_inside_muscv2.sh 時將自動依 UNIQUE_TAG (${UNIQUE_TAG}) 載入此設定。"