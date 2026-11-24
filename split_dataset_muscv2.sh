#!/usr/bin/env bash
# 檔案位置: MuSc-V2/split_dataset_muscv2.sh
# 用途: 將大型晶圓資料夾以「軟連結 + 隨機均分」拆成每包 <= 160 張（不佔硬碟空間），
#       並自動將原資料夾的 mask_<UNIQUE_TAG>.env 繼承複製給每個 part_XX 供 run_inside_muscv2.sh 自動載入。
# 用法:
#   bash split_dataset_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status
#   bash split_dataset_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 150
set -e

SRC_DIR="$1"
MAX_PER_BATCH="${2:-160}"
SPLIT_ROOT="./datasets"
MASK_CFG_DIR="./wafer_escape_results/mask_configs"

if [ -z "${SRC_DIR}" ] || [ ! -d "${SRC_DIR}" ]; then
  echo "❌ [路徑錯誤] 找不到資料夾: ${SRC_DIR}"
  echo "   用法: bash split_dataset_muscv2.sh <原始晶圓資料夾路徑> [每包上限張數, 預設160]"
  exit 1
fi

# 1. 使用與 test_mask_muscv2.sh / run_inside_muscv2.sh 完全一致的 3 層 UNIQUE_TAG 規則
ABS_POOL_DIR=$(cd "${SRC_DIR}" && pwd)
L1=$(basename "${ABS_POOL_DIR}")
P1=$(dirname "${ABS_POOL_DIR}")
L2=$(basename "${P1}")
P2=$(dirname "${P1}")
L3=$(basename "${P2}")

ORIG_TAG="${L1}"
if [[ -n "${L2}" && "${L2}" != "datasets" && "${L2}" != "dataset_use" && "${L2}" != "workspace" && "${L2}" != "." && "${L2}" != "/" ]]; then
  ORIG_TAG="${L2}_${ORIG_TAG}"
  if [[ -n "${L3}" && "${L3}" != "datasets" && "${L3}" != "dataset_use" && "${L3}" != "workspace" && "${L3}" != "." && "${L3}" != "/" ]]; then
    ORIG_TAG="${L3}_${ORIG_TAG}"
  fi
fi
ORIG_TAG="${ORIG_TAG//-/_}"
ORIG_TAG="${ORIG_TAG// /_}"

# 放在 ./datasets/split_<ORIG_TAG>/part_XX，讓 run_inside_muscv2.sh 算出的 UNIQUE_TAG 精準對齊 split_<ORIG_TAG>_part_XX
DST_DIR="${SPLIT_ROOT}/split_${ORIG_TAG}"
ORIG_MASK_CFG="${MASK_CFG_DIR}/mask_${ORIG_TAG}.env"

# 2. 掃描圖片並隨機打亂
mapfile -t FILES < <(
  find "${ABS_POOL_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) | shuf
)

TOTAL=${#FILES[@]}
if [ "${TOTAL}" -eq 0 ]; then
  echo "❌ [錯誤] 在 ${SRC_DIR} 找不到任何圖片！"
  exit 1
fi

NUM_PARTS=$(( (TOTAL + MAX_PER_BATCH - 1) / MAX_PER_BATCH ))

echo "==========================================================================="
echo " 📦 [MuSc-V2 資料集自動均勻分包 + Mask 繼承]"
echo " 原始路徑 : ${ABS_POOL_DIR}"
echo " 原始 TAG : ${ORIG_TAG}"
echo " 總張數   : ${TOTAL} 張 -> 均分為 ${NUM_PARTS} 包（每包約 $((TOTAL / NUM_PARTS)) 張，上限 ${MAX_PER_BATCH} 張）"
echo " 輸出目錄 : ${DST_DIR}"
echo "==========================================================================="

rm -rf "${DST_DIR}"
mkdir -p "${DST_DIR}"

# 3. 輪流發牌建立軟連結（保證各包最多只差 1 張）
for i in "${!FILES[@]}"; do
  part=$(printf "part_%02d" $(( i % NUM_PARTS + 1 )))
  mkdir -p "${DST_DIR}/${part}"
  ln -s "${FILES[$i]}" "${DST_DIR}/${part}/$(basename "${FILES[$i]}")"
done

# 4. 自動繼承 Mask 設定檔給每一個 part_XX
if [ -f "${ORIG_MASK_CFG}" ]; then
  for p_idx in $(seq -f "%02g" 1 "${NUM_PARTS}"); do
    PART_TAG="split_${ORIG_TAG}_part_${p_idx}"
    cp -f "${ORIG_MASK_CFG}" "${MASK_CFG_DIR}/mask_${PART_TAG}.env"
  done
  echo "💾 [Mask 繼承成功] 已將 ${ORIG_MASK_CFG} 自動複製給全部 ${NUM_PARTS} 個分包。"
else
  echo "⚠️  [提醒] 找不到原始 Mask 設定檔: ${ORIG_MASK_CFG}"
  echo "    建議先跑一次: bash test_mask_muscv2.sh \"${SRC_DIR}\" 再重新執行本分包腳本。"
fi

echo ""
echo "📊 [分包結果確認]"
for d in "${DST_DIR}"/part_*; do
  echo "   ├─ $(basename "$d") : $(ls -1 "$d" | wc -l) 張"
done

echo ""
echo "🚀 [下一步：直接複製以下指令開始排隊推論]"
echo "   nohup bash run_inside_muscv2.sh ${DST_DIR}/part_* > run_${ORIG_TAG}_split.log 2>&1 &"
echo "   tail -f run_${ORIG_TAG}_split.log"