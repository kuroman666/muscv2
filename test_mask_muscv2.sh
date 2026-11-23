#!/usr/bin/env bash
# 檔案位置: MuSc-V2/test_mask_muscv2.sh
# 用途: 推論前秒級測試圓形 Mask 大小與圓心位置（不跑模型、不切塊），測完自動報出 run_inside_muscv2.sh 參數設定
# 用法:
#   1) 使用預設自動偵測圓心 + 內縮 2% 測試:
#        bash test_mask_muscv2.sh ./datasets/fab_wafer_pool
#   2) 快速傳參調整 margin (第2個參數):
#        bash test_mask_muscv2.sh ./datasets/fab_wafer_pool 0.03
#   3) 手動指定圓心與半徑 (支援 0~1 比例 或 >1 像素):
#        bash test_mask_muscv2.sh ./datasets/fab_wafer_pool 0.0 0.5 0.5 0.46
#        bash test_mask_muscv2.sh ./datasets/fab_wafer_pool 0.0 2510 2480 2320
set -e

# ===================== [可自由修改的測試參數區] =====================
POOL_DIR="${1:-./datasets/fab_wafer_pool}"   # 待測晶圓資料夾路徑
PREVIEW_SAMPLES=5                            # 均勻抽樣張數 (涵蓋資料集前後段)
WAFER_EDGE_MODE="on"                         # 強制開啟圓形偵測 (on / auto)

PRE_MASK_MARGIN="${2:-0.02}"                 # 半徑內縮比例 (0.02=內縮2%；負值=外擴)
PRE_MASK_FILL=0                              # 背景填補像素值 (0=純黑)
MASK_CX="${3:--1}"                           # 手動圓心 X (-1=自動；0~1=寬度比例；>1=像素)
MASK_CY="${4:--1}"                           # 手動圓心 Y (-1=自動；0~1=高度比例；>1=像素)
MASK_R="${5:--1}"                            # 手動半徑 R (-1=自動；0~1=短邊比例；>1=像素)

OUT_DIR="./wafer_escape_results/mask_preview_test"
# ===================================================================

if [ ! -d "${POOL_DIR}" ]; then
    echo "❌ [路徑錯誤] 找不到資料夾: ${POOL_DIR}"
    echo "   用法: bash test_mask_muscv2.sh <晶圓資料夾路徑> [MARGIN] [CX] [CY] [R]"
    exit 1
fi

echo "==========================================================================="
echo " 🔍 [MuSc-V2 圓形 Mask 快速預覽測試]"
echo "    資料夾   : ${POOL_DIR}"
echo "    抽樣張數 : ${PREVIEW_SAMPLES} 張 | 模式: ${WAFER_EDGE_MODE}"
echo "    參數     : margin=${PRE_MASK_MARGIN} | fill=${PRE_MASK_FILL} | cx=${MASK_CX}, cy=${MASK_CY}, r=${MASK_R}"
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
