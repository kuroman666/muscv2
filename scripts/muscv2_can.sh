#!/bin/bash

# ==========================================
# 1. 在這裡集中設定你要測試的超參數
# ==========================================
DEVICE=0

# ====== [進公司需修改處 1：切換資料路徑與類別名稱] ======
# 【公司外 MVTec-2 測試設定】
DATA_PATH="./data/mvtec-2/can"
DATASET_NAME="mvtec_2"
# CLASS_NAME="test_public"

# 【進公司晶圓測試設定】(進公司時將上面三行註解，取消下面三行註解)
# DATA_PATH="./data/categorized_dataset/TMUN87"
# DATASET_NAME="mvtec_2"        # 固定維持 mvtec_2 不用改！
# CLASS_NAME="other_status"     # 對應 TMUN87 下的圖片資料夾名稱

# ====== [進公司需修改處 2：實驗超參數調整] ======
BACKBONE_NAME="dinov2_vitl14"
FEATURE_LAYERS="5 11 17 23"     # 維持原論文/標準四層設定，或依需求測 "11 17 23"
R_LIST="1 3 5"                  # 維持原論文標準 R_LIST="1 3 5"
# FEATURE_LAYERS="11 17 23"       # 進公司測 Wafer 建議用 "11 17 23" 避開細格線；測 can 可用 "5 11 17 23"
# R_LIST="3 5"                    # 進公司測 Wafer 建議用 "3 5"；測 can 可用 "1 3 5"
IMG_RESIZE=518
OUTPUT_DIR="./output/"

# 視覺化門檻 (現在已 100% 同步對齊 CSV 的 Anomaly_Score！)
# 例如設為 0.37，代表 CSV 中 Anomaly_Score >= 0.37 的圖片與像素才會被紅框標示出來
export THRESHOLD=0.37
export ALPHA=0.6


# ====== [進公司需修改處 3：晶圓專用前處理開關] ======
# ====== [晶圓專用資料前處理開關 (在 mvtec2.py 執行，不影響模型內部架構)] ======
# 【A. 公司外測 can 時，請用這組 (全關 = 原版 MuSc-V2)】：
export AUTO_ALIGN_WAFER=0
export PHOTOMETRIC_NORM=0
export FIX_90DEG_ROTATION=0
export WAFER_MASK_RATIO=0.0

# 【B. 進公司測整張 Wafer 時，請將上面四行註解，改取消註解下面四行】：
# export AUTO_ALIGN_WAFER=1     # 自動抓晶圓圓心置中、轉成正方形畫布、直徑統一為 90%
# export PHOTOMETRIC_NORM=1     # 晶圓內部光度正規化 (消除灰階色度差異，且不在綠圈內製造黑邊斷崖)
# export FIX_90DEG_ROTATION=1   # 自動與第一張晶圓對齊 0/90/180/270 度方向
# export WAFER_MASK_RATIO=0.84  # 晶圓直徑固定為 90%，此處設 0.84 確保綠圈 100% 落在晶圓內部安全區

mkdir -p "$OUTPUT_DIR"

DATASET_ID=$(basename "$DATA_PATH")
TIMESTAMP=$(date -u -d '+8 hours' '+%Y%m%d_%H%M%S')
READABLE_TIME=$(date -u -d '+8 hours' '+%Y-%m-%d %H:%M:%S')

export EXP_OUTPUT_DIR="${OUTPUT_DIR}${DATASET_NAME}/${BACKBONE_NAME}/imagesize${IMG_RESIZE}/${DATASET_ID}_${CLASS_NAME}_${TIMESTAMP}/"
mkdir -p "$EXP_OUTPUT_DIR"

LOG_FILE="${EXP_OUTPUT_DIR}${DATASET_ID}_${TIMESTAMP}_log.txt"

echo "========================================" > "$LOG_FILE"
echo "測試時間: ${READABLE_TIME} (台北時間)" >> "$LOG_FILE"
echo "輸出目錄: ${EXP_OUTPUT_DIR}" >> "$LOG_FILE"
echo "【本次實驗超參數】" >> "$LOG_FILE"
echo " - Data Path : $DATA_PATH" >> "$LOG_FILE"
echo " - Class Name : $CLASS_NAME" >> "$LOG_FILE"
echo " - Feature Layers : $FEATURE_LAYERS" >> "$LOG_FILE"
echo " - R List (聚合) : $R_LIST" >> "$LOG_FILE"
echo " - Image Resize : $IMG_RESIZE" >> "$LOG_FILE"
echo " - Backbone : $BACKBONE_NAME" >> "$LOG_FILE"
echo " - Threshold (瑕疵門檻) : $THRESHOLD" >> "$LOG_FILE"
echo " - Alpha (熱力圖透明度) : $ALPHA" >> "$LOG_FILE"
echo " - Wafer Mask Ratio : $WAFER_MASK_RATIO" >> "$LOG_FILE"
echo " - Auto Align / PhotoNorm / 90Deg : $AUTO_ALIGN_WAFER / $PHOTOMETRIC_NORM / $FIX_90DEG_ROTATION" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

START_TIME=$SECONDS

CUDA_VISIBLE_DEVICES=${DEVICE} python -u examples/muscv2_main.py \
  --device 0 \
  --data_path "$DATA_PATH" \
  --dataset_name "$DATASET_NAME" \
  --class_name "$CLASS_NAME" \
  --backbone_name "$BACKBONE_NAME" \
  --feature_layers $FEATURE_LAYERS \
  --img_resize "$IMG_RESIZE" \
  --r_list $R_LIST \
  --output_dir "$OUTPUT_DIR" \
  --vis True --save_excel True --vis_type whole_norm | tee -a "$LOG_FILE"

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "----------------------------------------" >> "$LOG_FILE"
echo "推論總耗時: ${ELAPSED_TIME} 秒" | tee -a "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
