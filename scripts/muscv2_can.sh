#!/bin/bash

# ==========================================
# 1. 在這裡集中設定你要測試的超參數
# ==========================================
DEVICE=0
DATA_PATH="./data/categorized_dataset/TMRU50"
DATASET_NAME="mvtec_2"   # 維持 mvtec_2 才能呼叫你寫好的讀取邏輯"
CLASS_NAME="other_status"       # 對應資料夾名稱
# 建議先用比較淺的特徵跟小的聚合範圍來抓紋理瑕疵
FEATURE_LAYERS="3 7 11"
R_LIST="1 3 5"
IMG_RESIZE=224
OUTPUT_DIR="./output/"
BACKBONE_NAME="dinov2_vitl14""

# 新增：設定視覺化的 Threshold 與 Alpha (使用 export 讓 Python 可以直接讀取)
export THRESHOLD=0.35
export ALPHA=0.6

# 確保輸出資料夾存在
mkdir -p $OUTPUT_DIR

# ==========================================
# 2. 建立【資料集名稱_台北時間】資料夾與紀錄檔
# ==========================================
# 自動從 DATA_PATH 擷取最後一層的資料集名稱 (例如 mvtec-2 或 TMUN87)
DATASET_ID=$(basename "$DATA_PATH")

# 強制取得台北時間 (UTC+8)
TIMESTAMP=$(date -u -d '+8 hours' '+%Y%m%d_%H%M%S')
READABLE_TIME=$(date -u -d '+8 hours' '+%Y-%m-%d %H:%M:%S')"

# 組合出完整的專屬輸出資料夾路徑，並 export 讓 Python 直接共用同一個路徑
export EXP_OUTPUT_DIR="${OUTPUT_DIR}${DATASET_NAME}/${BACKBONE_NAME}/imagesize${IMG_RESIZE}/${DATASET_ID}_${TIMESTAMP}/"

# 確保該專屬資料夾存在
mkdir -p "$EXP_OUTPUT_DIR"

# 將實驗紀錄檔 .txt 直接存放在這個【資料集名稱_台北時間】資料夾內
LOG_FILE="${EXP_OUTPUT_DIR}${DATASET_ID}_${TIMESTAMP}_log.txt"

echo "========================================" > "$LOG_FILE"
echo "測試時間: ${READABLE_TIME} (台北時間)" >> "$LOG_FILE"
echo "輸出目錄: ${EXP_OUTPUT_DIR}" >> "$LOG_FILE"
echo "【本次實驗超參數】" >> "$LOG_FILE"
echo " - Data Path : $DATA_PATH" >> "$LOG_FILE"
echo " - Feature Layers : $FEATURE_LAYERS" >> "$LOG_FILE"
echo " - R List (聚合) : $R_LIST" >> "$LOG_FILE"
echo " - Image Resize : $IMG_RESIZE" >> "$LOG_FILE"
echo " - Backbone : $BACKBONE_NAME" >> "$LOG_FILE"
echo " - Threshold (瑕疵門檻) : $THRESHOLD" >> "$LOG_FILE"
echo " - Alpha (熱力圖透明度) : $ALPHA" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

# ==========================================
# 3. 開始計時並執行模型
# ==========================================
START_TIME=$SECONDS

# 這裡完全自動讀取最上方的變數，你不用再改這段了
CUDA_VISIBLE_DEVICES=${DEVICE} python examples/muscv2_main.py \
--device 0 \
--data_path $DATA_PATH \
--dataset_name $DATASET_NAME \
--class_name $CLASS_NAME \
--backbone_name $BACKBONE_NAME \
--feature_layers $FEATURE_LAYERS \
--img_resize $IMG_RESIZE \
--r_list $R_LIST \
--output_dir $OUTPUT_DIR \
--vis True --save_excel True --vis_type whole_norm | tee -a $LOG_FILE

# ==========================================
# 4. 結算推論時間
# ==========================================
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "----------------------------------------" >> $LOG_FILE
echo "推論總耗時: ${ELAPSED_TIME} 秒" | tee -a $LOG_FILE
echo "========================================" >> $LOG_FILE
