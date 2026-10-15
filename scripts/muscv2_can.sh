#!/bin/bash

# ==========================================
# 1. 在這裡集中設定你要測試的超參數
# ==========================================
DEVICE=0
DATA_PATH="./data/categorized_dataset/TMUN87"
DATASET_NAME="mvtec_2"   # 維持 mvtec_2 才能呼叫你寫好的讀取邏輯"
CLASS_NAME="other_status"       # 對應資料夾名稱
# 建議先用比較淺的特徵跟小的聚合範圍來抓紋理瑕疵
FEATURE_LAYERS="3 7 11"
R_LIST="1 3 5"
IMG_RESIZE=224
OUTPUT_DIR="./output/"
BACKBONE_NAME="dinov2_vitl14"

# 確保輸出資料夾存在
mkdir -p $OUTPUT_DIR

# ==========================================
# 2. 建立動態時間檔名與紀錄檔
# ==========================================
TIMESTAMP=$(date '+%Y年%m月%d日%H時%M分%S秒')
LOG_FILE="${OUTPUT_DIR}${TIMESTAMP}.txt"

echo "========================================" > $LOG_FILE
echo "測試時間: $(date '+%Y-%m-%d %H:%M:%S')" >> $LOG_FILE
echo "【本次實驗超參數】" >> $LOG_FILE
echo "  - Feature Layers : $FEATURE_LAYERS" >> $LOG_FILE
echo "  - R List (聚合)  : $R_LIST" >> $LOG_FILE
echo "  - Image Resize   : $IMG_RESIZE" >> $LOG_FILE
echo "  - Backbone       : $BACKBONE_NAME" >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

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
