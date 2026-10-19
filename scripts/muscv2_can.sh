#!/bin/bash

# ==========================================
# 1. 在這裡集中設定你要測試的超參數
# ==========================================
DEVICE=0

# ====== [進公司需修改處 1：切換資料路徑與類別名稱] ======
# # 【公司外 MVTec-2 測試設定】
# DATA_PATH="./data/mvtec-2"
# DATASET_NAME="mvtec_2"          # 固定維持 mvtec_2，以呼叫 datasets/mvtec2.py
# CLASS_NAME="can"                # 對應 ./data/mvtec-2/ 下的 can 資料夾

# 將原本的 CLASS_NAME="can" 改為指定底下的測試子資料夾 (例如只測 test_public 或 test_private_mixed)
DATA_PATH="./data/mvtec-2/can"
DATASET_NAME="mvtec_2"
CLASS_NAME="test_public"        # 這樣就只會讀取 test_public 裡面的圖片，不會把 train 也混進來！


# 【進公司晶圓測試設定】(進公司時將上面三行註解，取消下面三行註解)
# DATA_PATH="./data/categorized_dataset/TMUN87"
# DATASET_NAME="mvtec_2"        # 固定維持 mvtec_2 不用改！
# CLASS_NAME="other_status"     # 對應 TMUN87 下的圖片資料夾名稱
# ========================================================

# ====== [進公司需修改處 2：實驗超參數調整] ======
BACKBONE_NAME="dinov2_vitl14"
# FEATURE_LAYERS="3 7 11"         # ViT-Large 共24層，後續也可測試 "5 11 17 23"
FEATURE_LAYERS="5 11 17 23"   # 加入第 17, 23 深層語意特徵，抑制正常貼紙反光誤判
R_LIST="1 3 5"                  # 包含 1 以保留微小瑕疵獨立特徵
IMG_RESIZE=518                  # 若要對齊網格數可測 392 或 518
OUTPUT_DIR="./output/"

# 設定視覺化的 Threshold 與 Alpha (使用 export 讓 Python 直接讀取 (對應論文 normalization01 後的 0.0 ~ 1.0 熱力圖尺度) )
export THRESHOLD=0.8
export ALPHA=0.6

# 新增：晶圓圓形有效區域比例 (0.0 代表關閉遮罩，適合公司外測 can；進公司測整張 Wafer 建議設為 0.92 ~ 0.95，代表只保留中心 92%~95% 半徑內的晶圓區域，濾除黑邊與邊緣反光)
export WAFER_MASK_RATIO=0.0
# ================================================

# 確保輸出資料夾存在
mkdir -p "$OUTPUT_DIR"

# ==========================================
# 2. 建立【資料集名稱_台北時間】資料夾與紀錄檔
# ==========================================
DATASET_ID=$(basename "$DATA_PATH")

# 強制取得台北時間 (UTC+8)
TIMESTAMP=$(date -u -d '+8 hours' '+%Y%m%d_%H%M%S')
READABLE_TIME=$(date -u -d '+8 hours' '+%Y-%m-%d %H:%M:%S')

# 組合出完整的專屬輸出資料夾路徑，並 export 讓 Python 直接共用同一個路徑
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
echo "----------------------------------------" >> "$LOG_FILE"

# ==========================================
# 3. 開始計時並執行模型
# ==========================================
START_TIME=$SECONDS

CUDA_VISIBLE_DEVICES=${DEVICE} python examples/muscv2_main.py \
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

# ==========================================
# 4. 結算推論時間
# ==========================================
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "----------------------------------------" >> "$LOG_FILE"
echo "推論總耗時: ${ELAPSED_TIME} 秒" | tee -a "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"