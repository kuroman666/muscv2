# MuSc-V2 — Zero-Shot & Training-Free 高解析度無 GT 晶圓瑕疵抓漏系統

本專案基於 **MuSc-V2 (TPAMI 2026)** 打造全自動、**完全免訓練（Training-Free）** 且 **零標註樣本（Zero-Shot / Zero-GT）** 的工業視覺與晶圓瑕疵異常偵測管線。

與需要挑選 Golden Support Set 並微調訓練的模型不同，**MuSc-V2 直接透過盲池內影像彼此之間的跨圖塊互評機制（Mutual Scoring Mechanism, MSM）與全域鄰域重評（RsCon）** 找出隔離的異常瑕疵，天生具備 **0 訓練時間、0 過擬合風險** 的優勢，同時整合了：

* **推論前圓形背景遮罩與秒級預覽工具（`test_mask_muscv2.sh`）**：可在正式推論前先將圓形晶圓外的背景塗黑（支援自動偵測圓心/半徑，或手動指定像素/比例），並提供獨立預覽腳本，免跑模型、幾秒內即可檢查遮罩貼合度並自動報出建議參數。
* **智慧解析度防呆切塊（Resolution-Aware Tiling）**：自動偵測圖片解析度；廠內大圖（如 `2048×2048`、`4096×4096`、`5000×5000`）自動切為 `512×512`（重疊 `64px` / 12.5%）並跳過純黑背景塊、排除同母圖互評；廠外中低解析度公開集（`< 1536px`）自動維持整張原圖推論，避免切碎物件語意。
* **無損加速與 RAM 即時縫合**：支援 TF32 / AMP (`bf16`/`fp16`)、多執行緒讀圖、RAM 充足時單次 Forward 收齊全部層特徵，以及降維畫布（`1024px`）多核心平行縫合與碎片 `.npy` 自動銷毀，僅異常圖映回 1:1 原圖畫框。
* **四合一防誤報純化濾波**：矩形邊界 Padding 抑制、晶圓圓周倒角（Bevel / Notch）自動偵測與餘弦衰減、灰階形態學微塵開運算濾波、峰值自適應紅框收斂。
* **背景系統資源自動監控**：執行 `.sh` 時自動每秒記錄 Host RAM 與 GPU VRAM 使用量，結束時自動統計峰值並安全回收監控程序。

---

## 一、環境與離線權重準備

1. **啟動 Conda 環境**：
   ```bash
   conda activate muscv2
   ```
2. **離線預訓練權重放置位置（公司無外網環境必看）**：
   腳本具備離線權重自動搜尋機制（`--ckpt auto`），會依序掃描以下路徑中的 `.pth` / `.pt` / `.bin` 權重檔（優先匹配 `dinov2_vitl14` → `dinov3` → `dinov2`）：
   * `./dinov2_vitl14_pretrain.pth`（直接放在專案根目錄最方便）
   * `./hub_cache/` 或 `./checkpoints/`
   * `~/.cache/torch/hub/checkpoints/`

---

## 二、Input 資料夾結構與放置位置

### 1. 公司廠內無 GT 實機測試（`run_inside_muscv2.sh`）
將待測的無標註晶圓圖片放入任意資料夾（支援 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`，含 **16-bit TIFF** 與巢狀子資料夾，程式會遞迴自動掃描）：

```text
MuSc-V2/
└── datasets/                         # 廠內測試資料集預設根目錄
    └── fab_wafer_pool/               # [Input] 待測無 GT 晶圓盲池（可自訂資料夾名稱與多層子目錄）
        ├── wafer_001.tif
        ├── wafer_002.tif
        ├── lot_A/
        │   └── wafer_003.png
        └── ...
```

### 2. 廠外公開資料集模擬無 GT 測試（`run_outside_muscv2.sh`）
將 MVTec AD、MVTec AD 2、Mura、AVMap 或 **YDFID-2** 資料集放入 `./data/` 下（腳本會自動相容 `good/` 與 `defect-free/`，抽取正常圖 `clean_*` 與少量隱藏瑕疵 `escape_*` 組成模擬盲池）：

```text
MuSc-V2/
└── data/                             # 廠外公開資料集根目錄
    ├── mvtec/                        # MVTec AD
    │   ├── bottle/
    │   │   └── test/
    │   │       ├── good/
    │   │       └── broken_large/ ...
    │   └── capsule/ ...
    ├── mvtec-2/                      # MVTec AD 2
    │   └── can/ ...
    └── YDFID-2/                      # YDFID-2（可直接指定整個根目錄自動展開所有類別）
        ├── CL1/
        │   └── test/
        │       ├── defect-free/
        │       └── defect/
        └── ...
```

---

## 三、執行指令教學（Mask 預覽、前景 `bash` 與背景 `nohup`）

首次使用請先賦予執行權限：

```bash
chmod +x test_mask_muscv2.sh run_inside_muscv2.sh run_outside_muscv2.sh
```

### 步驟 0（廠內推論前推薦）：秒級測試圓形 Mask 大小（`test_mask_muscv2.sh`）

若希望在推論前先用圓形 Mask 蓋掉晶圓外的背景雜訊，請先用此獨立腳本抽樣 5 張圖測試。**不載入模型、不切塊，幾秒內即完成**，並會在終端機直接印出可複製貼進 `run_inside_muscv2.sh` 的參數設定。

* **1. 先用自動偵測圓心 + 預設內縮 2% 測試**：
  ```bash
  bash test_mask_muscv2.sh ./datasets/fab_wafer_pool
  ```
* **2. 調整半徑內縮比例（第 2 個參數為 `MARGIN`；正值內縮、負值外擴）**：
  ```bash
  bash test_mask_muscv2.sh ./datasets/fab_wafer_pool 0.03
  ```
* **3. 若自動偵測圓心或半徑不準，手動指定 `CX CY R`**（參數順序：`資料夾 [MARGIN] [CX] [CY] [R]`，輸入 `0~1` 為原圖比例、`>1` 為絕對像素）：
  ```bash
  # 比例模式（置中，半徑取短邊 46%）
  bash test_mask_muscv2.sh ./datasets/fab_wafer_pool 0.0 0.5 0.5 0.46

  # 像素模式（參考預覽圖印出的中位數像素微調）
  bash test_mask_muscv2.sh ./datasets/fab_wafer_pool 0.0 2510 2480 2320
  ```
* **4. 確認結果並套用**：
  開啟 `./wafer_escape_results/mask_preview_test/<時間戳記>/mask_preview/` 檢視對照圖（**紅圈**＝原始半徑、**綠圈**＝實際遮罩半徑、**黃十字**＝圓心）。確認綠圈剛好貼合晶圓後，將終端機印出的參數直接貼進 `run_inside_muscv2.sh` 頂部的參數區：
  ```bash
  WAFER_EDGE_MODE="on"
  PRE_MASK="--pre_mask"
  PRE_MASK_MARGIN=0.02
  PRE_MASK_FILL=0
  MASK_CX=-1
  MASK_CY=-1
  MASK_R=-1
  ```

---

### 情境 A：公司廠內無 GT 實機盲池（`run_inside_muscv2.sh`）

#### 1. 前景直接執行（`bash`）
* **模式 1：自動百分位門檻（預設 Top 5% P95 畫框 + Top 3% P97 參考）**
  ```bash
  bash run_inside_muscv2.sh ./datasets/fab_wafer_pool
  ```
* **模式 2：指定參考門檻數值**
  ```bash
  bash run_inside_muscv2.sh ./datasets/fab_wafer_pool 0.32
  ```
* **模式 3：一次批次跑多個盲池資料夾**
  ```bash
  bash run_inside_muscv2.sh ./datasets/lot_01 ./datasets/lot_02
  ```

#### 2. 背景掛機執行（`nohup`，適合大批量晶圓、關閉 SSH 不中斷）
```bash
nohup bash run_inside_muscv2.sh ./datasets/fab_wafer_pool > run_inside.nohup.log 2>&1 &
```
* **即時查看執行進度**：
  ```bash
  tail -f run_inside.nohup.log
  ```
* **即時查看系統 RAM 與 GPU VRAM 消耗**：
  ```bash
  tail -f wafer_escape_results/latest_mem_monitor_inside.log
  ```
* **確認背景任務是否仍在執行 / 手動中止**：
  ```bash
  ps aux | grep 1_infer_muscv2_no_gt.py
  # 若需中止：kill <PID>
  ```

---

### 情境 B：廠外公開資料集模擬測試（`run_outside_muscv2.sh`）

* **執行腳本內預設的公開資料集類別**：
  ```bash
  bash run_outside_muscv2.sh
  ```
* **指定特定公開資料集類別，或直接傳入整個 YDFID-2 根目錄自動展開全部類別**：
  ```bash
  bash run_outside_muscv2.sh ./data/mvtec/bottle ./data/mvtec-2/can
  bash run_outside_muscv2.sh ./data/YDFID-2
  ```
* **背景 `nohup` 執行並即時查看進度與資源監控**：
  ```bash
  nohup bash run_outside_muscv2.sh ./data/YDFID-2 > outside_test_YDFID-2.log 2>&1 &
  tail -f outside_test_YDFID-2.log
  # 另開終端機監控記憶體：
  tail -f output/latest_mem_monitor.log
  ```

---

## 四、Output 資料夾結構與輸出檔案說明

### 1. 廠內實機測試輸出結構（`./wafer_escape_results/`）

```text
wafer_escape_results/
├── latest_inside_muscv2_summary.txt              # [總表] 最新多盲池統計總結 (TXT)
├── latest_inside_muscv2_summary.csv              # [總表] 最新多盲池統計總結 (CSV)
├── latest_mem_monitor_inside.log                 # [監控] 最新每秒 RAM / GPU VRAM 消耗紀錄
├── mask_preview_test/                            # [預覽] test_mask_muscv2.sh 產出的圓形 Mask 對照圖
│   └── <YYYYMMDD_HHMMSS>/
│       └── mask_preview/
│           └── preview_wafer_001.png
└── inside_no_gt_muscv2_<盲池資料夾名稱>/
    ├── latest_log.txt                            # 該盲池最新完整終端機日誌
    └── <YYYYMMDD_HHMMSS>/                        # 每次執行的獨立時間戳記目錄（不互相覆蓋）
        ├── log.txt                               # 本次執行完整日誌與耗時統計
        ├── tile_manifest.json                    # 切塊座標與圓心幾何快取紀錄
        ├── escape_ranking_report.txt             # [核心報表] 全盲池由高至低異常分數排名 (含門檻統計)
        ├── escape_ranking_report.csv             # [核心報表] CSV 格式完整分數明細 (含 Raw 與 Filtered 分數)
        ├── amap_npy/                             # [熱力圖矩陣] 僅儲存判定為異常的整張 Wafer .npy（碎片已自動銷毀）
        │   └── Rank001_Score0.4120_wafer_005.npy
        └── anomaly_images/                       # [異常視覺化] 僅輸出超過門檻的異常晶圓（正常圖自動略過）
            ├── Rank001_Score0.4120_Box02_wafer_005.png   # 三合一對照圖 (原圖 | 熱力疊圖 | 紅框定位)
            └── full_res_bbox_only/
                └── Rank001_Score0.4120_Box02_wafer_005.png # 1:1 原生解析度純紅框標註圖
```

### 2. 廠外公開資料集輸出結構（`./output/`）
結構與上方一致，根目錄位於 `./output/`：
* **彙整總表**：`./output/latest_multi_dataset_summary_muscv2.txt` 與 `.csv`（含 **I-AUROC**、**I-AP**、**F1-max**、**Recall**、**FPR** 與正常圖分數分佈）。
* **記憶體監控**：`./output/latest_mem_monitor.log`。
* **個別資料集結果**：`./output/2_fab_no_gt_<資料集>_<類別>_muscv2/<YYYYMMDD_HHMMSS>/`。

---

## 五、常用超參數調整指南

在 `run_inside_muscv2.sh` 或 `run_outside_muscv2.sh` 頂部的 `[可自由修改的參數區]` 可依機台特性微調：

| 參數名稱 | 預設值 | 說明與調整建議 |
| :--- | :--- | :--- |
| `PRE_MASK` | `""` | 推論前圓形背景遮罩開關（僅廠內腳本）：設為 `"--pre_mask"` 啟用，留空關閉。建議先用 `test_mask_muscv2.sh` 測試。 |
| `PRE_MASK_MARGIN` | `0.02` | 圓形 Mask 半徑內縮比例（`0.02` = 內縮 2%，負值如 `-0.01` 為外擴）。 |
| `PRE_MASK_FILL` | `0` | 圓外背景填補像素值（預設 `0` = 純黑，能讓純黑角落塊被自動跳過以加速推論）。 |
| `MASK_CX` / `MASK_CY` / `MASK_R` | `-1` | 手動指定圓心與半徑：`-1` 為自動偵測；`0~1` 為原圖寬高/短邊比例；`>1` 為絕對像素值。 |
| `WAFER_EDGE_MODE` | `"auto"` | 後處理晶圓圓周倒角（Bevel）邊緣抑制：`auto`（自動偵測）、`on`（強制開啟）、`off`（關閉）。 |
| `WAFER_EDGE_MARGIN` | `0.04` | 後處理圓周外圈平滑衰減寬度佔半徑比例（預設外圈 `4%` 半徑平滑抑制）。 |
| `THRESH_MODE` | `"auto"` | 門檻模式：預設 `auto` 採用 Top 5% (P95) 作為畫框基準，並同步於日誌回報 Top 3% (P97) 切點參考。 |
| `MIN_TILING_RES` | `1536` | 大圖防呆門檻：圖片長邊 ≥ `1536px` 才啟動切塊，低於此解析度自動整張推論。 |
| `TILE_SIZE` / `TILE_OVERLAP` | `512` / `64` | 切塊解析度與重疊寬度（`64 / 512 = 12.5%` 重疊）。 |
| `SPECKLE_FILTER_RATIO` | `0.008` | 形態學開運算濾除單一 Patch 微塵雜訊的比例（若要抓極微小顆粒，可調降至 `0.003` 或 `0`）。 |
| `MAX_PEER_TILES` | `400` | 大盲池互評時，每個 Query Tile 最多抽樣比較的他片晶圓切塊數（兼顧速度與準確度）。 |

---

## 六、公司廠內實機執行突發狀況排解（FAQ）

### Q1：自動偵測的圓形 Mask 圓心歪掉，或切到晶圓有效區域 / 留到背景怎麼辦？
* **解決方法**：先跑 `test_mask_muscv2.sh` 看預覽圖與終端機印出的中位數像素/比例，再微調參數重測：
  * **切太裡面（吃掉晶圓邊緣）**：調小 `PRE_MASK_MARGIN`（可設負值如 `-0.01`）或調大 `MASK_R`。
  * **切太外面（留到背景雜訊）**：調大 `PRE_MASK_MARGIN`（如 `0.03`）或調小 `MASK_R`。
  * **黃十字圓心偏移**：直接手動傳入 `MASK_CX` / `MASK_CY` / `MASK_R`（支援 `0~1` 比例或 `>1` 像素），確認綠圈貼合後再貼回 `run_inside_muscv2.sh`。

### Q2：執行時無預警中斷，終端機只顯示 `Killed`（Linux OOM Killer）怎麼辦？
* **原因**：公司盲池圖片數量過多（例如一次丟入 500 張 `5000×5000` 大圖，切出數萬張切塊），導致系統主記憶體（Host RAM）耗盡。
* **解決方法**：
  1. 確認已啟用 `--pre_mask` 蓋掉圓外背景，能讓大量純黑角落塊被自動略過，顯著減少總切塊數。
  2. 程式內建記憶體防爆調節器與單層串流備援模式；若單批圖片仍超過數百張大圖，建議在 `run_inside_muscv2.sh` 將 `TILE_SIZE=1024`、`TILE_OVERLAP=128`（切塊數降為 1/4），或分批（每批 100~200 張）執行。

### Q3：出現 `torch.OutOfMemoryError: CUDA out of memory`（GPU 顯存不足）怎麼辦？
* **解決方法**：
  1. 調降 `run_inside_muscv2.sh` 中的 `INFER_BATCH_SIZE`（例如改為 `8` 或 `4`）。
  2. 調降 `MAX_PEER_TILES`（從 `400` 改為 `200` 或 `250`）。
  3. 透過 `nvidia-smi` 確認空閒顯卡，並以 `CUDA_VISIBLE_DEVICES=1 bash run_inside_muscv2.sh ...` 指定顯卡執行。

### Q4：公司內網無法連上網際網路，報錯 `URLError` 或找不到權重檔怎麼辦？
* **解決方法**：
  1. 將離線的 `dinov2_vitl14_pretrain.pth`（或 `dinov3` 權重檔）複製到 `MuSc-V2/` 專案根目錄下。
  2. 或在 `run_inside_muscv2.sh` 直接指定絕對路徑：
     ```bash
     DINO_CKPT_PATH="/home/你的帳號/project/muscv2/dinov2_vitl14_pretrain.pth"
     ```

### Q5：機台拍出來的 `.tif` / `.tiff` 是 16-bit 灰階圖，讀進來全黑或對比度異常？
* **說明**：本專案內建 `_robust_normalize_to_uint8()`，會自動針對 `I;16` / 單通道灰階 / 浮點數 TIFF 採用 **P0.1 ~ P99.9 穩健百分位數拉伸** 轉為 8-bit RGB，防止極端飽和亮點壓扁整體對比；若 `PIL` 無法開啟會自動切換 `OpenCV IMREAD_UNCHANGED` 備援讀取。

### Q6：推論結果正常晶圓誤報（False Positive）偏多，尤其是晶圓外圈背景或邊緣倒角（Bevel）被框起來？
* **解決方法**：
  1. **推論前先遮背景**：使用 `test_mask_muscv2.sh` 調好圓形 Mask 並在 `run_inside_muscv2.sh` 開啟 `PRE_MASK="--pre_mask"`，從源頭消除背景干擾。
  2. **加強後處理邊緣抑制**：在 `run_inside_muscv2.sh` 強制設為：
     ```bash
     WAFER_EDGE_MODE="on"
     WAFER_EDGE_MARGIN=0.05   # 將外圈抑制範圍從 4% 提高至 5%
     ```

### Q7：有真實微小的刮傷或 Particle 漏抓（False Negative）怎麼辦？
* **解決方法**：
  1. 將微塵開運算濾波調小或關閉（避免把真實微缺陷當成噪點濾掉）：
     ```bash
     SPECKLE_FILTER_RATIO=0.003   # 或設為 0
     MIN_AREA_RATIO=0.00005
     ```
  2. 關閉濾波對照原生分數：加上 `RAW_SCORE_MODE="--raw_score_mode"`，查看 `escape_ranking_report.csv` 內的 `Raw_Max_Score` 確認原始模型是否有偵測到該缺陷。

### Q8：替換了盲池資料夾內的圖片，但腳本卻顯示 `[⚡ 測試池清單快取命中]` 沿用舊資料？
* **說明**：快取指紋會比對所有圖片的「相對路徑 + 檔案大小」以及所有切塊與 Mask 參數（調整 `PRE_MASK`、`MASK_CX/CY/R` 會自動失效重切）。若僅替換了同名且檔案大小完全相同的圖片內容，請在 `run_inside_muscv2.sh` 設定 `FORCE_INFER="--force_infer"` 執行一次刷新快取（跑完記得改回 `FORCE_INFER=""`）。

### Q9：為什麼 `anomaly_images/` 裡只有少數圖片，其他正常圖沒輸出？
* **說明**：這是**正常的防呆省空間設計**。為了避免廠內數千張正常晶圓塞爆硬碟，只有分數 ≥ `Effective Threshold` 且通過微塵面積過濾的**異常晶圓**才會輸出紅框圖與 `.npy`；所有圖片（含正常圖）的完整分數與排名皆記錄在 `escape_ranking_report.csv` 與 `escape_ranking_report.txt` 中。