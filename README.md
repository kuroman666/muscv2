# MuSc-V2 — Zero-Shot & Training-Free 高解析度無 GT 晶圓瑕疵抓漏系統

本專案基於 **MuSc-V2 (TPAMI 2026)** 打造全自動、**完全免訓練（Training-Free）** 且 **零標註樣本（Zero-Shot / Zero-GT）** 的工業視覺與晶圓瑕疵異常偵測管線。

與需要挑選 Golden Support Set 並微調訓練的模型不同，**MuSc-V2 直接透過盲池內影像彼此之間的跨圖塊互評機制（Mutual Scoring Mechanism, MSM）與全域鄰域重評（RsCon）** 找出隔離的異常瑕疵，天生具備 **0 訓練時間、0 過擬合風險** 的優勢，同時完整還原原論文核心並整合廠內落地優化：

* **完整還原 MuSc-V2 原論文核心（SNAMD + MSM + RsCon）**：
  * **SNAMD（SWPooling 多尺度融合壓縮）**：對每層特徵進行 L2 正規化後，以相似度加權池化（Similarity-Weighted Pooling）融合多尺度鄰域（`r=1,3,5`）並壓縮回原維度，每層僅需執行 1 次 MSM 互評（較逐尺度互評提速 3 倍）。
  * **MSM（跨片互評）**：排除同母圖切塊，以固定參考池抽樣（`MAX_PEER_TILES=400`）將複雜度從 $O(N^2)$ 降為線性，並取最低距離前 30% 做區間平均（IA）。
  * **RsCon（$k=7$ WMO 重評校準）**：抽取倒數第二層異常分數最高的 Patch 特徵，與最相似的 $k=7$ 個鄰居進行 0.5/0.5 加權重評，抑制正常區域誤報。
* **推論前圓形背景遮罩與獨立設定檔（`test_mask_muscv2.sh`）**：可在正式推論前將圓形晶圓外的背景塗黑（支援自動偵測圓心/半徑，或手動指定像素/比例）。測完後會自動依資料夾階層（`UNIQUE_TAG`）存成獨立的 `.env` 設定檔，`run_inside_muscv2.sh` 執行時會自動對應載入，多資料夾批次跑也不會互相污染。
* **大資料集自動均勻分包與 Mask 繼承（`split_dataset_muscv2.sh`）**：針對數百至數千張 `5000×5000` 大圖與 128 GiB 容器記憶體限制，提供「隨機打亂 + 輪流發牌均勻分包（預設每包 ≤ 160 張）」腳本；全程使用軟連結（**0 額外硬碟空間**），並自動把原資料夾測好的 Mask `.env` 複製繼承給每一個 `part_XX`。
* **智慧解析度防呆切塊與 Docker cgroup 記憶體感知**：
  * 自動偵測圖片解析度：廠內大圖（≥ 1536px，如 `4096×4096`、`5000×5000`）自動切為 `512×512`（重疊 `64px` / 12.5%）並跳過純黑背景塊；廠外公開集（< 1536px）自動維持整張原圖推論。
  * 自動同時偵測實體主機與 Docker cgroup (v1/v2) 記憶體上限，防止容器 OOM 閃退。
* **無損加速、即時進度條與 RAM 平行縫合**：支援 TF32 / AMP (`bf16`/`fp16`)、多執行緒讀圖、單次 Forward 全層收集（RAM 充足時）、Step 2.2 每 50 批次即時進度條、降維畫布（`1024px`）多核心平行縫合與碎片 `.npy` 自動銷毀，僅異常圖映回 1:1 原圖畫框。
* **四合一防誤報純化濾波 & 背景資源監控**：包含矩形邊界抑制、圓周倒角餘弦衰減、形態學微塵開運算、峰值自適應紅框收斂，以及每秒自動記錄 RAM / GPU VRAM 峰值的背景監控程序。

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
/workspace/dataset_use/               # 任意廠內資料夾路徑（腳本會自動往上抓最多 3 層組成 UNIQUE_TAG）
└── W639/
    └── TMQB28/
        └── other_status/             # [Input] 待測無 GT 晶圓盲池（UNIQUE_TAG = W639_TMQB28_other_status）
            ├── wafer_001.tif
            ├── wafer_002.tif
            └── ...
```

### 2. 廠外公開資料集模擬無 GT 測試（`run_outside_muscv2.sh`）
將 MVTec AD、MVTec AD 2、Mura、AVMap 或 **YDFID-2** 資料集放入 `./data/` 下（自動相容 `good/` 與 `defect-free/`）：

```text
MuSc-V2/
└── data/
    ├── mvtec/
    │   └── bottle/test/{good, broken_large, ...}
    ├── mvtec-2/
    │   └── can/ ...
    └── YDFID-2/                      # 可直接指定根目錄自動展開所有類別
        └── CL1/test/{defect-free, defect}
```

---

## 三、執行指令教學（標準三步驟流程）

首次使用請先賦予所有腳本執行權限：

```bash
chmod +x test_mask_muscv2.sh split_dataset_muscv2.sh run_inside_muscv2.sh run_outside_muscv2.sh
```

### 情境 A：公司廠內無 GT 實機盲池（標準三步驟）

#### 步驟 1：秒級測試並儲存該資料夾專屬 Mask 設定檔（`test_mask_muscv2.sh`）
**不載入模型、不切塊，幾秒內即完成**。測試完畢後會自動將參數存成 `./wafer_escape_results/mask_configs/mask_<UNIQUE_TAG>.env`，不需再手動修改 `run_inside_muscv2.sh`。

* **自動偵測圓心 + 預設內縮 2%**：
  ```bash
  bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status
  ```
* **調整半徑內縮比例（第 2 個參數為 `MARGIN`；正值內縮、負值外擴）**：
  ```bash
  bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 0.03
  ```
* **手動指定圓心與半徑 `CX CY R`**（參數順序：`資料夾 [MARGIN] [CX] [CY] [R]`，輸入 `0~1` 為比例、`>1` 為像素）：
  ```bash
  # 比例模式（置中，半徑取短邊 46%）
  bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 0.0 0.5 0.5 0.46

  # 像素模式（參考預覽圖中位數微調）
  bash test_mask_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status 0.0 2510 2480 2320
  ```
* 開啟 `./wafer_escape_results/mask_preview_test/<時間戳記>/mask_preview/` 確認**綠圈**（實際遮罩）與**黃十字**（圓心）貼合晶圓即可。

#### 步驟 2（大資料集必跑）：自動均勻分包 + 繼承 Mask（`split_dataset_muscv2.sh`）
當單一資料夾超過 **160 張** `5000×5000` 大圖（在 128 GiB 容器下以 `TILE_SIZE=512` 推論會超過記憶體上限），請先執行分包腳本。它會隨機打亂並均分成每包 ≤ 160 張的軟連結子資料夾（不佔硬碟空間），並**自動把步驟 1 存好的 Mask `.env` 複製給每個 `part_XX`**：

```bash
# 預設每包上限 160 張（亦可於第 2 個參數自訂，如 150）
bash split_dataset_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status
```

*(註：若原始資料夾本來就 ≤ 160 張，可跳過步驟 2，直接執行步驟 3。)*

#### 步驟 3：啟動推論（`run_inside_muscv2.sh`，支援前景 `bash` 與背景 `nohup`）
* **單一中小型資料夾（≤ 160 張）**：
  ```bash
  nohup bash run_inside_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status > run_W639.log 2>&1 &
  ```
* **已分包的大型資料夾（直接貼上步驟 2 結尾印出的指令，全部 `part_*` 自動排隊跑完並各自載入 Mask）**：
  ```bash
  nohup bash run_inside_muscv2.sh ./datasets/split_W639_TMQB28_other_status/part_* > run_W639_split.log 2>&1 &
  ```
* **一次批次跑多個不同資料夾（各自自動載入對應的 `mask_<UNIQUE_TAG>.env`）**：
  ```bash
  nohup bash run_inside_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status /workspace/dataset_use/W642/TMQB28/other_status > run_multi.log 2>&1 &
  ```
* **即時監控進度與系統資源**：
  ```bash
  tail -f run_W639_split.log
  # 另開終端機看每秒 RAM / VRAM：
  tail -f wafer_escape_results/latest_mem_monitor_inside.log
  ```

---

### 情境 B：廠外公開資料集模擬測試（`run_outside_muscv2.sh`）

```bash
# 執行腳本內預設類別
bash run_outside_muscv2.sh

# 指定特定類別或整個 YDFID-2 根目錄（背景掛機）
nohup bash run_outside_muscv2.sh ./data/YDFID-2 > outside_test_YDFID-2.log 2>&1 &
tail -f outside_test_YDFID-2.log
```

---

## 四、Output 資料夾結構與輸出檔案說明

### 1. 廠內實機測試輸出結構（`./wafer_escape_results/`）

```text
MuSc-V2/
├── datasets/
│   └── split_<ORIG_TAG>/                         # [分包] split_dataset_muscv2.sh 建立的軟連結子包（0 額外空間）
│       ├── part_01/
│       └── part_02/ ...
└── wafer_escape_results/
    ├── latest_inside_muscv2_summary.txt          # [總表] 最新多盲池統計總結 (TXT)
    ├── latest_inside_muscv2_summary.csv          # [總表] 最新多盲池統計總結 (CSV)
    ├── latest_mem_monitor_inside.log             # [監控] 最新每秒 RAM / GPU VRAM 消耗紀錄
    ├── mask_configs/                             # [Mask 設定] 各資料夾與分包專屬的 Mask .env 設定檔
    │   ├── mask_<ORIG_TAG>.env
    │   └── mask_split_<ORIG_TAG>_part_01.env
    ├── mask_preview_test/                        # [預覽] test_mask_muscv2.sh 產出的圓形 Mask 對照圖
    │   └── <YYYYMMDD_HHMMSS>/mask_preview/preview_*.png
    └── inside_no_gt_muscv2_<UNIQUE_TAG>/
        ├── latest_log.txt                        # 該盲池最新完整終端機日誌
        └── <YYYYMMDD_HHMMSS>/                    # 每次執行的獨立時間戳記目錄（不互相覆蓋）
            ├── log.txt                           # 本次執行完整日誌與耗時統計
            ├── tile_manifest.json                # 切塊座標與圓心幾何快取紀錄
            ├── escape_ranking_report.txt         # [核心報表] 全盲池由高至低異常分數排名 (含門檻統計)
            ├── escape_ranking_report.csv         # [核心報表] CSV 格式完整分數明細 (含 Raw 與 Filtered 分數)
            ├── amap_npy/                         # [熱力圖矩陣] 僅儲存判定為異常的整張 Wafer .npy
            └── anomaly_images/                   # [異常視覺化] 僅輸出超過門檻的異常晶圓（正常圖自動略過）
                ├── Rank001_Score0.4120_Box02_wafer_005.png   # 三合一對照圖 (原圖 | 熱力疊圖 | 紅框定位)
                └── full_res_bbox_only/
                    └── Rank001_Score0.4120_Box02_wafer_005.png # 1:1 原生解析度純紅框標註圖
```

### 2. 廠外公開資料集輸出結構（`./output/`）
* **彙整總表**：`./output/latest_multi_dataset_summary_muscv2.txt` 與 `.csv`（含 **I-AUROC**、**I-AP**、**F1-max**、**Recall**、**FPR**）。
* **記憶體監控**：`./output/latest_mem_monitor.log`。
* **個別資料集結果**：`./output/2_fab_no_gt_<資料集>_<類別>_muscv2/<YYYYMMDD_HHMMSS>/`。

---

## 五、常用超參數調整指南

在 `run_inside_muscv2.sh` 或 `run_outside_muscv2.sh` 頂部的 `[可自由修改的參數區]` 可依機台特性微調：

| 參數名稱 | 預設值 | 說明與調整建議 |
| :--- | :--- | :--- |
| `AUTO_LOAD_MASK_CFG` | `1` | `1` = 自動讀取 `mask_configs/mask_<UNIQUE_TAG>.env`；`0` = 改用腳本內手動填寫的預設 Mask 參數。 |
| `PRE_MASK` | `""` | 推論前圓形背景遮罩預設開關（找不到 `.env` 或關閉自動載入時生效）。 |
| `PRE_MASK_MARGIN` | `0.02` | 圓形 Mask 半徑內縮比例（`0.02` = 內縮 2%，負值如 `-0.01` 為外擴）。 |
| `MASK_CX` / `MASK_CY` / `MASK_R` | `-1` | 手動指定圓心與半徑：`-1` 為自動偵測；`0~1` 為原圖寬高/短邊比例；`>1` 為絕對像素值。 |
| `WAFER_EDGE_MODE` | `"auto"` | 後處理晶圓圓周倒角（Bevel）邊緣抑制：`auto`、`on`、`off`。 |
| `WAFER_EDGE_MARGIN` | `0.04` | 後處理圓周外圈平滑衰減寬度佔半徑比例（預設外圈 `4%`）。 |
| `THRESH_MODE` | `"auto"` | 門檻模式：預設 `auto` 採用 Top 5% (P95) 作為畫框基準，並同步回報 Top 3% (P97) 切點參考。 |
| `MIN_TILING_RES` | `1536` | 大圖防呆門檻：圖片長邊 ≥ 1536px 才啟動切塊，低於此解析度自動整張推論。 |
| `TILE_SIZE` / `TILE_OVERLAP` | `512` / `64` | 切塊解析度與重疊寬度（`64 / 512 = 12.5%` 重疊）。 |
| `INFER_BATCH_SIZE` | `16` | 特徵提取 Batch Size（80 GB 顯卡可視需求調高至 `64` 以加速）。 |
| `SPECKLE_FILTER_RATIO` | `0.008` | 形態學開運算濾除單一 Patch 微塵雜訊的比例（抓極微小顆粒時可降至 `0.003` 或 `0`）。 |
| `MAX_PEER_TILES` | `400` | 大盲池互評時，每個 Query Tile 最多抽樣比較的他片晶圓切塊數。 |
| `RSCON_WEIGHT` | `0.15` | RsCon 重評開關：`> 0` 啟用原論文 $k=7$ WMO 校準，`0` 關閉。 |

---

## 六、公司廠內實機執行突發狀況排解（FAQ）

### Q1：執行時無預警中斷，終端機顯示 `Killed`（Linux OOM Killer）怎麼辦？
* **原因**：公司 Docker 容器通常設有記憶體上限（例如 **128 GiB**）。雖然 `free -m` 看到實體機有 1 TB，但只要容器用量超過 128 GiB 就會被系統直接殺掉。以 `TILE_SIZE=512` 切 `5000×5000` 大圖時，128 GiB 的安全上限約為 **150～160 張原圖（約 1.8～2 萬塊切塊）**。
* **解決方法**：
  1. **推薦解法（不降解析度）**：使用 `bash split_dataset_muscv2.sh <原始資料夾>` 自動將大資料集（如 700 張或 2,500 張）均勻拆成每包 ≤ 160 張的軟連結子包，再交給 `run_inside_muscv2.sh` 依序排隊跑完。
  2. 確認已透過 `test_mask_muscv2.sh` 啟用 `--pre_mask` 蓋掉圓外背景，能讓大量純黑角落塊被自動略過，進一步降低 RAM 峰值。

### Q2：自動偵測的圓形 Mask 圓心歪掉，或切到晶圓有效區域 / 留到背景怎麼辦？
* **解決方法**：重跑 `test_mask_muscv2.sh` 微調參數，新的設定會自動覆蓋舊的 `.env` 檔：
  * **切太裡面（吃掉晶圓邊緣）**：調小 `MARGIN`（可設負值如 `-0.01`）或調大 `R`。
  * **切太外面（留到背景雜訊）**：調大 `MARGIN`（如 `0.03`）或調小 `R`。
  * **黃十字圓心偏移**：直接手動傳入 `CX CY R`（支援 `0~1` 比例或 `>1` 像素）。
  * *(若已經分包過，微調完 Mask 後記得再跑一次 `split_dataset_muscv2.sh` 同步更新各分包的 `.env`。)*

### Q3：出現 `torch.OutOfMemoryError: CUDA out of memory`（GPU 顯存不足）怎麼辦？
* **解決方法**：
  1. 調降 `run_inside_muscv2.sh` 中的 `INFER_BATCH_SIZE`（例如改為 `8` 或 `4`）。
  2. 調降 `MAX_PEER_TILES`（從 `400` 改為 `200` 或 `250`）。
  3. 透過 `nvidia-smi` 確認空閒顯卡，並以 `CUDA_VISIBLE_DEVICES=1 bash run_inside_muscv2.sh ...` 指定顯卡。

### Q4：公司內網無法連上網際網路，報錯 `URLError` 或找不到權重檔怎麼辦？
* **解決方法**：將離線的 `dinov2_vitl14_pretrain.pth`（或 `dinov3` 權重檔）複製到 `MuSc-V2/` 專案根目錄下，或在 `run_inside_muscv2.sh` 直接指定 `DINO_CKPT_PATH` 的絕對路徑。

### Q5：機台拍出來的 `.tif` / `.tiff` 是 16-bit 灰階圖，讀進來全黑或對比度異常？
* **說明**：程式內建 `_robust_normalize_to_uint8()`，會自動針對 `I;16` / 單通道灰階 / 浮點數 TIFF 採用 **P0.1 ~ P99.9 穩健百分位數拉伸** 轉為 8-bit RGB，防止極端飽和亮點壓扁整體對比。

### Q6：有真實微小的刮傷或 Particle 漏抓（False Negative）怎麼辦？
* **解決方法**：
  1. 調小或關閉微塵開運算濾波（避免把真實微缺陷當噪點濾掉）：設 `SPECKLE_FILTER_RATIO=0.003`（或 `0`）、`MIN_AREA_RATIO=0.00005`。
  2. 加上 `RAW_SCORE_MODE="--raw_score_mode"`，查看 `escape_ranking_report.csv` 內的 `Raw_Max_Score` 確認原始模型是否有偵測到該缺陷。

### Q7：替換了盲池資料夾內的圖片，但腳本卻顯示 `[⚡ 測試池清單快取命中]` 沿用舊資料？
* **說明**：快取指紋會比對所有圖片的「相對路徑 + 檔案大小」以及切塊與 Mask 參數。若僅替換了同名且檔案大小完全相同的圖片內容，請在 `run_inside_muscv2.sh` 設定 `FORCE_INFER="--force_infer"` 執行一次刷新快取（跑完記得改回 `FORCE_INFER=""`）。

### Q8：為什麼 `anomaly_images/` 裡只有少數圖片，其他正常圖沒輸出？
* **說明**：這是**正常的防呆省空間設計**。只有分數 ≥ `Effective Threshold` 且通過面積過濾的**異常晶圓**才會輸出紅框圖與 `.npy`；所有圖片（含正常圖）的完整分數與排名皆記錄在 `escape_ranking_report.csv` 與 `escape_ranking_report.txt` 中。