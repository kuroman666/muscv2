# MuSc-V2 — Zero-Shot & Training-Free 高解析度無 GT 晶圓瑕疵抓漏系統

本專案基於 **MuSc-V2 (TPAMI 2026 / arXiv:2511.10047v2)** 打造全自動、**完全免訓練（Training-Free）** 且 **零標註樣本（Zero-Shot / Zero-GT）** 的工業視覺與晶圓瑕疵異常偵測管線。

與需要挑選 Golden Support Set 並微調訓練的模型不同，**MuSc-V2 直接透過盲池內影像彼此之間的跨圖塊互評機制（Mutual Scoring Mechanism, MSM）與全域鄰域重評（RsCon）** 找出隔離的異常瑕疵，天生具備 **0 訓練時間、0 過擬合風險** 的優勢，同時完整還原原論文核心並針對廠內高解析度晶圓與容器記憶體限制完成落地優化：

* **Step 0 自動方向對齊與逆轉還原（`--align_mode`）**：
  * 自動以資料集第 1 張圖為基準，透過「粗搜（每 $5^\circ$）＋精搜（$\pm 5^\circ$ 內每 $0.5^\circ$）」將其餘晶圓旋轉對齊後再切塊，解決機台拍攝轉向問題。
  * 推論前圓形遮罩先在未旋轉原圖上套用，再同步旋轉影像與圓心座標，確保手動指定的 `MASK_CX / MASK_CY` 100% 精準對齊。
  * **原圖無損輸出**：Step 3 縫合與防誤報濾波後，熱力圖會自動依負旋轉角逆轉回原圖方向，所有輸出的 `.npy`、三合一對照圖與 1:1 紅框圖皆直接疊在未旋轉的原圖上。
* **完整還原 MuSc-V2 原論文核心（Position-wise SNAMD + MSM + 全晶圓級 RsCon）**：
  * **Position-wise 同座標分批互評**：依 `(y, x)` 座標將切塊分組（邊緣只跟邊緣比、中心只跟中心比），單批記憶體僅取決於總晶圓數 $N$。在 128 GiB Docker 容器下，**無需分包即可直接執行 700 張（峰值約 12–15 GiB）乃至 2,770 張（峰值約 42–45 GiB）$5000 \times 5000$ 大圖**。
  * **SNAMD（SWPooling 多尺度融合壓縮）**：對每層特徵進行 L2 正規化後，以相似度加權池化（Similarity-Weighted Pooling）融合多尺度鄰域（`r=1,3,5`）並壓縮回原維度，每層僅需執行 1 次 MSM 互評（提速 3 倍）。
  * **MSM（跨片互評）**：排除同母圖切塊，並依原論文取最低距離前 $30\%$ 做區間平均（IA）。支援 `MAX_PEER_TILES=0`（100% $N \times N$ 全體互評）或指定上限（如 `150`–`300`，精準對齊原論文規模並大幅加速大資料集）。
  * **全晶圓級 RsCon（$k=7$ WMO 重評校準）**：跨座標追蹤每片晶圓「全片最高異常分 Patch」的倒數第二層特徵，全部座標跑完後統一執行整片晶圓級 $k=7$ WMO 重評，並於 Step 3 縫合時直接乘回校準係數（0 額外磁碟 I/O）。
* **推論前圓形背景遮罩與獨立設定檔（`test_mask_muscv2.sh`）**：可在正式推論前將圓形晶圓外的背景塗黑（支援自動偵測圓心/半徑，或手動指定像素/比例）。測完後會自動依資料夾階層（`UNIQUE_TAG`）存成獨立的 `.env` 設定檔，`run_inside_muscv2.sh` 執行時自動對應載入，多資料夾批次跑也不會互相污染。
* **智慧解析度防呆切塊與 Docker cgroup 記憶體感知**：
  * 自動偵測圖片解析度：廠內大圖（$\ge 1536\text{px}$，如 $4096 \times 4096$、$5000 \times 5000$）自動切為 $512 \times 512$（重疊 $64\text{px}$ / $12.5\%$）並跳過純黑背景塊；廠外公開集（$< 1536\text{px}$）自動維持整張原圖推論。
  * 自動同時偵測實體主機與 Docker cgroup (v1/v2) 記憶體上限，安全切換單次 Forward 全層收集（4 倍提速）或單層串流模式。
* **無損加速、四合一防誤報濾波 & 背景資源監控**：支援 TF32 / AMP (`bf16`/`fp16`)、多執行緒讀圖、降維畫布（`1024px`）多核心平行縫合與碎片 `.npy` 自動銷毀、四合一防誤報純化濾波，以及每秒自動記錄 RAM / GPU VRAM 峰值的背景監控程序。
* **CSV 自動化比對、母體維護與抽樣實驗工具組**：內建 5 支獨立 Python 腳本（`build_master_gt.py`、`recalc_metrics.py`、`calc_batch_distribution.py`、`sample_experiment.py`、`eval_dataset1.py`），以 CSV 全面取代人工檔名比對，涵蓋 FoundAD / MuSc-V2 重疊扣除、工程師複判一鍵重算、批次比例統計、Uniform Sampling 抽樣建集與最佳 Threshold 自動掃描（詳見[第六節](#六自動化評估母體維護與抽樣實驗工具組csv-流程)）。

---

## 一、環境與離線權重準備

1. **啟動 Conda 環境**：
   ```bash
   conda activate muscv2
   ```
   > 若使用第六節的自動化評估腳本，請確認環境已安裝 `pandas`（若尚未安裝可執行 `pip install pandas`）。
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

## 三、執行指令教學（標準兩步驟流程）

首次使用請先賦予腳本執行權限：

```bash
chmod +x test_mask_muscv2.sh run_inside_muscv2.sh run_outside_muscv2.sh
```

### 情境 A：公司廠內無 GT 實機盲池（標準兩步驟，免拆包直跑）

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

#### 步驟 2：啟動推論（`run_inside_muscv2.sh`，支援前景 `bash` 與背景 `nohup`）
改為 Position-wise 同座標分批互評後，**700 張與 2,770 張原始資料集皆可直接整包送入推論，無需再手動拆包**（執行大資料集前建議先以 `df -h` 確認硬碟剩餘空間）：

* **單一資料夾背景掛機推論（推薦，關閉 SSH 不中斷）**：
  ```bash
  nohup bash run_inside_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status > run_W639.log 2>&1 &
  ```
* **一次批次跑多個不同資料夾（各自自動載入對應的 `mask_<UNIQUE_TAG>.env`）**：
  ```bash
  nohup bash run_inside_muscv2.sh /workspace/dataset_use/W639/TMQB28/other_status /workspace/dataset_use/W642/TMQB28/other_status > run_multi.log 2>&1 &
  ```
* **即時監控進度與系統資源**：
  ```bash
  tail -f run_W639.log
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
└── wafer_escape_results/
    ├── latest_inside_muscv2_summary.txt          # [總表] 最新多盲池統計總結 (TXT)
    ├── latest_inside_muscv2_summary.csv          # [總表] 最新多盲池統計總結 (CSV)
    ├── latest_mem_monitor_inside.log             # [監控] 最新每秒 RAM / GPU VRAM 消耗紀錄
    ├── mask_configs/                             # [Mask 設定] 各資料夾專屬的 Mask .env 設定檔
    │   └── mask_<UNIQUE_TAG>.env
    ├── mask_preview_test/                        # [預覽] test_mask_muscv2.sh 產出的圓形 Mask 對照圖
    │   └── <YYYYMMDD_HHMMSS>/mask_preview/preview_*.png
    └── inside_no_gt_muscv2_<UNIQUE_TAG>/
        ├── latest_log.txt                        # 該盲池最新完整終端機日誌
        └── <YYYYMMDD_HHMMSS>/                    # 每次執行的獨立時間戳記目錄（不互相覆蓋）
            ├── log.txt                           # 本次執行完整日誌與耗時統計
            ├── tile_manifest.json                # 切塊座標、對齊旋轉角與圓心幾何快取紀錄
            ├── escape_ranking_report.txt         # [核心報表] 全盲池由高至低異常分數排名 (含門檻統計)
            ├── escape_ranking_report.csv         # [核心報表] CSV 格式完整分數明細 (含 Raw 與 Filtered 分數)
            ├── amap_npy/                         # [熱力圖矩陣] 僅儲存判定為異常的原圖方向 .npy
            └── anomaly_images/                   # [異常視覺化] 僅輸出超過門檻的異常晶圓（皆已逆轉回原圖角度）
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
| `ALIGN_MODE` | `"full360"` | Step 0 方向對齊模式：`full360`（任意角度對齊第一張圖）、`discrete90`（僅 $90^\circ$ 倍數）、`off`（關閉）。 |
| `MAX_PEER_TILES` | `0` *(或 `200`–`300`)* | 單座標互評時每個 Query 比較的他片圖塊上限。`0` 為 $100\%$ $N \times N$ 全體互評（適合 $\le 700$ 張）；**大資料集（如 2,770 張）建議設為 `150`～`300`**，最貼近原論文標準（Sec. IV-D9 為 150 張）且大幅提速。 |
| `AUTO_LOAD_MASK_CFG` | `1` | `1` = 自動讀取 `mask_configs/mask_<UNIQUE_TAG>.env`；`0` = 改用腳本內手動填寫的預設 Mask 參數。 |
| `PRE_MASK` | `""` | 推論前圓形背景遮罩預設開關（找不到 `.env` 或關閉自動載入時生效）。 |
| `PRE_MASK_MARGIN` | `0.02` | 圓形 Mask 半徑內縮比例（`0.02` = 內縮 $2\%$，負值如 `-0.01` 為外擴）。 |
| `MASK_CX` / `MASK_CY` / `MASK_R` | `-1` | 手動指定圓心與半徑：`-1` 為自動偵測；`0~1` 為原圖寬高/短邊比例；`>1` 為絕對像素值。 |
| `WAFER_EDGE_MODE` | `"auto"` | 後處理晶圓圓周倒角（Bevel）邊緣抑制：`auto`、`on`、`off`。 |
| `WAFER_EDGE_MARGIN` | `0.04` | 後處理圓周外圈平滑衰減寬度佔半徑比例（預設外圈 $4\%$）。 |
| `THRESH_MODE` | `"auto"` | 門檻模式：預設 `auto` 採用 Top 5% (P95) 作為畫框基準，並同步回報 Top 3% (P97) 切點參考。 |
| `MIN_TILING_RES` | `1536` | 大圖防呆門檻：圖片長邊 $\ge 1536\text{px}$ 才啟動切塊，低於此解析度自動整張推論。 |
| `TILE_SIZE` / `TILE_OVERLAP` | `512` / `64` | 切塊解析度與重疊寬度（$64 / 512 = 12.5\%$ 重疊）。 |
| `INFER_BATCH_SIZE` | `64` | 特徵提取 Batch Size（80 GB 顯卡可穩定維持 `64`；若顯存較小可降至 `16` 或 `8`）。 |
| `SPECKLE_FILTER_RATIO` | `0.008` | 形態學開運算濾除單一 Patch 微塵雜訊的比例（抓極微小顆粒時可降至 `0.003` 或 `0`）。 |
| `RSCON_WEIGHT` | `0.15` | 全晶圓級 RsCon 重評開關：`> 0` 啟用原論文 $k=7$ WMO 校準，`0` 關閉。 |

---

## 六、自動化評估、母體維護與抽樣實驗工具組（CSV 流程）

為解決廠內長檔名人工核對容易出錯的問題，本專案新增 5 支獨立 Python 自動化腳本，透過 CSV 串聯「**母體建置 → 工程師複判重算 → 批次比例統計 → Uniform Sampling 抽樣建集 → 推論自動對答案**」完整閉環。

### 1. 總體執行順序與相依關係（必看）

這 5 支腳本之間有明確的資料相依性，請務必依照以下階段順序執行：

```text
[階段一：初次建置母體總表（必須最先執行，僅初次建立時跑一次）]
  1. python build_master_gt.py
       └── 產出：master_gt.csv（已自動剝除 Rank 前綴並扣除 FoundAD / MuSc-V2 重疊）

              │
              ▼ （若工程師複判退回部分圖片，開啟 master_gt.csv 將該列 engineer_confirmed 改為 0 並存檔）

[階段二：母體指標重算 & 原批次分布統計（以下兩支皆依賴 master_gt.csv，彼此無先後順序限制）]
  2A. python recalc_metrics.py          ──▶ 產出：summary_metrics.csv（最新母體數、Overlap、雙模型 Recall）
  2B. python calc_batch_distribution.py ──▶ 產出：batch_distribution.csv（各批正常/瑕疵數與比例，供簡報使用）

              │
              ▼ （確認 master_gt.csv 為最新狀態後，再進行抽樣實驗）

[階段三：抽樣實驗與自動對答案閉環（以下 3 步須嚴格依序執行）]
  3. python sample_experiment.py
       └── 產出：exp_<MODE>/images/（抽樣圖檔）＋ exp_<MODE>/dataset1_gt.csv（標準答案卷）
  4. bash run_inside_muscv2.sh ./exp_<MODE>/images
       └── 產出：wafer_escape_results/.../<時間戳記>/escape_ranking_report.csv（模型推論分數報表）
  5. python eval_dataset1.py（將 GT_CSV 與 PRED_CSV 指向步驟 3 與步驟 4 的檔案）
       └── 產出：最佳 Threshold / Accuracy / F1 報表 ＋ dataset1_eval_detail.csv（逐張對答案明細）
```

| 執行順序 | 腳本名稱 | 前置必備檔案 / 條件 | 主要輸入 | 主要輸出 |
| :---: | :--- | :--- | :--- | :--- |
| **Step 1** | `build_master_gt.py` | 無（須先備妥各批原圖與雙模型瑕疵資料夾） | `raw_dir`、`foundad_dir`、`muscv2_dir` | `master_gt.csv` |
| **Step 2A** | `recalc_metrics.py` | `master_gt.csv`（已更新 `engineer_confirmed`） | `master_gt.csv` | `summary_metrics.csv` |
| **Step 2B** | `calc_batch_distribution.py` | `master_gt.csv` + 各批 `raw_dir` | `master_gt.csv`、`raw_dir` | `batch_distribution.csv` |
| **Step 3** | `sample_experiment.py` | `master_gt.csv` + 各批 `raw_dir` | `master_gt.csv`、`raw_dir` | `exp_<MODE>/images/`、`exp_<MODE>/dataset1_gt.csv` |
| **Step 4** | `run_inside_muscv2.sh` | Step 3 產出的 `exp_<MODE>/images/` | `exp_<MODE>/images/` | `escape_ranking_report.csv` |
| **Step 5** | `eval_dataset1.py` | Step 3 的 `dataset1_gt.csv` + Step 4 的 `escape_ranking_report.csv` | `dataset1_gt.csv`、`escape_ranking_report.csv` | `dataset1_eval_detail.csv` + 終端機指標報告 |

---

### 2. 各腳本詳細用法與設定說明

#### (1) `build_master_gt.py` — 自動還原原檔名、扣 Overlap 並建置母體總表
* **用途**：掃描各批次（`W639`、`W642`、`W646`）的 FoundAD 與 MuSc-V2 瑕疵資料夾，自動剝除 MuSc-V2 檔名前綴（`Rank001_Score0.8720_Box01_`）、子資料夾連接符（`__`）與副檔名，利用集合（Set）自動扣除重疊並建立 Dataset 2 瑕疵母體總表。
* **雙重防呆機制**：只要在設定區填入原始圖片資料夾 `raw_dir`，程式會優先比對乾淨檔名，若遇 FoundAD 或人工加過其他前後綴，會自動啟用「包含比對」救援；若仍完全對不上，終端機會印出 `⚠️ [警告]` 列出異常檔名，絕不默默算錯。
* **執行前修改（檔案頂部 `BATCH_CONFIG`）**：
  ```python
  BATCH_CONFIG = {
      "W639": {
          "raw_dir": "/path/to/W639_raw",               # 強烈建議填寫作防呆字典；暫無可填 None
          "foundad_dir": "/path/to/W639_foundad_defects",
          "muscv2_dir": "/path/to/W639_muscv2_defects", # 含 Rank 前綴免手動改名
      },
      # ... W642, W646 同理
  }
  ```
* **執行指令**：
  ```bash
  python build_master_gt.py
  ```
* **輸出檔案**：`master_gt.csv`（UTF-8 BOM 編碼，Excel 開啟不亂碼），欄位包含：
  * `batch`：批次名稱（如 `W639`）
  * `filename`：不含副檔名的最乾淨原檔名
  * `foundad_hit` / `muscv2_hit`：`1` 代表該模型有檢出、`0` 代表未檢出
  * `engineer_confirmed`：預設全部填 `1`；日後工程師退回誤報時，改為 `0` 即可
* **⚠️ 重要提醒**：若您已開始手動在 `master_gt.csv` 維護工程師複判結果（將部分列改成 `0`），**請勿再次執行 `build_master_gt.py` 覆蓋原檔**，否則 `engineer_confirmed` 欄位會全數重設回 `1`。

---

#### (2) `recalc_metrics.py` — 工程師複判更新後一鍵重算 Recall
* **用途**：當工程師回饋剔除部分誤報（例如 W639 母體從 49 張修正為 40 張）時，自動重算各批及跨批總計（`ALL (總計)`）的最新母體數、退回張數、雙模型獨有張數、Overlap 與 Recall。
* **日常更新 SOP**：
  1. 使用 Excel 或 VS Code 開啟 `master_gt.csv`，將工程師認定非瑕疵的圖片其 `engineer_confirmed` 欄位從 `1` 改為 `0` 並存檔（腳本內建數值轉型防呆，可容錯 Excel 存檔格式）。
  2. 執行重算：
     ```bash
     python recalc_metrics.py
     ```
* **輸出檔案**：`summary_metrics.csv`（欄位：`Batch`, `原候選總數`, `工程師退回(0)`, `最新母體數(1)`, `FoundAD獨有`, `MuSc-V2獨有`, `Overlap重疊`, `FoundAD抓到`, `FoundAD_Recall`, `MuSc-V2抓到`, `MuSc-V2_Recall`），可直接貼入進度簡報表格。

---

#### (3) `calc_batch_distribution.py` — 計算各批次真實正常／瑕疵比例
* **用途**：遞迴統計各批 `raw_dir` 原始總張數，搭配 `master_gt.csv` 中最新確認的真瑕疵張數（`engineer_confirmed == 1`），自動算出每批及三批合計的正常張數、瑕疵張數、百分比與 `正常:瑕疵比`（例如 `25.3 : 1`），供簡報說明各批真實分布與設計抽樣比例。
* **執行前修改（檔案頂部 `RAW_DIRS`）**：
  ```python
  MASTER_GT_CSV = "master_gt.csv"
  RAW_DIRS = {
      "W639": "/path/to/W639_raw",
      "W642": "/path/to/W642_raw",
      "W646": "/path/to/W646_raw",
  }
  ```
* **執行指令**：
  ```bash
  python calc_batch_distribution.py
  ```
* **輸出檔案**：`batch_distribution.csv`（欄位：`Batch`, `總張數`, `正常張數`, `瑕疵張數`, `正常比例`, `瑕疵比例`, `正常:瑕疵比`）。

---

#### (4) `sample_experiment.py` — Uniform Sampling 自動抽樣建集與產出答案卷
* **用途**：讀取 `master_gt.csv` 中 `engineer_confirmed == 1` 的最新確認瑕疵，將各批 `raw_dir` 自動切分為「瑕疵池」與「正常池」，以固定隨機種子（預設 `RANDOM_SEED = 42`）執行無放回均勻抽樣（Uniform Sampling），自動將圖片複製到獨立實驗資料夾並同步產出 `dataset1_gt.csv`，完全免人工挑圖與貼檔名。
* **執行前修改（檔案頂部 `RAW_DIRS` 與 `ACTIVE_MODE`）**：
  只要切換 `ACTIVE_MODE` 字串即可快速產出不同比例的實驗集（亦可在 `EXPERIMENT_PRESETS` 內自訂張數，`"defect": "all"` 代表全數取用該批已確認瑕疵）：

  | `ACTIVE_MODE` 模式名稱 | W639 (瑕疵/正常) | W642 (瑕疵/正常) | W646 (瑕疵/正常) | 總張數與實驗目的 |
  | :--- | :---: | :---: | :---: | :--- |
  | `"1_to_1"` | 5 / 5 | 5 / 5 | 5 / 5 | 共 30 張（15 瑕疵 : 15 正常），原始 1:1 基準實驗 |
  | `"7_to_3"` | 5 / 12 | 5 / 12 | 5 / 11 | 共 50 張（15 瑕疵 : 35 正常），測試 30% : 70% 比例 |
  | `"100_normal_each"` *(預設)* | 5 / 100 | 5 / 100 | 5 / 100 | 共 315 張（15 瑕疵 : 300 正常），正常樣本放大 20 倍驗證 Threshold |
  | `"dataset2_all_defects"` | all / 100 | all / 100 | all / 100 | 全部最新確認瑕疵 + 各批均勻抽正常，驗證「跨批混合 vs. 各批獨立推論」 |

* **執行指令**：
  ```bash
  python sample_experiment.py
  ```
* **輸出結構**：自動依模式名稱建立 `exp_<ACTIVE_MODE>/`（不同模式互不覆蓋；重複執行同模式時會先清空該模式的 `images/` 避免殘留舊圖）：
  ```text
  exp_<ACTIVE_MODE>/
  ├── images/            # 抽出的所有實驗圖片（直接丟給 run_inside_muscv2.sh 跑推論）
  └── dataset1_gt.csv    # 對應標準答案卷（欄位：batch, filename, gt_label；1=瑕疵, 0=正常）
  ```

---

#### (5) `eval_dataset1.py` — 推論結果自動對答案與最佳 Threshold 掃描
* **用途**：讀取 `dataset1_gt.csv` 與 MuSc-V2 推論完畢後產出的 `escape_ranking_report.csv`（使用 `Original_Filename` 與 `Max_Score` 欄位），自動剝除路徑與副檔名完成 100% 檔名對齊，逐一掃描所有候選 Threshold，輸出**最高 Accuracy 切點**與**最高 F1-Score 切點**（在 7:3 或正常放大 20 倍的不平衡實驗中，建議同時參考 F1 切點避免 Accuracy 失真）。
* **執行前修改（檔案頂部路徑）**：
  ```python
  GT_CSV = "exp_100_normal_each/dataset1_gt.csv"  # 對應 Step 3 產出的答案卷
  PRED_CSV = "wafer_escape_results/inside_no_gt_muscv2_.../YYYYMMDD_HHMMSS/escape_ranking_report.csv"
  OUTPUT_DETAIL_CSV = "dataset1_eval_detail.csv"
  ```
* **執行指令**：
  ```bash
  python eval_dataset1.py
  ```
* **輸出內容**：
  1. **終端機報表**：成功比對總張數（若有遺漏會印出 `⚠️ [警告]`）、最高 Accuracy 與最高 F1 兩組切點的 Accuracy / Precision / Recall / F1 與混淆矩陣（TP / FP / TN / FN），以及依 `batch` 分組的各批次細部表現。
  2. **`dataset1_eval_detail.csv`**：依 `Max_Score` 由高到低排序的逐張對答案明細（含 `pred_label` 與 `is_correct`），方便直接篩選 `is_correct == False` 追查誤報（FP）與漏抓（FN）。

---

## 七、硬體資源評估與突發狀況排解（FAQ）

### Q1：跑 700 張或 2,770 張 $5000 \times 5000$ 大圖，CPU RAM、GPU VRAM 或硬碟會爆嗎？
* **CPU RAM（安全）**：採用 Position-wise 同座標分批互評後，700 張峰值約 **12–15 GiB**，2,770 張峰值約 **42–45 GiB**，在 128 GiB Docker 容器內皆有充足安全餘裕。
* **GPU VRAM（安全）**：互評採分塊送入顯卡，VRAM 峰值約 **10–20 GB**。
* **硬碟空間（開跑前必查）**：切塊預設採無壓縮 PNG（`compress_level=0`）以極大化讀寫速度：
  * 700 張（約 8.9 萬塊）：`./datasets/*_tiled_pool` 約佔 **68 GB**。
  * 2,770 張（約 35 萬塊）：約佔 **260–300 GB**。
  * 執行前請先用 `df -h` 確認 `./datasets` 所在磁碟空間足夠；若空間吃緊，將 `1_infer_muscv2_no_gt.py` 中的 `compress_level=0` 改為 `compress_level=1` 即可將硬碟佔用砍半以上。

### Q2：如果資料集裡正常圖跟瑕疵圖是 1:1（甚至幾乎都是瑕疵圖），MuSc-V2 還抓得到嗎？
* **抓得到**。原論文（Sec. IV-D5, Fig. 10）已證實即使正常樣本降到 0 張，分割指標 AUROC 仍僅掉不到 $0.23\%$，因為每張工業影像仍有 $> 99\%$ 的像素是正常的。
* **唯一需注意**：高瑕疵率批號中，模型依然會把瑕疵圖穩穩排在 `escape_ranking_report.csv` 前半段，但預設的 Top 5% (P95) 門檻只會畫出前 $5\%$ 的紅框；若批號真實瑕疵率 $> 5\%$，請檢視 CSV 分數分佈並改以固定門檻重新切分（或直接交由 `eval_dataset1.py` 掃描最佳切點）。

### Q3：自動偵測的圓形 Mask 圓心歪掉，或切到晶圓有效區域 / 留到背景怎麼辦？
* **解決方法**：重跑 `test_mask_muscv2.sh` 微調參數，新的設定會自動覆蓋舊的 `.env` 檔：
  * **切太裡面（吃掉晶圓邊緣）**：調小 `MARGIN`（可設負值如 `-0.01`）或調大 `R`。
  * **切太外面（留到背景雜訊）**：調大 `MARGIN`（如 `0.03`）或調小 `R`。
  * **黃十字圓心偏移**：直接手動傳入 `CX CY R`（支援 `0~1` 比例或 `>1` 像素），程式會先在未旋轉原圖上套用 Mask 再做 Step 0 旋轉對齊，不會發生圓心偏移。

### Q4：公司內網無法連上網際網路，報錯 `URLError` 或找不到權重檔怎麼辦？
* **解決方法**：將離線的 `dinov2_vitl14_pretrain.pth`（或 `dinov3` 權重檔）複製到 `MuSc-V2/` 專案根目錄下，或在 `run_inside_muscv2.sh` 直接指定 `DINO_CKPT_PATH` 的絕對路徑。

### Q5：機台拍出來的 `.tif` / `.tiff` 是 16-bit 灰階圖，讀進來全黑或對比度異常？
* **說明**：程式內建 `_robust_normalize_to_uint8()`，會自動針對 `I;16` / 單通道灰階 / 浮點數 TIFF 採用 **P0.1 ~ P99.9 穩健百分位數拉伸** 轉為 8-bit RGB，防止極端飽和亮點壓扁整體對比。

### Q6：有真實微小的刮傷或 Particle 漏抓（False Negative）怎麼辦？
* **解決方法**：
  1. 調小或關閉微塵開運算濾波（避免把真實微缺陷當噪點濾掉）：設 `SPECKLE_FILTER_RATIO=0.003`（或 `0`）、`MIN_AREA_RATIO=0.00005`。
  2. 加上 `RAW_SCORE_MODE="--raw_score_mode"`，查看 `escape_ranking_report.csv` 內的 `Raw_Max_Score` 確認原始模型是否有偵測到該缺陷。

### Q7：替換了盲池資料夾內的圖片，但腳本卻顯示 `[⚡ 測試池清單快取命中]` 沿用舊資料？
* **說明**：快取指紋會比對所有圖片的「相對路徑 + 檔案大小」以及對齊、切塊與 Mask 參數。若僅替換了同名且檔案大小完全相同的圖片內容，請在 `run_inside_muscv2.sh` 設定 `FORCE_INFER="--force_infer"` 執行一次刷新快取（跑完記得改回 `FORCE_INFER=""`）。

### Q8：為什麼 `anomaly_images/` 裡只有少數圖片，其他正常圖沒輸出？
* **說明**：這是**正常的防呆省空間設計**。只有分數 $\ge$ `Effective Threshold` 且通過面積過濾的**異常晶圓**才會輸出紅框圖與 `.npy`；所有圖片（含正常圖）的完整分數與排名皆記錄在 `escape_ranking_report.csv` 與 `escape_ranking_report.txt` 中。

### Q9：跑 `eval_dataset1.py` 時跳出「有 N 張 GT 圖片找不到對應分數」怎麼辦？
* **排查步驟**：
  1. 檢查 `GT_CSV` 與 `PRED_CSV` 是否指向**同一組實驗模式**（例如 `exp_100_normal_each/dataset1_gt.csv` 必須配對 `exp_100_normal_each/images` 跑出來的 `escape_ranking_report.csv`）。
  2. 確認 `sample_experiment.py` 抽樣後沒有手動刪除或改名 `images/` 裡的檔案。

### Q10：工程師後續又給了新的瑕疵複判結果，哪些腳本需要重跑？
* **處理方式**：**不需要重跑 `build_master_gt.py`**。只需修改 `master_gt.csv` 內的 `engineer_confirmed` 欄位並存檔，接著依需求執行：
  * 更新母體數與 Recall ➔ 重跑 `recalc_metrics.py`
  * 更新各批正常／瑕疵比例 ➔ 重跑 `calc_batch_distribution.py`
  * 以最新真瑕疵清單重跑抽樣實驗 ➔ 依序重跑 `sample_experiment.py` → `run_inside_muscv2.sh` → `eval_dataset1.py`
