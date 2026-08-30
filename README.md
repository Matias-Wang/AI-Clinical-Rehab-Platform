# AI Clinical Rehab Platform（智慧臨床復健動作辨識平台）

## 簡介 (Tagline)

基於穿戴式感測器的輕量化 AI 動作分類系統，透過 LiteCNN 模型辨識 12 種日常復健動作，目標應用於長照防跌偵測與穿戴式健康監測。

---

## 專案說明

### 問題背景

現有復健監測系統依賴人工觀察，無法做到即時、客觀的動作量化辨識。靜態動作（Standing vs. Sitting）與動態動作（Walking vs. Climbing Stairs）的感測器特徵高度相似，傳統時域特徵難以準確區分。

### 本專案的解決方案

- 以 mHealth Dataset（10 位受試者，50Hz 穿戴式感測器）作為核心訓練資料。
- 開發 **LiteCNN** 輕量化模型，適合 Edge AI 部署。
- 透過**混合域特徵工程**（時域 + 頻域 FFT），解決靜態/動態動作混淆問題（S2 坐姿召回率由 2% 提升至 98%）。
- 採用 LOSO（Leave-One-Subject-Out）交叉驗證，確保模型跨受試者泛化能力。

---

## 使用架構 / 技術

| 層次 | 技術 |
|------|------|
| 語言 | Python 3.13 |
| 套件管理 | UV |
| 深度學習框架 | TensorFlow / Keras |
| 數值計算 | NumPy、SciPy |
| 資料處理 | Pandas |
| 視覺化 | Matplotlib、Seaborn |
| 模型類型 | **LiteCNN（CNN 架構）**（Generation 4） |
| 特徵空間 | **8 維混合域**（時域 7D + FFT 頻域 1D） |
| 標準化策略 | **三段式**：Acc Z-score + Gravity ÷10 + FFT log1p/5 |
| 驗證協議 | **10-Fold LOSO** 交叉驗證（Mean Acc: 85.08%） |
| 資料集 | mHealth Dataset（10 受試者 × 12 動作類別） |

---

## 快速開始 (Quick Start)

### 環境建置

```bash
# 建立虛擬環境（使用 UV）
uv venv
source .venv/bin/activate      # Linux/macOS
.\.venv\Scripts\activate       # Windows

# 安裝依賴
uv pip install -r requirements.txt
```

### 資料準備

將 mHealth Dataset 的 `.log` 檔案放置於 `data/` 資料夾：

```
data/
  mHealth_subject1.log
  ...
  mHealth_subject10.log
```

### 執行批量驗收測試（最新版本）

```bash
python main.py
```

`main.py` 為系統主入口，支援：
- **單一受試者模擬會期**：測試單一受試者即時推論流程
- **S1–S10 批量自動化測試**：驗收全量受試者數據

> 歷史實驗訓練ipynb檔存放於 `development_history/`。

### 啟動即時生物回饋展示

```bash
# 啟動 WebSocket 橋接伺服器（預設 S10、埠 8765、20 倍速）
python ui_bridge.py --subject 10 --port 8765 --speed 20

# 伺服器啟動後，以瀏覽器開啟 frontend/demo.html 觀看即時紅/黃/綠燈號
```

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--subject` | 模擬串流的受試者編號 | 10 |
| `--port` | WebSocket 監聽埠 | 8765 |
| `--speed` | 播放加速倍率 | 20.0 |
| `--realtime` | 以真實 50Hz 播放（等同 `--speed 1.0`） | — |

### 執行回歸測試

```bash
# 全套測試（73 項，含需要 TensorFlow 的準確率基準）
pytest

# 僅快速測試（約 3 秒，免載入 TensorFlow）
pytest -m "not slow"
```

測試涵蓋 Fix 1/2/3 等致命修正的回歸保護、品質閘門的 NaN fail-safe、串流緩衝區復原與 DTW 數學性質。修改預處理管線或推論路徑後，測試必須維持全綠。

---

## 目標使用者

- **臨床復健工程師**：需要客觀、自動化的動作辨識工具。
- **AI / MLOps 研究人員**：研究 Edge AI 部署與穿戴式感測器的時序分類模型。
- **長照科技開發者**：建構防跌偵測與健康監測解決方案。

---

## 功能特色 (Features)

### 核心功能

- **多受試者資料處理管線**：`schema.py` 提供統一的資料讀取、清洗、特徵提取與視窗化介面。
- **物理邊界清洗**：自動剔除超出人體物理極限（> 50 m/s²）與坐姿異常尖峰（> 10.8 m/s²）的雜訊點。
- **重力分量分離**：Butterworth 低通濾波器（零相位 `filtfilt`）分離靜態重力向量，提供軀幹傾斜角度資訊。
- **FFT 頻域特徵**：計算滑動視窗內的頻譜能量，區分靜止與動態動作。
- **選擇性標準化**：前 4 維（Acc/Mag）執行受試者級別 Z-score；後 3 維（Gravity）執行全局縮放（÷10），保留物理角度參考值。
- **LOSO 交叉驗證**：按受試者分割訓練/測試集，模擬臨床部署情境。
- **臨床品質閘門 (Clinical Quality Gate)**：`ClinicalQualityGate` 類別，基於 Grav_Y 變異數閾值（≥ 0.0005）攔截低品質動作資料，防止治療師集群效應（Therapist Clustering）導致的偽陽性。採 Data-Centric AI 路線，取代後端數學補償策略。
- **即時串流處理器 (RealTimeStreamProcessor)**：`schema.py` 內建支援 50% 重疊率（Stride=64）的滑動視窗緩衝區，模擬穿戴裝置即時數據流。內建影格完整性驗證與緩衝區復原機制，攔截 NaN/Inf、維度錯誤與硬體雜訊，並在連續髒數據時判定感測器斷訊。
- **即時生物回饋引擎 (RealTimeBiofeedbackEngine)**：整合品質閘門與 DTW 相似度評分的即時決策中心，輸出 紅/黃/綠 三色 UI 引導狀態。
- **DTW 姿態相似度**：以 Dynamic Time Warping（Sakoe-Chiba band，radius=16）取代點對點的歐幾里德距離，容忍動作節奏差異——姿勢正確但速度略慢的患者不再被誤判為做錯。
- **即時資料橋接 (WebSocket)**：`ui_bridge.py` 將推論結果即時 broadcast 給前端，`frontend/demo.html` 提供無框架的最小展示頁面。
- **低延遲即時推論**：以 `predict_on_batch()` 執行單視窗推論（中位數 2.49 ms），相較批次導向的 `predict()`（94.91 ms）快約 38 倍且無記憶體累積，輸出逐位元相同。
- **自動化回歸測試**：`tests/` 提供 73 項測試，釘死 Fix 1/2/3 等致命修正與推論路徑選擇，並以突變測試驗證測試本身的鑑別力。

### 實驗筆記本

| 筆記本 | 對應世代 | 主要內容 |
|--------|----------|---------|
| `20260404_Project_Rehab_Optimization.ipynb` | Generation 4 (Current) | 治療師集群效應診斷、壓力測試、座標對齊實驗、臨床品質閘門 |
| `20260403_Project_Rehab_FFT.ipynb` | Generation 3 | 混合域特徵、FFT 頻譜、S2 修復 |
| `development_history/20260307_Project_Rehab_Training_LOSO.ipynb` | Generation 2 | 重力特徵、LOSO 驗證 |
| `development_history/20260228_Project_Rehab_Diagnostics.ipynb` | Generation 2 | 一致性檢定（KS Test）、物理稽核 |
| `development_history/20260222_Project_Rehab_Consistency_Test.ipynb` | Generation 1 | 基礎一致性測試 |

---

## 專案架構 (Project Structure)

```
AI-Clinical-Rehab-Platform/
├── schema.py                              # 核心管線（特徵工程、品質閘門、即時引擎）
├── main.py                                # 系統主入口（單次/批量驗收測試）
├── ui_bridge.py                           # WebSocket 即時資料橋接伺服器
├── console.py                             # UTF-8 輸出設定（避免 cp950 編碼崩潰）
├── requirements.txt                       # 依賴套件清單
├── pytest.ini                             # 測試設定
│
├── tests/                                 # 自動化回歸測試（73 項）
│
├── frontend/                              # 最小展示前端
│   └── demo.html                          # 自包含 HTML/JS，顯示即時燈號與評分
│
├── data/                                  # mHealth Dataset 原始資料（未納入版控）
│   ├── mHealth_subject1.log ~ subject10.log
│   └── data_raw_Info.md                   # 資料集說明文件
│
├── models/                                # 訓練完成的模型檔案
│   ├── clinical_rehab_model_v1.keras      # Generation 1 歷史模型（保留供追溯）
│   └── clinical_rehab_model_v3.keras      # Generation 4 當前模型
│
├── confusion_matrix/                      # 混淆矩陣圖
│
├── docs/                                  # 參考文獻（SaMD、AI 醫療法規）
│
├── development_history/                   # 歷史實驗 Notebook（未納入版控）
│
├── README.md
├── ARCHITECTURE.md
├── SPEC.md
├── CHANGELOG.md
├── ROADMAP.md                             # 交接說明書與研發計畫
└── Development_Log.md                     # 研發歷程紀錄
```

---

## 作者資訊

- 作者：Matias Wang
- Email：tzuanwork903@gmail.com
- GitHub：[Matias-Wang](https://github.com/Matias-Wang/AI-Clinical-Rehab-Platform)
