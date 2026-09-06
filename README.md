# AI Clinical Rehab Platform（智慧臨床復健動作辨識平台）

## 簡介 (Tagline)

基於穿戴式感測器的**即時臨床復健生物回饋平台**。系統以 50Hz 加速度訊號辨識
12 種復健動作，並在動作品質不足時**主動拒絕給分**，透過紅／黃／綠三色燈號
即時引導患者，而非事後才產出一份無法信任的報告。

---

## 這個專案在做什麼

復健動作的療效取決於「做得對不對」，但現行監測依賴治療師人工觀察，
無法即時、客觀地量化。本平台以穿戴式感測器的加速度訊號，在患者動作當下
給出三色燈號回饋。

系統要處理兩個難題：

1. **訊號相似** —— 靜態動作（Standing vs. Sitting）與動態動作
   （Walking vs. Climbing Stairs）的時域特徵高度重疊。系統以
   **8 維混合域特徵**（時域 7D + FFT 頻域 1D）區分。
2. **資料品質不均** —— 感測器配戴方式與引導手法的差異，會讓部分受試者的
   關鍵運動平面訊號嚴重不足。此時任何模型的輸出都不可信。

### 核心設計：品質閘門先於推論

面對第二個難題，本專案採 **Data-Centric AI** 路線，而非以演算法補償訊號缺陷。
系統的第一道關卡是**臨床品質閘門（Clinical Quality Gate）**：

> 動作幅度不足時，系統**拒絕預測**，直接亮紅燈提示「動作幅度不夠，請加大」。

寧可不輸出，也不輸出基於無效資料的結果 —— 一個沉默的系統，優於一個自信地
給出錯誤評分的系統。這也是為什麼在驗收報表中，低活動組的「品質合格率」
只有個位數：那是系統正確地攔截了無效訊號，而非效能不佳。

### 系統運作流程

```
穿戴式感測器 (50Hz)
        │
        ▼
  8 維混合域特徵          時域 7D（三軸加速度、合力、三軸重力分量）
  （128 步滑動視窗）       + 頻域 1D（FFT 頻譜能量）
        │
        ▼
  ① 臨床品質閘門          檢查 Grav_Y 變異數是否足夠
        │                 不足 ──► 🔴 紅燈「動作幅度嚴重不足，AI 停止預測」
        │                 訊號含 NaN/Inf ──► 🔴 紅燈「感測器異常」
        ▼
  ② AI 動作辨識           LiteCNN 判斷這是 12 種復健動作中的哪一種
        │
        ▼
  ③ DTW 姿態比對          與黃金範本比對動作軌跡
        │                 採 Dynamic Time Warping（Sakoe-Chiba band,
        │                 radius=16），只評估軌跡是否正確，容忍時間軸伸縮，
        │                 避免把「節奏較慢」誤判為「動作做錯」
        ▼
  ④ 綜合評分與燈號        Final = 品質 × 0.4 + 相似度 × 0.6
                          🟢 動作標準  🟡 需要修正  🔴 無效動作
```

評分結果透過 WebSocket 即時推播至前端，患者在動作當下就能看到燈號回饋。

> 各項設計決策的實驗依據與演進過程，記錄於 [Development_Log.md](Development_Log.md)；
> 版本層級的變更記錄見 [CHANGELOG.md](CHANGELOG.md)。

---

## 技術棧

| 層次 | 技術 |
|------|------|
| 語言 | Python 3.13 |
| 套件管理 | UV |
| 深度學習框架 | TensorFlow 2.20 / Keras 3.13 |
| 數值計算 | NumPy、SciPy |
| 資料處理 | Pandas、scikit-learn（StandardScaler） |
| 即時通信 | asyncio + websockets |
| 前端 | 原生 HTML/JS（無框架、無 build tooling） |
| 測試 | pytest（118 項，分五層 marker） |
| 模型類型 | **LiteCNN（CNN 架構）**（Generation 4） |
| 特徵空間 | **8 維混合域**（時域 7D + FFT 頻域 1D） |
| 標準化策略 | **三段式**：Acc/Mag 受試者級 Z-score + Gravity ÷10 + FFT log1p/5 |
| 品質控制 | ClinicalQualityGate（Grav_Y 變異數門檻 0.0005） |
| 姿態比對 | DTW（Sakoe-Chiba band, radius=16） |
| 驗證協議 | **10-Fold LOSO** 交叉驗證（Mean Acc: 85.08%） |
| 資料集 | mHealth Dataset（10 受試者 × 12 動作類別） |

---

## 技術架構

系統採**關注點分離**：可重用的資料處理與決策邏輯集中於 `schema.py`，
兩個入口點（批量驗收與即時橋接）共用同一套管線，確保兩者行為一致。

```
┌──────────────────────────────┬──────────────────────────────┐
│    批量驗收 (main.py)         │   即時橋接 (ui_bridge.py)     │
│    S01–S10 自動化報表         │   WebSocket 逐視窗推播        │
└──────────────┬───────────────┴───────────────┬──────────────┘
               │        共用 evaluate_window()  │
┌──────────────▼───────────────────────────────▼──────────────┐
│                   核心引擎層 (schema.py)                     │
│  ClinicalQualityGate          品質閘門 + NaN/Inf fail-safe   │
│  RealTimeBiofeedbackEngine    推論 + DTW 相似度 + 評分決策    │
│  RealTimeStreamProcessor      滑動視窗緩衝 + 髒數據攔截       │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                   資料管線層 (schema.py)                      │
│  讀取 → 物理清洗 → 重力分離 → 視窗化 → FFT → Z-score         │
└──────────────────────────┬───────────────────────────────────┘
                           │
              mHealth Dataset (10 受試者 × 50Hz)
```

### 核心模組

| 模組 | 職責 |
|------|------|
| `schema.py` | 資料管線與即時引擎。所有特徵工程與臨床決策邏輯的單一來源 |
| `main.py` | 批量驗收入口。逐視窗評估 S01–S10 並產出報表（進度訊息走 stderr，報表走 stdout） |
| `ui_bridge.py` | WebSocket 伺服器。依真實時間節奏逐視窗推播決策結果 |
| `console.py` | UTF-8 輸出設定，避免 Windows cp950 環境下輸出被重導時崩潰 |
| `verify.py` | 驗證閘門單一入口，五層檢查 |
| `frontend/demo.html` | 最小展示前端，顯示燈號、評分與攔截原因 |

### 驗證機制

專案以 `python verify.py` 作為單一驗證入口，共五層自動化閘門：

| 層 | 內容 |
|----|------|
| 單元與回歸測試 | 髒數據防護、緩衝區復原、DTW 數學性質、預處理修正 |
| 規範與一致性 | 程式碼規範、文件漂移偵測、notebook 相容性守門 |
| 模型準確率基準 | 對照訓練 notebook 的 S02/S07/S09/S10 準確率 |
| 驗收輸出平價 | `main.py` 輸出與 golden 基準逐行比對 |
| 效能煙霧測試 | 推論延遲與串流速率門檻 |

其中兩項針對本專案的特性設計：**文件漂移偵測**會解析 SPEC.md 的 API 表格，
斷言記載的每個符號都真實存在於程式碼中；**notebook 相容性守門**則確保
`development_history/` 的訓練 notebook 所引用的 `schema.py` 符號不會被誤刪
（部分符號在生產程式碼中沒有呼叫點，靜態掃描容易誤判為死碼）。

```bash
python verify.py          # 每次修改後，約 15 秒
python verify.py --full   # commit 前，約 1 分鐘
```

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

`main.py` 為批量驗收入口，逐視窗評估 S01–S10 全量受試者資料並產出報表。
進度訊息輸出至 stderr、報表輸出至 stdout，因此 `python main.py > report.txt`
會得到乾淨的報表檔。

實際輸出：

```
========================================================================
受試者      | 總視窗數     | 品質合格率      | L7 視窗數    | L7 召回率
------------------------------------------------------------------------
S01       | 548        |      5.5%    | 0           |        —
S02       | 554        |      8.7%    | 0           |        —
S03       | 551        |     23.0%    | 2           |   100.0%
...
S08       | 519        |     21.6%    | 15          |    93.3%
S10       | 525        |     22.5%    | 3           |    66.7%
========================================================================
```

**如何解讀**：品質合格率低是**正確行為**——S01/S02 的動作幅度大多不足以
支撐可信的 AI 判斷，系統選擇攔截而非硬給分數。L7 召回率則反映通過閘門後
模型的實際辨識能力（S08 的 15 個 Label 7 視窗中答對 14 個）。
顯示「—」代表該受試者沒有任何 Label 7 視窗通過品質閘門。

> 歷史實驗訓練 ipynb 檔存放於 `development_history/`。

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
# 全套測試（118 項，含需要 TensorFlow 的準確率基準）
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
- **臨床品質閘門 (Clinical Quality Gate)**：基於 Grav_Y 變異數閾值（≥ 0.0005）攔截低品質動作資料，避免系統對訊號不足的動作給出不可信評分。另含 NaN/Inf fail-safe，訊號含無效值時一律攔截。
- **即時串流處理器 (RealTimeStreamProcessor)**：`schema.py` 內建支援 50% 重疊率（Stride=64）的滑動視窗緩衝區，模擬穿戴裝置即時數據流。內建影格完整性驗證與緩衝區復原機制，攔截 NaN/Inf、維度錯誤與硬體雜訊，並在連續髒數據時判定感測器斷訊。
- **即時生物回饋引擎 (RealTimeBiofeedbackEngine)**：整合品質閘門與 DTW 相似度評分的即時決策中心，輸出 紅/黃/綠 三色 UI 引導狀態。
- **DTW 姿態相似度**：以 Dynamic Time Warping（Sakoe-Chiba band，radius=16）比對動作軌跡，容忍時間軸伸縮，使「姿勢正確但速度較慢」不會被判為做錯。
- **即時資料橋接 (WebSocket)**：`ui_bridge.py` 依真實時間節奏逐視窗推播推論結果（`--realtime` 下每 1.28 秒一個視窗），`frontend/demo.html` 提供無框架的最小展示頁面，顯示紅/黃/綠燈號與攔截原因。
- **低延遲即時推論**：單視窗推論中位數 2.49 ms、長時間運行無記憶體累積，足以支撐 50Hz 即時串流。
- **自動化回歸測試**：`tests/` 提供 118 項測試，涵蓋特徵管線正確性、品質閘門 fail-safe、串流韌性、文件一致性與效能門檻，並以 `verify.py` 分五層執行。

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
├── tests/                                 # 自動化回歸測試（118 項）
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

**相關背景**：具醫療專案管理（Cathay General Hospital PM）與 AI 產品架構
（CTBC AI Architect）經驗，並具備統計學專業基礎。本專案的臨床框架設計
——以品質閘門優先於模型效能——即源自醫療場域的實務考量。
