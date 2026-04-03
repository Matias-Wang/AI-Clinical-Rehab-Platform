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
| 模型類型 | LiteCNN (Conv1D) |
| 特徵空間 | 8 維混合域（時域 7D + FFT 頻域 1D） |
| 驗證協議 | LOSO 交叉驗證、Stratified Train/Test Split |
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

### 執行訓練（最新版本）

開啟並執行 `20260403_Project_Rehab_FFT.ipynb`（Generation 3，混合域特徵版本）。

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

### 實驗筆記本

| 筆記本 | 對應世代 | 主要內容 |
|--------|----------|---------|
| `20260403_Project_Rehab_FFT.ipynb` | Generation 3 (Candidate) | 混合域特徵、FFT 頻譜、S2 修復 |
| `development_history/20260307_Project_Rehab_Training_LOSO.ipynb` | Generation 2 | 重力特徵、LOSO 驗證 |
| `development_history/20260228_Project_Rehab_Diagnostics.ipynb` | Generation 2 | 一致性檢定（KS Test）、物理稽核 |
| `development_history/20260222_Project_Rehab_Consistency_Test.ipynb` | Generation 1 | 基礎一致性測試 |

---

## 專案架構 (Project Structure)

```
AI-Clinical-Rehab-Platform/
├── schema.py                          # 核心資料處理管線（特徵工程、濾波、視窗化）
├── 20260403_Project_Rehab_FFT.ipynb   # 最新實驗：Generation 3 混合域特徵
├── requirements.txt                   # 依賴套件清單
│
├── data/                              # mHealth Dataset 原始資料
│   ├── mHealth_subject1.log ~ subject10.log
│   └── data_raw_Info.md               # 資料集說明文件
│
├── models/                            # 訓練完成的模型檔案
│   └── clinical_rehab_model_v1.keras
│
├── architecture/                      # 架構圖與評估結果
│   └── confusion_matrix/              # 混淆矩陣圖
│       ├── confusion_matrix_v3_multi.png
│       ├── confusion_matrix_v3_1_multi.png
│       ├── S2_Confusion_Matrix_FFT_Acc.png
│       └── S2_Confusion_Matrix_FFT_Acc_0.78.png
│
├── Spec/                              # 技術規格文件
│   └── v3_1_multi_Technical_Spec.md
│
├── docs/                              # 參考文獻（SaMD、AI 醫療法規）
│
├── development_history/               # 歷史實驗筆記本存檔
│   ├── 20260222_Project_Rehab_Consistency_Test.ipynb
│   ├── 20260228_Project_Rehab_Diagnostics.ipynb
│   └── 20260307_Project_Rehab_Training_LOSO.ipynb
│
├── README.md
├── ARCHITECTURE.md
├── SPEC.md
├── CHANGELOG.md
├── 檔案說明.md
├── CLAUDE.md
└── GEMINI.md
```

---

## 未來優化 (Future Work)

- **架構升級**：評估引入 Transformer (Self-Attention) 機制以處理長序列動作關聯。
- **數據增強**：針對小樣本動作類別（如 Jump，< 50 筆）實作 Data Augmentation。
- **產品化**：結合 AWS Cloud 架構與醫療法規 (SaMD)，設計自動化 MLOps 流程。
- **驗證協議強化**：全面採用 10-Fold LOSO 交叉驗證取代隨機切分。

---

## 作者資訊

- 作者：Matias Wang
- Email：tzuanwork903@gmail.com
- GitHub：[Matias-Wang](https://github.com/Matias-Wang/AI-Clinical-Rehab-Platform)
