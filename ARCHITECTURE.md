# ARCHITECTURE.md — 系統架構設計

## 1. 系統概覽

本專案為一套**穿戴式感測器動作辨識研究平台**，採用模組化的 Notebook + 共用管線架構。核心設計原則是將可重用的資料處理邏輯集中於 `schema.py`，讓各實驗筆記本專注於模型訓練與評估，實現**關注點分離（Separation of Concerns）**。

```
┌─────────────────────────────────────────────────────────┐
│                    研究實驗層 (Notebooks)                 │
│   20260403_FFT.ipynb  │  development_history/*.ipynb     │
└────────────────┬────────────────────────────────────────┘
                 │ 呼叫
┌────────────────▼────────────────────────────────────────┐
│              核心管線層 (schema.py)                       │
│  資料讀取 → 清洗 → 特徵工程 → 視窗化 → 輸出 (X, y)       │
└────────────────┬────────────────────────────────────────┘
                 │ 讀取
┌────────────────▼────────────────────────────────────────┐
│                    資料層 (data/)                         │
│         mHealth_subject1.log ~ subject10.log             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 模組依賴關係

```
schema.py
├── os, pandas, numpy          # 基礎數值計算
├── scipy.stats                # mode()：視窗標籤多數投票
├── scipy.signal (butter, filtfilt)  # 重力向量低通濾波
└── numpy.fft (rfft)           # FFT 頻譜能量計算

Notebooks
├── schema.py                  # 共用資料管線
└── tensorflow.keras           # 模型定義、訓練、評估
```

---

## 3. 資料處理管線（Pipeline）

### 完整流程

```
mHealth_subjectN.log
        │
        ▼
[Step 1] load_mhealth_subject()
        原始 TSV 讀取（24 欄，無標頭）
        │
        ▼
[Step 2] 過濾 Label == 0（Null Class 剔除）
        │
        ▼
[Step 3] add_magnitude_feature()
        計算 √(X²+Y²+Z²) 並插入欄位 3
        │
        ▼
[Step 4] 物理邊界清洗
        剔除：Label==2 AND Magnitude > 10.8 m/s²
        │
        ▼
[Step 5] apply_low_pass_filter()
        Butterworth 低通濾波（0.3Hz, order=2, filtfilt 零相位）
        提取 Gravity_X, Gravity_Y, Gravity_Z
        │
        ▼
[Step 6] 特徵矩陣合併
        [Acc_X, Acc_Y, Acc_Z, Mag, Grav_X, Grav_Y, Grav_Z, Label]
        │
        ▼
[Step 7] create_sliding_windows_with_indices()
        Window=128, Overlap=64, 多數投票標籤
        輸出：X shape (n, 128, 7)
        │
        ▼
[Step 8] FFT 頻域特徵廣播
        對每個視窗計算 extract_window_fft_energy()
        廣播為 (128, 1) 後 hstack
        輸出：X shape (n, 128, 8)
        │
        ▼
    (X, y) → 模型訓練
```

---

## 4. 模型架構（LiteCNN）

```
Input: (128, 8)
        │
┌───────▼────────┐
│ Conv1D          │  filters=32, kernel_size=3, activation='relu'
│                 │  → 提取時序訊號中的局部動作模式
└───────┬────────┘
        │
┌───────▼────────┐
│ MaxPooling1D    │  pool_size=2
│                 │  → 降採樣，減少計算量
└───────┬────────┘
        │
┌───────▼────────┐
│ Flatten         │
└───────┬────────┘
        │
┌───────▼────────┐
│ Dense(64)       │  activation='relu'
│                 │  → 高階特徵組合
└───────┬────────┘
        │
┌───────▼────────┐
│ Dropout(0.2)    │  → 正則化（由 0.5 優化至 0.2 解決欠擬合）
└───────┬────────┘
        │
┌───────▼────────┐
│ Dense(13)       │  activation='softmax'
│                 │  → 12 類動作輸出（labels 1–12）
└───────┬────────┘
        │
   輸出：動作類別預測
```

**設計選擇說明：**
- **Conv1D 而非 Conv2D**：感測器訊號為一維時間序列，Conv1D 可捕捉局部時序模式，計算成本低。
- **Dropout 0.2 而非 0.5**：實驗發現 0.5 導致欠擬合（Training Acc 僅 80%），降至 0.2 後模型成功收斂至 93%。
- **單層 Conv1D**：符合 Edge AI 部署需求的輕量化設計（LiteCNN）。

---

## 5. 技術世代演進

| 世代 | 輸入維度 | 核心技術 | 主要問題 | 狀態 |
|------|---------|---------|---------|------|
| Generation 1 | (128, 4) | Acc + Magnitude | S2 坐姿誤判（Recall 2%） | 已退役 |
| Generation 2 | (128, 7) | + Gravity 向量 | 靜態/動態角度重疊悖論 | 過渡期 |
| Generation 3 | (128, 8) | + FFT 頻譜能量 | S2 修復（Recall 98%） | **當前候選版本** |

---

## 6. 特徵設計原理

### 6.1 為何需要重力分量？

原始加速度包含動作加速度與重力加速度的混疊。低通濾波後的重力分量代表感測器相對於重力的傾斜角度，可區分「站立」與「坐姿」等姿態差異。

### 6.2 為何需要 FFT 頻域特徵？

「坐姿」（Sitting）與「彎腰」（Waist Bends Forward）在時域角度特徵上高度相似（皆為軀幹前傾），但頻域能量完全不同：
- **Sitting**：低頻譜能量（靜止狀態）
- **Waist Bends**：高頻譜能量（週期性動作）

### 6.3 選擇性標準化設計

| 特徵類型 | 標準化方式 | 原因 |
|---------|-----------|------|
| Acc / Magnitude（前 4 維） | 受試者 Z-score | 消除個體體重、肌力差異導致的強度偏移 |
| Gravity（後 3 維） | 全局縮放 ÷10.0 | 重力絕對值承載物理意義（傾斜角度），不可歸零 |

> 失敗案例：對全部 7 個特徵執行 Z-score → 重力絕對值歸零 → S2 準確率重挫至 61.7%

---

## 7. 驗證架構

```
訓練資料來源：10 位受試者合併資料
        │
        ├── 隨機切分模式：stratified train_test_split (80/20)
        │   └── 用於快速迭代實驗
        │
        └── LOSO 模式：按受試者編號分割
            └── 模擬臨床部署：模型從未見過測試受試者
                確保跨個體泛化能力
```

---

## 8. 資料夾架構

```
AI-Clinical-Rehab-Platform/
│
├── schema.py                        # [核心] 資料處理管線
├── 20260403_Project_Rehab_FFT.ipynb # [主實驗] Generation 3
├── requirements.txt                 # 依賴清單
│
├── data/                            # 原始資料
│   ├── mHealth_subject1.log
│   ├── ...
│   ├── mHealth_subject10.log
│   └── data_raw_Info.md             # 資料集規格說明
│
├── models/                          # 已訓練模型存放
│   └── clinical_rehab_model_v1.keras
│
├── architecture/                    # 架構圖與評估視覺化
│   └── confusion_matrix/
│       ├── confusion_matrix_v3_multi.png       # Generation 2 混淆矩陣
│       ├── confusion_matrix_v3_1_multi.png     # Generation 2.1 混淆矩陣
│       ├── S2_Confusion_Matrix_FFT_Acc.png     # Generation 3 S2 混淆矩陣
│       └── S2_Confusion_Matrix_FFT_Acc_0.78.png
│
├── Spec/                            # 技術規格存檔
│   └── v3_1_multi_Technical_Spec.md
│
├── docs/                            # 參考文獻（醫療法規、SaMD）
│
├── development_history/             # 歷史實驗存檔
│   ├── 20260222_Project_Rehab_Consistency_Test.ipynb
│   ├── 20260228_Project_Rehab_Diagnostics.ipynb
│   └── 20260307_Project_Rehab_Training_LOSO.ipynb
│
├── README.md                        # 專案門面與快速開始
├── ARCHITECTURE.md                  # 系統架構（本文件）
├── SPEC.md                          # 技術規格與實作準則
├── CHANGELOG.md                     # 版本更新日誌
├── 檔案說明.md                       # 專案文件結構說明
├── CLAUDE.md                        # Claude AI 設定
└── GEMINI.md                        # Gemini AI 設定
```

---

## 9. 未來架構演進

### 近期（Generation 4 候選）

- **Transformer Self-Attention**：取代 Conv1D 以捕捉長距離時序依賴關係。
- **Wavelet Transform**：評估小波轉換與 FFT 的互補性。
- **Per-subject Normalization**：針對 S1/S10 的 15% 分佈偏移問題引入個體化預處理。

### 中期（產品化）

- **Edge AI 部署**：TensorFlow Lite 模型轉換，適配穿戴式裝置推論。
- **MLOps Pipeline**：AWS SageMaker 自動化訓練、評估、部署流程。
- **SaMD 合規**：依據 IMDRF SaMD 框架進行風險分類與技術文件準備。
