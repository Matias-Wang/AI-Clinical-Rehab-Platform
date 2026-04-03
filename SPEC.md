# SPEC.md — 技術規格與實作準則

## 1. 資料規格 (Data Specification)

### 1.1 資料集來源

- **名稱**：mHealth Dataset
- **來源機構**：University of Granada (UGR)
- **感測器**：Shimmer2 穿戴式感測器
- **取樣頻率**：50 Hz
- **受試者數量**：10 位志願者（S1～S10）
- **受試者部位**：胸口（Chest）、左腳踝（Left Ankle）、右手腕（Right Lower Arm）

### 1.2 原始欄位定義（24 欄）

| 欄位索引 | 說明 | 單位 |
|---------|------|------|
| 0 | Chest Acceleration X | m/s² |
| 1 | Chest Acceleration Y | m/s² |
| 2 | Chest Acceleration Z | m/s² |
| 3 | ECG Lead 1 | mV |
| 4 | ECG Lead 2 | mV |
| 5–7 | Left Ankle Acceleration X/Y/Z | m/s² |
| 8–10 | Left Ankle Gyroscope X/Y/Z | deg/s |
| 11–13 | Left Ankle Magnetometer X/Y/Z | local |
| 14–16 | Right Arm Acceleration X/Y/Z | m/s² |
| 17–19 | Right Arm Gyroscope X/Y/Z | deg/s |
| 20–22 | Right Arm Magnetometer X/Y/Z | local |
| 23 | Activity Label | — |

> Label 0 代表 Null Class（動作轉換期），在預處理階段剔除。

### 1.3 活動標籤定義

| Label | 活動名稱 | 備註 |
|-------|---------|------|
| 1 | Standing Still | 靜態 |
| 2 | Sitting and Relaxing | 靜態（重點問題類別） |
| 3 | Lying Down | 靜態 |
| 4 | Walking | 動態 |
| 5 | Climbing Stairs | 動態 |
| 6 | Waist Bends Forward | 功能性 |
| 7 | Frontal Elevation of Arms | 功能性 |
| 8 | Knees Bending (Crouching) | 功能性 |
| 9 | Cycling | 動態 |
| 10 | Jogging | 動態 |
| 11 | Running | 動態 |
| 12 | Jump Front & Back | 動態（小樣本類別） |

---

## 2. 特徵工程規格 (Feature Engineering)

### 2.1 特徵維度演進

| 版本 | 特徵維度 | 特徵內容 |
|------|---------|---------|
| Generation 1 | 4-D | Acc_X, Acc_Y, Acc_Z, Magnitude |
| Generation 2 | 7-D | + Gravity_X, Gravity_Y, Gravity_Z |
| Generation 3 (當前) | 8-D | + FFT Spectral Energy |

### 2.2 當前特徵定義（Generation 3，8 維）

| 索引 | 特徵名稱 | 來源 | 正規化方式 |
|------|---------|------|-----------|
| 0 | Acc_X | Chest Sensor Col 0 | 受試者 Z-score |
| 1 | Acc_Y | Chest Sensor Col 1 | 受試者 Z-score |
| 2 | Acc_Z | Chest Sensor Col 2 | 受試者 Z-score |
| 3 | Magnitude | √(X²+Y²+Z²) | 受試者 Z-score |
| 4 | Gravity_X | 低通濾波 (0.3Hz Butterworth) | 全局縮放 ÷10.0 |
| 5 | Gravity_Y | 低通濾波 (0.3Hz Butterworth) | 全局縮放 ÷10.0 |
| 6 | Gravity_Z | 低通濾波 (0.3Hz Butterworth) | 全局縮放 ÷10.0 |
| 7 | FFT_Energy | rfft(Magnitude)[1:] 能量，log1p(x)/5.0 縮放 | 每視窗計算後廣播 |

### 2.3 物理邊界清洗規則

```
坐姿雜訊剔除條件：Label == 2 AND Magnitude > (9.80665 + 1.0) = 10.8 m/s²
全局異常剔除條件：Magnitude > 50.0 m/s²（ACC_ERROR_THRESHOLD）
資料有效性確認：9.0 ≤ Magnitude ≤ 11.0 表示感測器單位為 m/s²（正常重力範圍）
```

### 2.4 重力向量分離規格

- **濾波器類型**：Butterworth 低通濾波器（order=2）
- **截止頻率**：0.3 Hz（人體靜止重力特徵通常低於此頻率）
- **實作方式**：`filtfilt`（零相位濾波，避免時間延遲）
- **取樣頻率**：50 Hz

### 2.5 FFT 頻譜能量計算規格

```python
# 針對 Magnitude 欄位（索引 3）執行 rfft
fft_vals = rfft(window[:, 3])
energy = np.sum(np.abs(fft_vals[1:])**2) / len(window)  # 去除 DC 分量
# 縮放：log1p(energy) / 5.0（解決能量值跨度過大導致的梯度問題）
```

---

## 3. 視窗化規格 (Windowing Specification)

| 參數 | 數值 |
|------|------|
| Window Size | 128 time steps（= 2.56 秒 @ 50Hz） |
| Overlap (Stride) | 64 time steps（50% 重疊） |
| Label 策略 | 視窗內多數投票（majority vote via `scipy.stats.mode`） |
| Null Class 過濾 | majority_label == 0 的視窗直接捨棄 |

---

## 4. 模型規格 (Model Specification)

### 4.1 架構（LiteCNN v3.1_multi）

```
Input Shape: (128, 4)   ← Generation 2 之前；Generation 3 為 (128, 8)
├── Conv1D(filters=32, kernel_size=3, activation='relu')
├── MaxPooling1D(pool_size=2)
├── Flatten()
├── Dense(64, activation='relu')
├── Dropout(0.2)
└── Dense(13, activation='softmax')   ← 12 類動作 + 1 (label 0 已剔除，labels 1-12)
```

> 注意：輸出層為 13 個神經元，對應 label 1～12（label 0 為 Null Class 已預處理剔除）。

### 4.2 訓練策略

| 參數 | 設定 |
|------|------|
| 資料切分 | 80/20 train_test_split（stratify=y） |
| 驗證協議 | LOSO 交叉驗證（按受試者編號切分） |
| 優化器 | Adam（預設學習率） |
| 損失函數 | sparse_categorical_crossentropy |
| 訓練輪數 | ~50 Epoch（驗證損失於 Epoch 35 後出現輕微震盪） |

### 4.3 性能指標（Generation 3 候選版本）

| 指標 | 數值 |
|------|------|
| Training Accuracy | ~93% |
| Validation Accuracy | ~91% |
| S2 Sitting Recall（修復後）| ~98% |

---

## 5. 資料管線介面 (Pipeline API)

### `schema.py` 核心函式

| 函式名稱 | 用途 | 輸入 | 輸出 |
|---------|------|------|------|
| `load_mhealth_subject(subject_id, folder_path)` | 讀取單一受試者 `.log` 原始檔 | 受試者編號、資料夾路徑 | `pd.DataFrame` |
| `add_magnitude_feature(df)` | 計算並插入 Magnitude 欄位 | DataFrame | DataFrame（含 magnitude 欄） |
| `apply_low_pass_filter(data, cutoff, fs, order)` | Butterworth 低通濾波分離重力 | ndarray | ndarray（gravity 分量） |
| `extract_window_fft_energy(window_data)` | 計算單視窗 FFT 頻譜能量 | ndarray (128, n) | float |
| `create_sliding_windows_with_indices(df, feature_indices, label_index, window_size, overlap)` | 視窗切割 | DataFrame、索引列表 | `(X, y)` ndarray |
| `load_and_preprocess_subject(subject_id, folder_path)` | 單受試者完整預處理管線 | 受試者編號 | `(X, y)` ndarray，X shape: (n, 128, 8) |
| `get_all_subjects_for_analysis(folder_path)` | 所有受試者獨立字典 | 資料夾路徑 | `{sid: (X, y)}` |
| `get_final_training_data(folder_path)` | 合併所有受試者用於訓練 | 資料夾路徑 | `(X_final, y_final)` |

---

## 6. 受試者診斷資訊 (Subject-level Diagnostics)

### 6.1 KS Test 分佈一致性（2026/02/28）

| 受試者 | 分佈偏移 D 統計量 | 備註 |
|--------|----------------|------|
| S1, S10 | 最高，最高達 0.15 | 需額外正規化處理 |
| S4, S9 | 最低，約 0.012 | 高一致性群組 |

### 6.2 物理邊界稽核（Sitting Label 2）

| 受試者 | 物理違規比例 | 判定 |
|--------|------------|------|
| S2 | 0.03% | 動作標籤污染（可安全清洗） |
| S8 | 0.19% | 動作標籤污染（可安全清洗） |
| S3 | 中位數低於 9.8 | 感測器偏置（硬體偏移或配戴角度偏差） |

### 6.3 LOSO 容量分析

- 最高容量受試者：S2（約 550 個有效視窗）
- 最低容量受試者：S6、S7（約 500 個有效視窗）
- 全體平均：約 530 個有效視窗（支持 10-Fold 公平交叉驗證）

---

## 7. 環境與依賴 (Environment)

| 項目 | 規格 |
|------|------|
| Python | 3.13 |
| 套件管理 | UV |
| 主要依賴 | TensorFlow, NumPy, SciPy, Pandas, Matplotlib, Seaborn |
| 依賴清單 | `requirements.txt` |
