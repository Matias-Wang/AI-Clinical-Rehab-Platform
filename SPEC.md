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

> **實作備註**：受試者級 Z-score（欄位 0-3）與 FFT log1p/5.0 縮放（欄位 7）曾在 Week 5 產品化遷移時遺漏，於 2026-07-12 修復（見 ROADMAP.md Fix 3）。**順序要求**：FFT 能量必須用「原始尺度」的 Magnitude 計算，因此 Z-score 必須排在 FFT 廣播之後才能執行，順序顛倒會讓 FFT 特徵失真、模型準確率崩壞（實測：順序錯誤時整體準確率僅 ~5-48%，正確順序下為 ~90-97%）。
>
> **已知限制**：此 Z-score 為「受試者級」統計量，需要該受試者的完整視窗集合才能計算，適用於目前 `main.py`／`ui_bridge.py` 這種「預先載入整個受試者資料、再重播模擬即時串流」的架構。若未來串接真正的穿戴式硬體（逐筆即時進來、無法預知整個 session 的統計量），需要另外設計滾動統計量或線上標準化策略，不在本次修復範圍內。

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
| `extract_window_fft_energy(window_data)` | 計算單視窗 FFT 頻譜能量（含 log1p/5.0 縮放） | ndarray (128, n) | float |
| `dtw_distance(seq_a, seq_b, radius)` | 計算兩序列的 DTW 距離（Sakoe-Chiba band 限制版） | ndarray (n,), ndarray (m,), int | float |
| `create_sliding_windows_with_indices(df, feature_indices, label_index, window_size, overlap)` | 視窗切割 | DataFrame、索引列表 | `(X, y)` ndarray |
| `load_and_preprocess_subject(subject_id, folder_path)` | 單受試者完整預處理管線，含受試者級 Acc/Mag Z-score 標準化 | 受試者編號 | `(X, y)` ndarray，X shape: (n, 128, 8) |
| `get_all_subjects_for_analysis(folder_path)` | 所有受試者獨立字典 | 資料夾路徑 | `{sid: (X, y)}` |
| `get_final_training_data(folder_path)` | 合併所有受試者用於訓練 | 資料夾路徑 | `(X_final, y_final)` |
| `align_coordinates(X)` | Rodrigues' 旋轉公式校正座標偏差（實驗性，已驗證在低訊噪比環境下效果反向） | ndarray (N, 128, 8) | ndarray |
| `extract_golden_template(X, y, target_label)` | 從指定受試者資料中擷取第一個符合 target_label 的視窗作為黃金範本 | ndarray, ndarray, int | ndarray (128, 8) |

### `ClinicalQualityGate` 類別

| 方法 | 用途 | 輸入 | 輸出 |
|------|------|------|------|
| `__init__(golden_template)` | 初始化，設定黃金標準範本（建議使用 S10） | ndarray (128, 8) | — |
| `get_quality_report(X_window)` | 評估單視窗品質。含 Stage 8 的 NaN/Inf fail-safe：`var_y` 非有限值時一律攔截 | ndarray (128, 8) 或 (N, 128, 8) | tuple `(is_valid: bool, score: float, msg: str)` |

### `RealTimeStreamProcessor` 類別（Stage 5 新增，Stage 8 擴充）

| 方法 | 用途 | 輸入 | 輸出 |
|------|------|------|------|
| `__init__(window_size, stride, sanity_abs_limit, disconnect_dirty_frames)` | 初始化滑動視窗緩衝區（預設 window=128、stride=64，即 50% 重疊）與 Stage 8 防護參數 | int, int, float, int | — |
| `push_data(sensor_row)` | 驗證並推入單一時間步資料；達步長且緩衝已滿時回傳視窗 | ndarray (8,) | tuple `(ready: bool, window: ndarray (128, 8) \| None)` |
| `validate_frame(sensor_row)` | 檢查影格完整性（Stage 8） | array_like | ndarray (8,) float64，失敗則拋 `DirtyFrameError` |
| `reset_buffer()` | 清空緩衝區與步長計數器（Stage 8） | — | — |
| `get_health_stats()` | 串流健康度統計（Stage 8） | — | dict（`total_frames`、`dirty_frames`、`dirty_ratio`、`buffer_resets`、`consecutive_dirty`、`buffer_len`） |

### `RealTimeBiofeedbackEngine` 類別（Stage 5 新增）

| 方法 | 用途 | 輸入 | 輸出 |
|------|------|------|------|
| `__init__(quality_gate, golden_template, model, window_size, stride, dtw_radius)` | 注入已建立的品質閘門、黃金範本與 Keras 模型（模型由呼叫端載入，非傳入路徑） | `ClinicalQualityGate`, ndarray (128, 8), Keras model, int, int, int | — |
| `process_live_frame(new_frame)` | 逐幀推入；湊滿視窗時委派給 `evaluate_window()` | ndarray (8,) | dict 或 `None`（視窗未湊滿時） |
| `evaluate_window(window_data)` | 對單一完整視窗執行臨床決策流程（Stage 8 B1 抽出，供串流與批量／重播路徑共用） | ndarray (128, 8) | dict |
| `calculate_similarity(current_window)` | 計算與黃金範本 Grav_Y 的姿勢相似度評分（Stage 6 起：DTW，Sakoe-Chiba band） | ndarray (128, 8) | float（0–100） |

`process_live_frame()` 回傳 dict 的欄位為 `status`、`ui_color`、`msg`、`score`、`similarity`、`predict_label`、`reason`（詳見 9.5 節）。

### 串流例外階層（Stage 8 新增）

| 例外 | 觸發時機 |
|------|----------|
| `SensorStreamError` | 所有串流層例外的基底類別 |
| `DirtyFrameError` | 單一影格未通過 `validate_frame()`：維度錯誤、NaN/Inf、非數值型別，或數值超出 `SANITY_ABS_LIMIT` |
| `SensorDisconnectedError` | 連續髒影格數達 `DISCONNECT_DIRTY_FRAMES`（預設 25，即 0.5 秒 @ 50Hz） |

`process_live_frame()` 會攔截上述例外並轉為紅燈結果（`status="HALT"`），不向外拋出，確保長時間串流不因單一壞封包而中斷。

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

### 6.4 治療師集群效應診斷（2026/04/04）

**現象**：受試者 S1–S4 在 Label 7（Frontal Elevation of Arms）的 `Grav_Y` 變異數極低（< 0.0003），疑因不同治療師引導或感測器配戴習慣，導致關鍵側向平面訊號缺失。

| 群組 | 受試者 | Grav_Y 變異數（Label 7）| 特性 |
|------|--------|------------------------|------|
| 低活動組 | S1–S4 | < 0.0003 | 治療師集群效應，訊號缺失 |
| 邊緣案例 | S7、S9 | 0.00066–0.00096 | 低活動但仍超過及格閾值 |
| 黃金標準 | S10 | 0.001595 | 完整側向動作訊號 |

**壓力測試結果（Cross-Group Validation）**：

| 訓練集 | 測試集 | 準確率 |
|--------|--------|--------|
| S5–S10（高活動） | 全體 | 92% |
| S5–S10（高活動） | S1–S4（低活動） | 68.89%（崩跌 -23.11%）|
| 座標對齊後（Rodrigues'） | S1–S4 | 62.31%（反降 -6.58%）|

> **結論**：在極低訊噪比（Grav_Y_Var < 0.0003）環境下，數學補償放大隨機噪音。後端演算法補償失效，確立採用「臨床品質閘門（Clinical Quality Gate）」的前端引導策略。

### 6.5 Permutation Importance 分析（2026/04/04）

- **FFT Energy** 在跨受試者預測中重要度 > 40%，為最關鍵特徵。
- **Grav_Y** 在治療師集群效應情境下失效，導致低活動組準確率崩潰。

---

## 7. 環境與依賴 (Environment)

| 項目 | 規格 |
|------|------|
| Python | 3.13 |
| 套件管理 | UV |
| 主要依賴 | TensorFlow, NumPy, SciPy, Pandas, Matplotlib, Seaborn |
| 依賴清單 | `requirements.txt` |

---

## 8. 臨床品質閘門規格 (Clinical Quality Gate Specification)

> 確立於 2026/04/04，採 Data-Centric AI 路線，取代後端演算法補償。

### 8.1 設計動機

治療師集群效應（Therapist Clustering）導致 S1–S4 群組的 Label 7 動作 `Grav_Y` 變異數極低，使跨群組預測準確率從 92% 崩跌至 68.89%。座標對齊（Rodrigues' 旋轉）驗證失敗（準確率反降至 62.31%），確認後端補償策略在低訊噪比環境下無效。

### 8.2 閾值定義

| 閾值名稱 | 數值 | 依據 |
|---------|------|------|
| `GOLDEN_VAR_LIMIT`（黃金標準）| 0.001595 | S10 在 Label 7 的 Grav_Y 變異數實測值 |
| `MIN_SAFE_LIMIT`（及格線） | 0.0005 | S7（0.00066）、S9（0.00096）邊緣案例診斷，確保 ≥ 90% 準確率；切開 S1–S4（< 0.0003）與 S5–S10（> 0.0006）兩群 |

### 8.3 品質評分邏輯

```python
class ClinicalQualityGate:
    GOLDEN_VAR_LIMIT = 0.001595  # S10 黃金標準
    MIN_SAFE_LIMIT   = 0.0005    # 邊緣案例及格線

    def get_quality_report(self, X_window):
        var_grav_y = np.var(X_window[:, 5])  # 特徵索引 5 = Grav_Y
        score = (var_grav_y / self.GOLDEN_VAR_LIMIT) * 100
        passed = var_grav_y >= self.MIN_SAFE_LIMIT
        return {"score": score, "passed": passed, "grav_y_var": var_grav_y}
```

- **通過（passed=True）**：`Grav_Y_Var ≥ 0.0005`，動作資料有效，可供模型推論。
- **攔截（passed=False）**：`Grav_Y_Var < 0.0005`，輸出品質警示，應提示臨床人員重新引導病患動作或校正感測器配戴角度。

### 8.4 臨床意義

- 品質評分 < 30（如 S2 實測 20.9 分）：側向運動平面訊號嚴重缺失，模型推論結果不可信。
- 此機制將「資料品質保障」前移至採集端，符合 Data-Centric AI 原則，避免以後端模型補償掩蓋前端採集缺陷。

### 8.5 姿態相似度規格 (Posture Similarity Specification)
演算法：**DTW（Dynamic Time Warping，Sakoe-Chiba band 限制版，radius=16 samples ≈ 0.32 秒 @ 50Hz）**（Stage 6 起，取代原歐幾里德距離映射）。
特徵軸向：主要針對 Grav_Y (Index 5) 進行序列比對。
距離定義：Local cost 採平方差 `(a_i - b_j)^2`，最終距離為 `sqrt(累積最小路徑成本)`，與原歐幾里德距離同尺度。因對角線路徑恆為合法解，DTW 距離恆 `≤` 對應的歐幾里德距離；節奏完全一致時兩者相等，節奏不同時 DTW 更寬容，故 `* 15` 係數與 GREEN/YELLOW 門檻無需重新校準。
評分公式：$Score_{sim} = \max(0, 100 - Distance \times 15)$。
綜合評分權重：$Final\_Score = (Score_{quality} \times 0.4) + (Score_{sim} \times 0.6)$。

> 實作：`dtw_distance(seq_a, seq_b, radius)`，位於 `schema.py`。

### 8.6 UI 顏色狀態機與臨床引導邏輯系統
根據即時運算結果，將數值轉化為視覺回饋：

| 狀態 (Status) | 觸發條件 | UI 顏色 | 臨床提示訊息 |
| ---- | ---- | ---- | ---- |
| HALT | $Grav\_Y\_Var < 0.0005$ | 🔴 紅色 | ⚠️ 動作幅度嚴重不足，AI 停止預測。 |
| PROCEED | $Score_{sim} \le 80$ | 🟡 黃色 | ⚠️ 姿勢與標準範本有偏移，請修正。 |
| PROCEED | $Score_{sim} > 80$ | 🟢 綠色 | ✅ 動作標準，請保持！ |

---

## 9. 即時資料橋接規格 (Realtime Bridge Specification)

> 確立於 Stage 7，實作於 `ui_bridge.py`，展示前端 `frontend/demo.html`。

### 9.1 設計動機

`RealTimeBiofeedbackEngine` 的分析結果原本僅能在終端機文字輸出（`main.py`）。`ui_bridge.py` 提供一個獨立的 WebSocket 通信層，將每次 `process_live_frame()` 產生的非 `None` 結果即時推播給所有已連線的前端 client，不修改 `schema.py` 既有邏輯，純粹重用 `load_and_preprocess_subject`、`ClinicalQualityGate`、`RealTimeBiofeedbackEngine`、`ACTIVITY_LABELS`。

### 9.2 資料來源

沒有真實硬體輸入，改用指定受試者（預設 S10）的 mHealth log 檔案重播：`load_and_preprocess_subject(subject_id, "data/")` 取得的視窗集合逐一餵入 `engine.evaluate_window()`，依真實時間節奏推播（相鄰視窗相距 stride 幀，故間隔為 `stride / 50 / speed` 秒，`--realtime` 下為 1.28 秒）。播放到結尾後自動從頭重播。黃金範本固定取 S10（`load_and_preprocess_subject(10, ...)`），與 `--subject` 參數無關。

> **Stage 8 B1 變更**：原作法將已 50% 重疊的視窗攤平為逐幀串流（`X.reshape(-1, 8)`）再逐幀餵入 `process_live_frame()`。該作法使串流長度膨脹為真實訊號的 2.00 倍、每 128 步在時間軸倒退 64 步，且引擎重新切窗後有 50% 的視窗橫跨兩個原始視窗——第 8 維 FFT 能量因此出現兩個不同值，而該特徵在訓練資料中永遠是整個視窗的單一常數，構成 train/serving skew。FFT 為視窗級特徵且須以原始尺度的 Magnitude 計算（見 Fix 3），在已標準化的逐幀串流中無法重建，故改為逐視窗評估。

### 9.3 連線與埠號

- 協定：WebSocket（`ws://`）。
- 預設埠：`8765`（可用 `--port` 覆寫）。
- 一個伺服器行程對應一個受試者的模擬串流；所有已連線 client 收到相同的 broadcast。

### 9.4 CLI 參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--subject` | 模擬串流的受試者編號 | 10 |
| `--port` | WebSocket 監聽埠 | 8765 |
| `--speed` | 播放加速倍率（1.0 為真實 50Hz） | 20.0 |
| `--realtime` | 等同 `--speed 1.0` | — |

### 9.5 推播訊息格式 (JSON)

每當 `process_live_frame()` 產生一筆結果，伺服器即以下列 JSON 格式透過 WebSocket 推播給所有已連線 client：

```json
{
  "status": "PROCEED",
  "ui_color": "GREEN",
  "msg": "✅ AI 辨識：Walking",
  "score": 62.29,
  "similarity": 37.14,
  "predict_label": 4,
  "predict_label_name": "Walking",
  "reason": "OK",
  "subject_id": 10,
  "frame_index": 506
}
```

| 欄位 | 型別 | 說明 |
|------|------|------|
| `status` | string | `"HALT"` 或 `"PROCEED"`，對應 `ClinicalQualityGate` 判定 |
| `ui_color` | string | `"RED"` / `"YELLOW"` / `"GREEN"` |
| `msg` | string | 臨床提示訊息（含 emoji） |
| `score` | float | 綜合評分 Final Score |
| `similarity` | float | 姿態相似度評分（DTW，見 8.5 節） |
| `predict_label` | int \| null | 模型預測標籤（`HALT` 時為 `null`） |
| `predict_label_name` | string \| null | 由 `ACTIVITY_LABELS` 查得的中文動作名稱 |
| `reason` | string | 攔截／通過原因（Stage 8 新增）：`OK`、`LOW_QUALITY`、`DIRTY_DATA`、`DISCONNECTED` |
| `subject_id` | int | 目前模擬串流的受試者編號 |
| `frame_index` | int | 視窗序號，每次重播歸零 |

### 9.6 錯誤處理

- **Client 連線例外**：單一 client 的連線例外（如非正常關閉、無 close frame）僅記錄該 client 並從連線集合移除，不影響伺服器與其他 client 的運作（見 `handle_client()`）。
- **串流例外（Stage 8）**：`stream_subject()` 的 except 區塊在印出錯誤後會呼叫 `engine.reset_buffer()`。若不清空，造成例外的資料仍殘留於 deque 中，後續最多 `window_size`（128）步的視窗都會沿用被污染的緩衝，系統將輸出「看起來正常但基於污染資料」的評分。
- **輸出編碼（Stage 8）**：入口點於任何輸出前呼叫 `console.enable_utf8_output()`。Windows 繁中環境 locale 為 cp950，stdout 被重導至管道或檔案時會退回 locale 編碼，訊息中的 emoji 將觸發 `UnicodeEncodeError` 使程式中斷。