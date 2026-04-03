# CHANGELOG

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/) 規範。

---

## [Generation 3] — 2026-04-03 — Candidate 版本

### Added
- **FFT 頻域特徵**：在 `schema.py` 新增 `extract_window_fft_energy()` 函式，對 Magnitude 欄位（索引 3）執行 `rfft`，計算去除 DC 分量後的頻譜能量。
- **混合域輸入（8-D）**：模型輸入從 7 維提升至 8 維，新增 FFT Spectral Energy 特徵欄位，每視窗廣播為 (128, 1) 後與時域特徵合併。
- **實驗筆記本**：新增 `20260403_Project_Rehab_FFT.ipynb`，記錄 Generation 3 完整實驗流程。

### Changed
- **`load_and_preprocess_subject()`**：整合 FFT 特徵廣播邏輯，輸出 X shape 從 (n, 128, 7) 升級至 (n, 128, 8)。
- **能量縮放策略**：FFT 能量值執行 `log1p(x) / 5.0` 縮放，解決能量值跨度過大導致的梯度爆炸問題。

### Fixed
- **S2 坐姿誤判問題（核心修復）**：`Sitting`（Label 2）與 `Waist Bends Forward`（Label 6）的召回率混淆由頻域特徵解決，S2 Sitting Recall 由 Generation 2 的 ~2% 提升至 ~98%。

---

## [Generation 2.1] — 2026-03-08 — 過渡版本

### Added
- **重力特徵分離**：在 `schema.py` 實作 `apply_low_pass_filter()`，使用 Butterworth 低通濾波器（截止頻率 0.3Hz，order=2）從 Chest Acceleration 分離重力向量。
- **零相位濾波**：使用 `filtfilt` 替代 `lfilter`，消除時間延遲，確保重力分量與動作訊號在時間軸對齊。
- **7 維特徵矩陣**：新增 Gravity_X, Gravity_Y, Gravity_Z 三個重力分量，輸入維度從 4-D 提升至 7-D。

### Changed
- **選擇性標準化策略**：
  - 前 4 欄（Acc/Mag）：執行受試者 Z-score（消除個體強度差異）。
  - 後 3 欄（Gravity）：執行全局固定縮放（÷10.0），保留物理角度參考值。
- **`load_and_preprocess_subject()`**：整合重力特徵提取邏輯，重構特徵合併矩陣。
- **`create_sliding_windows_with_indices()`**：`feature_indices` 改為 `[0,1,2,3,4,5,6]`，`label_index` 改為 `7`。

### Fixed
- **S3 性能回復**：選擇性標準化後 S3 準確率提升至 91.8%（相較統一 Z-score 的 61.7%）。

---

## [Generation 2] — 2026-03-07 — 診斷驅動版本

### Added
- **物理邊界清洗**：在 `load_and_preprocess_subject()` 新增物理濾鏡，剔除 Sitting（Label 2）狀態下 Magnitude > 10.8 m/s² 的雜訊點。
- **LOSO 訓練驗證**：新增 `development_history/20260307_Project_Rehab_Training_LOSO.ipynb`，實作 Leave-One-Subject-Out 交叉驗證。
- **受試者標準化實驗**：針對 S1/S10 分佈偏移引入受試者層級 Z-score 正規化。

### Changed
- **驗證協議**：從隨機 train_test_split 改為基於受試者編號的 LOSO 交叉驗證。

---

## [Generation 1.2] — 2026-02-28 — 數據診斷版本

### Added
- **KS Test 一致性分析**：新增 `development_history/20260228_Project_Rehab_Diagnostics.ipynb`，對 10 位受試者的 Magnitude 分佈執行 Kolmogorov-Smirnov 檢定。
- **物理邊界稽核**：識別 S2（0.03%）與 S8（0.19%）的 Sitting 標籤污染。
- **LOSO 容量分析**：模擬 10-Fold LOSO 切分，評估各受試者有效視窗數（平均約 530）。

### Fixed
- **感測器偏置識別**：識別 S3 的中位數低於理想重力線問題，判定為硬體偏移。

---

## [Generation 1] — 2026-02-22 — 基礎版本

### Added
- **基礎資料管線**：建立 `schema.py` 核心架構，包含 `load_mhealth_subject()`、`add_magnitude_feature()`、`create_sliding_windows_with_indices()`。
- **4 維時域特徵**：初始特徵空間 Acc_X, Acc_Y, Acc_Z, Magnitude。
- **多受試者支援**：從單受試者（v3）擴展至 10 位受試者（v3.1_multi）。
- **Dropout 優化**：從 Dropout(0.5)（過度正則化，Training Acc ~80%）優化至 Dropout(0.2)（模型收斂至 ~93%）。
- **一致性測試**：新增 `development_history/20260222_Project_Rehab_Consistency_Test.ipynb`。

### Fixed
- **欠擬合問題**：Dropout 從 0.5 降至 0.2，解決 v3_1_big 的欠擬合問題（Training Acc < Validation Acc）。

---

## 技術世代摘要

| 世代 | 日期 | 特徵維度 | 核心改變 | 狀態 |
|------|------|---------|---------|------|
| Generation 1 | 2026-02-22 | 4-D | 基礎時域特徵，多受試者擴展 | 已退役 |
| Generation 2 | 2026-03-07 | 7-D | 重力向量 + 物理清洗 + LOSO | 過渡期 |
| Generation 2.1 | 2026-03-08 | 7-D | 零相位濾波 + 選擇性標準化 | 過渡期 |
| **Generation 3** | **2026-04-03** | **8-D** | **FFT 頻域特徵，S2 問題修復** | **當前候選版本** |
