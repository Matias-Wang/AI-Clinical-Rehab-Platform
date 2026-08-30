# CHANGELOG

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/) 規範。

---

## [Stage 8 — C] — 2026-08-31 — 即時推論路徑最佳化

### Changed
- **`RealTimeBiofeedbackEngine.process_live_frame()`**：模型推論由 `model.predict(x, verbose=0)` 改為 `model.predict_on_batch(x)`。`predict()` 為批次導向 API，每次呼叫都會建立 data adapter 與 tf.function 追蹤，用於「單一視窗、高頻重複」的即時推論時開銷極高，且會持續累積記憶體。

### Notes
- 實測數據（Keras 3.13.2，單筆 `(1, 128, 8)` 輸入，150 次取中位數）：

  | 方案 | 延遲中位數 | p95 | 2000 次後 RSS |
  |------|-----------|-----|---------------|
  | `predict()`（原） | 94.91 ms | 212.65 ms | +5.5 MB |
  | `model(x, training=False)` | 145.64 ms | 420.59 ms | +0.0 MB |
  | **`predict_on_batch()`（採用）** | **2.49 ms** | **3.38 ms** | **+0.0 MB** |
  | `tf.function()` 包裝 | 2.57 ms | 3.59 ms | +0.0 MB |

- **輸出逐位元相同**（最大絕對差異 `0.000e+00`，argmax 全部一致），故所有既有驗收數字不受影響。
- 端到端：`python main.py` 由 **256 秒降至 44 秒**（5.8 倍），輸出與修改前逐行完全相同。
- 修正先前規劃：Roadmap 原記載此項應改用 `model(x, training=False)`（Keras 通則），但在本專案的 Keras 3.13 環境下實測反而最慢，已改採 `predict_on_batch()`。
- 新增回歸守門測試 `test_uses_predict_on_batch_not_predict`，以 stub model 分別計數兩條推論路徑，防止日後回退。

---

## [Stage 8 — A/D/F] — 2026-08-30 — 臨床串流防護、緩衝區復原與回歸測試

### Added
- **`tests/`**：專案首套自動化測試（72 項），涵蓋預處理修正、品質閘門、串流處理器、DTW 數學性質、引擎狀態機與模型準確率基準。準確率基準測試標記 `slow`，缺少 TensorFlow 或模型檔時自動 skip。新增 `pytest.ini` 設定 testpaths 與 marker。
- **`console.py`**：新增 `enable_utf8_output()`，於入口點將 stdout/stderr 設為 UTF-8。
- **`RealTimeStreamProcessor.validate_frame()`**：於影格進入緩衝區前攔截維度錯誤、NaN/Inf、非數值型別與超出生理範圍的資料。
- **`RealTimeStreamProcessor.reset_buffer()`**：清空 deque 與步長計數器，供髒數據攔截後的復原使用。
- **`RealTimeStreamProcessor.get_health_stats()`**：回傳總影格數、髒影格數、髒數據比例、緩衝區重置次數與目前緩衝長度，供長時間壓力測試監控。
- **例外階層**：`SensorStreamError`（基底）、`DirtyFrameError`、`SensorDisconnectedError`。
- **`STREAM_GUARD` 常數**：`N_FEATURES=8`、`SANITY_ABS_LIMIT=50.0`（標準化後尺度）、`DISCONNECT_DIRTY_FRAMES=25`（0.5 秒 @ 50Hz）。
- **結果 `reason` 欄位**：`OK` / `LOW_QUALITY` / `DIRTY_DATA` / `DISCONNECTED`，同步加入 WebSocket 推播 JSON。

### Fixed
- **`ClinicalQualityGate.get_quality_report()`**：補上 `np.isfinite(var_y)` 防護。原本 `NaN < MIN_SAFE_LIMIT` 在 IEEE 754 下恆為 False，含 NaN 的視窗會被判定為「品質良好」並直接送入模型推論，品質閘門在最需要攔截時完全失效。
- **`ui_bridge.py` 例外處理**：except 區塊補上 `engine.reset_buffer()`。原本只印出錯誤就繼續，造成例外的資料仍殘留在 deque 中，後續最多 128 步的視窗都會沿用被污染的緩衝。
- **cp950 輸出編碼**：`python main.py > out.log` 原本會在第一位受試者結果處崩潰（`UnicodeEncodeError`），批量驗收報告無法完整輸出。問題橫跨 `main.py`、`inference_test.py` 與 `schema.py`；因 `schema.py` 的訊息字元會送至前端顯示，改以入口點統一設定 UTF-8 解決，而非移除 emoji。

### Removed
- **`inference_test.py`**：呼叫 `RealTimeBiofeedbackEngine` 時缺少 `model` 參數，已無法執行；驗證範圍由 `tests/test_biofeedback_engine.py` 涵蓋。

### Notes
- `status` 欄位維持原值（`HALT` / `PROCEED`），髒數據與斷訊皆歸類為 `HALT`，因此 `main.py` 與 `frontend/demo.html` 無需任何改動。
- 經突變測試驗證測試品質：逐一注入 4 個已修復的回歸（移除重力 ÷10.0、Z-score 提前至 FFT 之前、移除 NaN 防護、移除緩衝區清空），測試分別失敗 2／7／4／3 項。
- 驗收數字完全保留：`python main.py` 端到端通過，S01=3.5%、S02=5.1%、S03=16.5%、S04=15.1%、S10=20.6%。
- 專案維護：以 `git filter-repo` 清除誤入版控的 217MB `data_raw/` 歷史 blob，`.git` 由 95MB 降至 7.7MB，21 個 commit 全數保留。

---

## [Fix 3] — 2026-07-12 — Acc/Mag Z-score 與 FFT 縮放修復（Train/Serving Skew）

### Fixed
- **`load_and_preprocess_subject()`**：補回 Week 5 產品化遷移時遺漏的受試者級 `StandardScaler` Z-score（僅針對 Acc_X/Y/Z/Magnitude），並確保順序正確——Z-score 排在 FFT 廣播之後，避免 FFT 能量被非原始尺度的 Magnitude 污染。
- **`extract_window_fft_energy()`**：補回文件（SPEC.md §2.5）記載但程式碼裡缺失的 `log1p(x) / 5.0` 縮放。
- 兩者皆為導致模型從未預測 Label 1/5/6/7、真實 Label 7 視窗 98% 誤判為 Label 10 的根本原因。修復後全受試者平均準確率 92.6%，個別受試者準確率（S02=0.8430、S07=0.9045、S09=0.9065、S10=0.9029）與訓練 notebook 記錄完全吻合。

### Added
- **`extract_golden_template(X, y, target_label=7)`**：新增於 `schema.py`，篩選指定標籤的第一個視窗作為黃金範本，取代 `main.py`／`ui_bridge.py` 原本直接取 `X[0]`（未必是 Label 7 動作）的作法。

### Notes
- 此修復不影響 `ClinicalQualityGate`（Grav_Y 變異數）與 Stage 6 的 DTW 相似度計算，兩者使用獨立的 Gravity 特徵路徑；Stage 5-7 的品質閘門驗收數字維持有效。
- 已知限制：此 Z-score 為受試者級統計量，僅適用於目前「預先載入整個受試者、重播模擬即時串流」的架構；真正逐筆即時的硬體串流需另外設計線上標準化策略。

---

## [Stage 7] — 2026-07-11 — 即時資料橋接與展示前端

### Added
- **`ui_bridge.py`**：新增 WebSocket 即時資料橋接伺服器（`asyncio` + `websockets`），將 `RealTimeBiofeedbackEngine` 的即時推論結果（`ui_color`、`final_score`、`similarity` 等）broadcast 給所有已連線 client。重用 `schema.py` 既有的 `load_and_preprocess_subject`、`ClinicalQualityGate`、`RealTimeBiofeedbackEngine`、`ACTIVITY_LABELS`，不修改任何既有邏輯。支援 `--subject`／`--port`／`--speed`／`--realtime` CLI 參數。
- **`frontend/demo.html`**：新增最小展示前端（自包含 HTML/JS，無框架），以瀏覽器原生 WebSocket API 連線，顯示紅/黃/綠燈號與即時評分，用於驗證後端到前端的串接管線。
- **依賴**：`requirements.txt` 新增 `websockets`（保留原檔案 UTF-16 LE 編碼）。

### Notes
- 已驗證：伺服器啟動流程、單/雙 client 收發、broadcast 一致性、client 異常斷線時伺服器不崩潰、`python main.py` 既有驗收流程不受影響。
- 3D 前端視覺化與 50Hz 真實速率長時間壓力測試留待後續 Stage。

---

## [Stage 6] — 2026-07-11 — DTW 相似度演算法

### Added
- **`dtw_distance()`**：在 `schema.py` 新增純 numpy 實作的 DTW（Dynamic Time Warping）距離函式，採 Sakoe-Chiba band 限制（`radius=16` samples ≈ 0.32 秒 @ 50Hz），local cost 為平方差、最終距離開根號，與原歐幾里德距離同尺度。

### Changed
- **`RealTimeBiofeedbackEngine.calculate_similarity()`**：改用 `dtw_distance()` 取代 `np.linalg.norm()`，容忍受試者動作節奏（相位）與 S10 黃金範本的差異，減少「姿勢正確但節奏不同」被誤判黃燈的情況。
- **`RealTimeBiofeedbackEngine.__init__()`**：新增可調參數 `dtw_radius`（預設 16）。

### Notes
- 因對角線路徑恆為 DTW 搜尋空間中的合法解，DTW 距離恆 `≤` 對應的歐幾里德距離，故評分公式（`* 15` 係數）與 GREEN/YELLOW 門檻無需重新校準；`ClinicalQualityGate` 攔截邏輯未受影響。

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
