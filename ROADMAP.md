# 🚀 AI 臨床復健平台 - 專案交接與研發計畫說明書

---

## 📌 專案概觀

| 項目 | 內容 |
|------|------|
| **專案名稱** | AI-Clinical-Rehab-Platform |
| **核心目標** | 利用 mHealth 感測器數據（加速度、重力分量、FFT 能量）實現即時復健動作辨識與臨床品質監控，並提供生物回饋評分 |
| **目前進度** | Stage 8 之 A/C/D/F 完成（髒數據防護、推論路徑最佳化、緩衝區復原、回歸測試基礎建設 + cp950 修正）；剩餘 B 與 E |
| **當前日期** | 2026-08-31 |

---

## 🛠️ 當前技術架構與檔案說明

### 核心檔案清單

| 檔案 | 說明 |
|------|------|
| `schema.py` | 核心邏輯庫。包含資料預處理（Low-pass Filter, FFT）、`ClinicalQualityGate`（品質攔截器）與 `RealTimeBiofeedbackEngine`（即時推論引擎） |
| `main.py` | 產品入口點。實現自動化批量測試與實時模擬，支持多受試者會期切換 |
| `ui_bridge.py` | Stage 7 新增。WebSocket 即時資料橋接伺服器，將 `RealTimeBiofeedbackEngine` 的推論結果推播給前端 |
| `frontend/demo.html` | Stage 7 新增。最小展示前端，連線 `ui_bridge.py` 顯示紅/黃/綠燈號與評分 |
| `console.py` | Stage 8 新增。提供 `enable_utf8_output()`，於入口點將 stdout/stderr 設為 UTF-8，避免輸出被重導時因 cp950 崩潰 |
| `tests/` | Stage 8 新增。73 項自動化回歸測試，釘死 Fix 2／Fix 3 等致命修正。以 `pytest` 執行，`-m "not slow"` 可略過需要 TensorFlow 的準確率基準測試 |
| `models/clinical_rehab_model_v3.keras` | Stage 3 訓練之 v3.1_multi 模型。輸入維度為 $(1, 128, 8)$ |

---

### 數據規格 (Generation 3 - 8D Features)

每筆影格資料必須包含以下 **8 個特徵軸**：

| # | 特徵名稱 | 說明 |
|---|----------|------|
| 1 | `Acc_X` | 加速度 X 軸（必須執行受試者級 Z-score） |
| 2 | `Acc_Y` | 加速度 Y 軸（必須執行受試者級 Z-score） |
| 3 | `Acc_Z` | 加速度 Z 軸（必須執行受試者級 Z-score） |
| 4 | `Magnitude` | 合力向量 $\sqrt{x^2+y^2+z^2}$（必須執行受試者級 Z-score） |
| 5 | `Grav_X` | 低通濾波分離後的重力分量（必須執行全局縮放 ÷ 10.0） |
| 6 | `Grav_Y` | 低通濾波分離後的重力分量（必須執行全局縮放 ÷ 10.0） |
| 7 | `Grav_Z` | 低通濾波分離後的重力分量（必須執行全局縮放 ÷ 10.0） |
| 8 | `FFT_Energy` | 視窗內之頻譜能量，採 Log 縮放 $\frac{\log(1 + x)}{5.0}$ |

---

## ⚠️ 關鍵研發修正記錄 (Fatal Fixes)

> 接手 AI 必須注意以下兩點，這是確保系統不會崩潰或邏輯空轉的關鍵。

### Fix 1 — 維度對齊修復

- **問題**：曾發生 $(1, 128, 7)$ 維度錯誤
- **原因**：推論端遺漏了第 8 個特徵（FFT Energy）
- **修正時間**：Stage 5
- **現狀**：統一由 `schema.py` 的 `load_and_preprocess_subject` 提供標準 8D 數據

### Fix 2 — 重力縮放校準 (÷10.0)

- **問題**：這是最嚴重的 Bug
- **原因**：`schema.py` 遺漏了縮放，導致變異數放大 100 倍，使品質閘門完全失效
- **修正方式**：

  ```python
  gravity_values = gravity_values / 10.0
  ```

### Fix 3 — Acc/Mag 受試者級 Z-score 與 FFT 縮放遺失（Train/Serving Skew）

- **問題**：模型從未預測過 Label 1、5、6、7（12 類中有 4 類完全消失），真實 Label 7 視窗有 98% 以上被誤判成 Label 10；`main.py` 報告的「AI 辨識 L7 次數」因此永遠是 0。
- **原因**：對照訓練用 notebook（`development_history/20260404_Project_Rehab_Optimization.ipynb`）才發現，`schema.py` 的 `load_and_preprocess_subject()` 遺漏了兩個關鍵步驟：(1) Acc_X/Y/Z/Magnitude 的**受試者級 `StandardScaler` Z-score**；(2) FFT 能量的 `log1p(x)/5.0` 縮放。兩者都只存在於 SPEC.md／訓練 notebook 的文件與紀錄裡，Week 5 產品化遷移時沒有真正落實到程式碼。
- **順序陷阱**：FFT 能量必須用「原始尺度」的 Magnitude 計算，Z-score 一定要排在 FFT 廣播之後；順序顛倒過（Z-score 在 FFT 之前）整體準確率反而崩到 ~5-48%，看起來像修好了實際上更糟，必須靠對照 notebook 的基準準確率（S10=0.9029、S2=0.8430）才抓出來。
- **修正時間**：2026-07-12
- **現狀**：`load_and_preprocess_subject()` 已補回正確順序的 Z-score 與 FFT 縮放；`extract_window_fft_energy()` 回傳值已含 log1p/5.0。修復後全受試者平均準確率 92.6%，個別受試者數字（S02=0.8430、S07=0.9045、S09=0.9065、S10=0.9029）與 notebook 記錄完全吻合。同時修正 `main.py`／`ui_bridge.py` 原本直接取 `X[0]` 當黃金範本的問題，改用新增的 `extract_golden_template(X, y, target_label=7)` 明確篩選 Label 7 視窗。
- **範圍確認**：只影響 Acc/Mag 與 FFT 這兩條特徵路徑，`ClinicalQualityGate`（Grav_Y）與 Stage 6 的 DTW 相似度計算不受影響，Stage 5-7 的品質閘門驗收數字維持有效。
- **已知限制**：Z-score 是受試者級統計量，需要該受試者完整視窗集合才能計算，僅適用於目前「預先載入整個受試者、重播模擬即時串流」的架構；若未來接上真正逐筆即時的穿戴式硬體，需另外設計線上/滾動標準化策略。

---

## 📊 臨床評分邏輯 (Scoring Logic)

系統採用**雙軌評分機制**，確保回饋的真實性：

### 品質分數 ($Score_{quality}$)

- 檢查 $Grav\_Y$ 變異數是否大於 `0.0005`
- 若低於此值，系統判定為靜態或無效動作，強制亮起 **🔴 紅燈 (RED / HALT)**

### 相似度分數 ($Score_{sim}$)

- 基於 **DTW（Dynamic Time Warping，Sakoe-Chiba band, radius=16）** 比對受試者與 **S10 黃金範本**（Stage 6 起，取代原歐幾里德距離；因對角線路徑恆為合法解，DTW 距離恆 ≤ 原歐幾里德距離，故評分公式與門檻無需重新校準）

$$Score_{sim} = \max(0,\ 100 - Distance \times 15)$$

### 綜合評分 (Final Score)

$$Final = (Score_{quality} \times 0.4) + (Score_{sim} \times 0.6)$$

---

## 📅 未來研發規劃 (Roadmap)

### Stage 6 — 演算法魯棒性優化（DTW 實作）✅ 已完成 (2026-07-11)

**核心痛點**：舊版 schema.py 中的動作相似度比對採用歐幾里德距離（Euclidean Distance）。這是點對點的剛性計算，只要受試者在復健時的節奏或速度與黃金範本（S10）差了 0.05 秒，分數就會暴跌並觸發黃燈（🟡）。

**已完成內容**：

- 新增 `dtw_distance(seq_a, seq_b, radius=16)`：純 numpy 手刻 DTW（Sakoe-Chiba band 限制，radius=16 samples ≈ 0.32 秒 @ 50Hz），local cost 採平方差、最終距離開根號，與原歐幾里德距離同尺度。
- `RealTimeBiofeedbackEngine.calculate_similarity()` 改用 `dtw_distance()` 取代 `np.linalg.norm()`；`__init__` 新增可調參數 `dtw_radius`（預設 16）。
- 因對角線路徑恆為 DTW 搜尋空間中的合法解，DTW 距離恆 `≤` 原歐幾里德距離，評分公式（`*15` 係數）與 GREEN/YELLOW 門檻**無需重新校準**，`ClinicalQualityGate` 完全未變動。
- 已跑 `python main.py` 端到端驗證，S1–S04 合格率仍明顯低於 S10，攔截行為與 Stage 5 基準一致。

**尚未涵蓋（留給後續 Stage）**：

- 多標籤模型適配性測試：除了目前的 Label 7（手臂前舉）之外，測試 v3 模型對其他復健動作標籤的即時推論與相似度比對。

### Stage 7 — 系統整合與 UI 橋接 ✅ 已完成 (2026-07-11)

**核心痛點**：舊版系統輸出（品質報告、AI 預測、相似度分數）只能在終端機以文字印出，尚未與任何前端畫面接軌。

**已完成內容**：

- 新增 `ui_bridge.py`：以 `asyncio` + `websockets` 建立 WebSocket 伺服器（預設埠 8765），重用 `schema.py` 既有的 `load_and_preprocess_subject`、`ClinicalQualityGate`、`RealTimeBiofeedbackEngine`、`ACTIVITY_LABELS`，模擬指定受試者（預設 S10）的逐幀串流並將每個視窗結果即時 broadcast 給所有已連線 client（JSON 格式，見 SPEC.md 9.5 節）。支援 `--subject`／`--port`／`--speed`／`--realtime` 參數。
- 新增 `frontend/demo.html`：自包含 HTML/JS 最小展示前端（無框架、無 build tooling），以瀏覽器原生 WebSocket API 連線，顯示紅/黃/綠燈號與 final_score／similarity／AI 辨識動作。
- 已驗證：伺服器啟動流程（STEP 1–4 皆正常）、單一 client 收發、雙 client 同時連線收到相同 broadcast、client 異常斷線時伺服器不崩潰並正確清理連線集合、`python main.py` 既有驗收流程不受影響。
- 依賴新增 `websockets`（已寫入 `requirements.txt`，保留其原有 UTF-16 編碼）。

**尚未涵蓋（留給後續 Stage 或正式前端專案）**：

- 正式的 3D 前端視覺組件（`frontend/demo.html` 僅為驗證串接用的 2D 展示頁）。
- 高頻率（50Hz 真實速率）長時間串流的延遲與穩定性壓力測試（見 Stage 8）。

### Stage 8 — 最終壓力測試與交付（進行中）

**目前痛點**：系統即將交付，必須確保在長時間、高併發或異常斷訊的臨床環境下，系統不會發生記憶體洩漏（Memory Leak）或無預警崩潰。

依風險排序拆為 A–F 六項子任務，A/D/F 已於 2026-08-30 完成。

#### A — 髒數據防護 ✅ 已完成 (2026-08-30)

- `ClinicalQualityGate.get_quality_report()` 補上 `np.isfinite()` 防護。原本 `NaN < MIN_SAFE_LIMIT` 在 IEEE 754 下恆為 False，含 NaN 的視窗會被判定為「品質良好」並直接送入模型推論。
- 新增 `RealTimeStreamProcessor.validate_frame()`，於影格進入緩衝區前攔截維度錯誤、NaN/Inf、非數值型別與超出生理範圍的資料。
- 新增例外階層 `SensorStreamError` / `DirtyFrameError` / `SensorDisconnectedError`；連續 25 幀髒數據（0.5 秒 @ 50Hz）升級為斷訊。
- **設計要點**：邊界檢查採標準化後尺度（Z-score 50 個標準差），而非文件中的 50 m/s² 原始物理門檻——進入串流處理器的資料已完成標準化，沿用原始門檻會使檢查完全失效。

#### D — 緩衝區復原 ✅ 已完成 (2026-08-30)

- 新增 `reset_buffer()`，髒影格被攔截時同步清空 deque 與步長計數器，避免後續最多 128 步的視窗沿用被污染的緩衝。
- `process_live_frame()` 將例外轉為紅燈結果而非向外拋出，並僅在「乾淨→髒」轉換點輸出，避免洪水式推播。
- `ui_bridge.py` 的 except 區塊補上 `engine.reset_buffer()`。
- 結果新增 `reason` 欄位（`OK` / `LOW_QUALITY` / `DIRTY_DATA` / `DISCONNECTED`），`status` 維持原值，故 `main.py` 與 `demo.html` 無需改動。
- 新增 `get_health_stats()`，供壓力測試監控緩衝區與髒數據比例。

#### F — 回歸測試基礎建設 ✅ 已完成 (2026-08-30)

- 新增 `tests/`，共 72 項測試。專案在此之前沒有任何自動化測試，Fix 2／Fix 3 僅存在於文件記載。
- **Fix 3 順序陷阱的精確不變量**：若 Z-score 誤排到 FFT 之前，第 7 欄能量會恰好等於以已標準化的 Magnitude 重算的結果，直接斷言兩者必須不同即可在 3 秒內攔截，無需載入 TensorFlow。
- 準確率基準測試對照訓練 notebook（S02=0.8430、S07=0.9045、S09=0.9065、S10=0.9029），標記 `slow`。
- 經突變測試驗證：注入 4 個已修復的回歸，測試分別失敗 2／7／4／3 項。

#### 附帶修正 — cp950 輸出編碼 ✅ 已完成 (2026-08-30)

`python main.py > out.log` 原本會崩潰於 `UnicodeEncodeError`，批量驗收報告無法完整輸出。新增 `console.py` 於入口點統一設定 UTF-8 輸出。

#### B — 推論阻塞 event loop ⏳ 待重新評估

推論為同步呼叫，跑在 `ui_bridge.py` 的 asyncio 主迴圈上，執行期間整個 WebSocket 伺服器無法服務其他事件。原規劃以 `asyncio.to_thread()` 隔離，但 C 完成後單次推論僅 2.49 ms，阻塞影響大幅下降（見 C 的連帶影響說明），須先實測 event loop 實際延遲再決定是否值得引入執行緒的複雜度。

#### C — 推論路徑最佳化 ✅ 已完成 (2026-08-31)

- `process_live_frame()` 的推論由 `model.predict(x, verbose=0)` 改為 `model.predict_on_batch(x)`。
- **實測推翻了原規劃**：本項原記載應改用 `model(x, training=False)`（Keras 通則），但在此專案的 Keras 3.13.2 環境下實測反而最慢（145.64 ms）。四個候選方案實測後採用 `predict_on_batch()`。

  | 方案 | 延遲中位數 | 2000 次後 RSS |
  |------|-----------|---------------|
  | `predict()`（原） | 94.91 ms | +5.5 MB |
  | `model(x, training=False)` | 145.64 ms | +0.0 MB |
  | **`predict_on_batch()`（採用）** | **2.49 ms** | **+0.0 MB** |

- 輸出逐位元相同（最大絕對差異 0.0），所有既有驗收數字不受影響。
- 端到端 `python main.py` 由 256 秒降至 44 秒（5.8 倍），輸出逐行完全相同。
- 新增守門測試防止回退至 `predict()`。

**對 B 的連帶影響**：B 的急迫性建立在「推論很慢」之上。推論由 94.91 ms 降至 2.49 ms 後，以 `--speed 20` 計算（每 64 ms 產生一個視窗），event loop 阻塞佔比由約 148%（根本追不上）降至約 4%，B 已非急迫項目，需重新量測後再決定是否實作。

#### E — 壓力測試與交付文件 ⏳ 待處理

- 極限壓力測試：長時間循環運行全量受試者數據（S1–S10），以 `get_health_stats()` 監控 `RealTimeStreamProcessor` 緩衝區在極端情況下的穩定性。
- 撰寫技術文檔與手冊：將 Week 1 到 Week 8 的研發歷程、8D 特徵技術規格、物理門檻參數（0.0005 變異數限制）以及系統啟動指令，完整整理為正式的部署指南（README.md 與技術白皮書）。

**預期產出**：

- 經過優化與封裝、可隨時部署上線的生產級代碼。

- 全套專案交付文件（包含系統架構圖、UAT 測試報告與環境部署手冊）。

- 專案正式結案。

---

## 📋 驗收數據參考 (Stage 5 Batch Report)

執行 `python main.py` 的預期表現：

| 受試者群組 | 品質合格率 | 說明 |
|-----------|-----------|------|
| S01 - S04（低活動組） | 3.5% - 16.5% | 大量攔截為**正確行為** |
| S10（黃金標準） | 約 20.6% | 僅在動態執行區間通過，靜態區間自動攔截 |

> 品質合格率不受 Fix 3 影響（`ClinicalQualityGate` 只看 Grav_Y，跟 Acc/Mag Z-score 是分開的路徑）。Fix 3 修的是「AI 辨識 L7 次數」這欄——修復前恆為 0，修復後應能反映真實 Label 7 辨識率（全受試者平均約 92.6% 準確率）。

---

## 📝 指令給下一位 AI

Stage 8 的 A（髒數據防護）、D（緩衝區復原）、F（回歸測試基礎建設）與 cp950 編碼修正已完成，請接續執行 **Stage 8 的 B、C、E**。

**首要任務**：執行 E 的長時間壓力測試（以 `get_health_stats()` 監控緩衝區），並重新評估 B 是否值得實作。

**動手前請先執行 `pytest`**：專案現有 73 項回歸測試。若要動到推論路徑，準確率基準測試（`tests/test_model_accuracy_baseline.py`）可直接驗證行為未變。任何修改後測試必須維持全綠。

**完成 B/C 後**：執行 E 的長時間壓力測試（以 `get_health_stats()` 監控緩衝區），並整理交付文件。
