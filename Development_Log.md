# AI 專案技術存檔：LiteCNN 運動訊號分類系統 (v3.1_multi)
## 1. 專案背景與目標 (Project Overview)
- 核心目標：開發一套能處理多受試者（10人）運動感測數據的輕量化分類模型，建立智慧醫療與運動科技的技術儲備。
- 應用場景：穿戴式裝置動作辨識、長照防跌偵測。
- 開發者背景：具備醫療專案管理 (Cathay General Hospital PM) 與 AI 產品架構 (CTBC AI Architect) 經驗，具備統計學專業基礎 。

## 2. 數據規格 (Data Specification)
- 輸入維度 (Input Shape)：$128 \times 4$（代表時序長度 128，特徵維度 4，如加速規三軸 + 總加速度）。
- 受試者規模：由 v3 (1人) 擴展至 v3.1_multi (10人)，提升模型泛化能力。
- 資料切分：採用 train_test_split (80/20)，並使用 stratify=y 確保 13 個類別在訓練與驗證集中的分佈一致。

## 3. 模型架構設計 (Model Architecture)
本架構採 LiteCNN 設計，兼顧推論效率與特徵提取能力，適合部署於 Edge AI 環境。

    Python# v3.1_multi 架構代碼：
    from tensorflow.keras import layers, models
    brain_v3_1_multi = models.Sequential([
        # 特徵提取層：捕捉時序訊號中的局部動作模式
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(128, 4)),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Flatten(),

        # 決策層
        layers.Dense(64, activation='relu'), 
        
        # 正則化：由 0.5 調優至 0.2，解決欠擬合問題
        layers.Dropout(0.2), 
        
        # 輸出層：13 類動作分類
        layers.Dense(13, activation='softmax') 
    ])

## 4. 訓練策略與實驗紀錄 (Training & Optimization)
- 優化歷程：
    * Iteration 1 (v3_1_big)：使用 Dropout(0.5)，導致 Training Accuracy 僅 ~80%，低於 Validation，判定為過度正則化導致的欠擬合。
    * Iteration 2 (v3_1_multi)：優化 Dropout 至 $0.2$，模型成功收斂。
- 性能指標：
    * Training Accuracy: ~93%
    * Validation Accuracy: ~91%
    * Loss 趨勢: 驗證損失在 Epoch 35 後出現輕微震盪，顯示模型已達該架構性能上限。
    
## 5. 性能瓶頸與診斷 (Evaluation & Diagnostics)
透過 Confusion Matrix 識別出系統性問題：
1. 靜態動作混淆 (Stand vs. Sit)：加速度規訊號高度相似，單靠原始波形難以完全區分。
2. 動態頻率干擾 (Walk vs. Stairs)：模型對於垂直位移特徵捕捉不足。
3. 小樣本問題 (Jump)：樣本數不足 50 筆，未來需補強數據增強 (Data Augmentation)。

## 6. 未來演進路徑 (Future Roadmap)
- 架構升級：評估引入 Transformer (Self-Attention) 機制以處理長序列動作關聯。
- 特徵工程：導入傅立葉轉換 (FFT) 或小波轉換 (Wavelet Transform) 提取頻域特徵。
- 產品化：結合 AWS Cloud 端架構 與醫療法規 (SaMD)，設計自動化 MLOps 流程 。

## 20260228 數據診斷與一致性分析報告
# 1. 受試者間分佈一致性 (KS Test)
針對 10 位受試者的加速度量值（Magnitude）進行 Kolmogorov-Smirnov 檢定，量化個體間的分佈偏移：
- 關鍵離群值：受試者 S1 與 S10 表現出最高的分佈差異，其 $D$ 統計量最高達 0.15。
- 高一致性群組：S4 與 S9 的分佈極為接近，其 $D$ 統計量僅為 0.012。
- 結論：模型在面對 S1 與 S10 時可能需要額外的數據正規化處理，以克服約 $15\%$ 的分佈偏移。

## 2. 物理邊界與靜止標籤稽核 (Physical Audit)
針對 Sitting (Label 2) 動作進行物理真實性檢查，基準值為地球重力 $9.81\ m/s^2$：
- 瞬時雜訊觀察：受試者 S2 與 S8 的 Magnitude 數據出現突破 $11.0\ m/s^2$ 的尖峰，代表存在動作標籤污染。
- 感測器偏置 (Bias)：受試者 S3 的中位數明顯低於理想重力線，判定為硬體偏移或配戴角度偏差。
- 雜訊比例量化：
    * S2 物理違規比例：0.03%。
    * S8 物理違規比例：0.19%。
- 結論：由於雜訊佔比極低（$< 0.2\%$），可在預處理階段安全剔除大於 $10.8\ m/s^2$ 的坐姿視窗，而不損害數據真實性。

##　3. LOSO 模擬：受試者數據容量分析
模擬 Leave-One-Subject-Out (LOSO) 切分，評估各受試者作為測試集時的代表性：
- 最高容量：受試者 S2 擁有最多的有效視窗數（約 550 個）。
- 最低容量：受試者 S6 與 S7 的數據量相對較少（約 500 個）。
- 結論：全體受試者的有效視窗數均在平均線（約 530）附近波動，數據量足以支持穩定且公平的 10-Fold 交叉驗證。

## 20260228 診斷總結與後續行動
- 數據清洗：確定將在 schema.py 邏輯中加入物理濾鏡，剔除 Sitting 狀態下的瞬時離群點。
- 模型方向：針對 S1/S10 的偏移，20260301 後的實驗將導入受試者層級的標準化（Per-subject Normalization）。
- 驗證協議：捨棄隨機切分，全面改採基於受試者編號的 LOSO 交叉驗證訓練。

## 20260308 實驗紀錄：重力特徵提取與預處理策略修正
### 1. 特徵擴展與物理濾波
- 引入重力向量：在 `schema.py` 實作低通濾波分離技術，將特徵維度從 4 維（Acc+Mag）提升至 7 維（新增 $Grav\_X, Y, Z$）。
- 相位修正：將 `lfilter` 替換為 `filtfilt` (零相位濾波)，確保重力分量與原始動作訊號在時間軸上完全對齊。

### 2. 選擇性標準化 (Selective Normalization) 實驗
- 失敗案例：對 7 個特徵統一執行受試者級別 $Z$-score。導致重力絕對值歸零，S2 準確率重挫至 $61.7\%$。
- 成功案例：
    * 前 4 欄 (Acc/Mag)：執行受試者 $Z$-score，消除個體強度差異。
    * 後 3 欄 (Gravity)：執行全局固定縮放 (Global Scaling, $/10.0$)，保留物理角度參考值。
- 結果：S3 性能提升至 $91.8\%$，S2 回升至 $70.5\%$。

### 3. 本週剩餘性能瓶頸
- S2 誤判診斷：混淆矩陣顯示 `Sitting` 與 `Waist Bends` 依然高度混淆。主因為時域角度特徵無法區分「靜止傾斜」與「動態傾斜」。
- 下週路徑：計畫於 `schema.py` 導入 FFT (快速傅立葉轉換) 特徵，利用頻譜能量區分靜態與動態動作。


## 2026/04/03 實驗紀錄：特徵工程 
* **混合域特徵 (8-D Input)**:
    * `Time-Domain`: X, Y, Z, Mag, Gravity_X/Y/Z (時域物理量)。
    * `Frequency-Domain`: FFT Spectral Energy (頻譜總能量)。
* **頻域處理邏輯**: 
    * 使用 `rfft` 計算 Mag 欄位能量。
    * 執行 `log1p(x) / 5.0` 縮放，解決能量值跨度過大導致的梯度問題。

##  2026/04/04  研發歷程紀錄：從演算法補償轉向臨床品質控制
### 1. 核心問題識別 (Problem Identification)
- 現象紀錄：發現受試者數據存在明顯的「治療師集群效應（Therapist Clustering）」。
- 數據特徵：S1-S4 群組在標籤 7（側向動作）的 $Grav\_Y$ 變異數極低（< 0.0003），疑似因不同治療師引導或感測器配戴習慣，導致關鍵運動平面訊號缺失。
    
### 2. 實驗與驗證 (Experiments & Validation)
- 壓力測試 (Stress Test)：執行跨群組交叉驗證。結果：當模型只看過高活動數據（S5-S10）去預測低活動數據（S1-S4）時，準確率從 92% 崩跌至 68.89%。
- 座標對齊實驗 (Coordinate Alignment)：實作 Rodrigues' 旋轉公式試圖物理性校正佩戴偏差。失敗分析：對齊後測試 B 準確率反而下降至 62.31% (-6.58%)。結論：在極低訊噪比環境下，數學補償會放大隨機噪音，證實「後端補償」不如「前端引導」。

### 3. 技術決策與實作 (Technical Decision)
- 策略轉向：確立 Data-Centric AI 路線，放棄黑箱補償，改採「臨床品質閘門（Clinical Quality Gate）」。
- 物理標竿設定：
    - 黃金標準 (Gold Standard)：S10 ($Grav\_Y\_Var = 0.001595$)。
    - 及格閾值 (Min Safe Threshold)：0.0005 (基於 S7/S9 邊緣案例診斷，確保 90%+ 準確率)。
- 模組化封裝：完成 ClinicalQualityGate 類別，成功攔截 Subject 2 的無效動作（品質評分：20.9 分），解決了全量數據導致的偽陽性 Bug。

##  2026/04/06  研發歷程紀錄：核心推論引擎產品化與批量臨床驗收
本週完成從實驗環境（Jupyter Notebook）向生產環境（Python Scripts）的全面遷移，並建立自動化驗收管線：
1. 技術遷移與架構重構 (Productization)
    - 核心組件封裝：將 20260404_Project_Rehab_Optimization.ipynb 中的離線邏輯遷移至 schema.py。
        - 實作 RealTimeStreamProcessor：建立支援 50% 重疊率（64 steps stride）的滑動視窗緩衝區。
        - 實作 RealTimeBiofeedbackEngine：整合品質閘門與相似度計算的即時決策中心。
    - 入口檔案建立：建立 main.py 作為系統啟動點，支援單一受試者模擬會期與 S1–S10 批量自動化測試。

2. 關鍵技術修復：訓練與推論一致性 (Training/Serving Skew Fix)
- 重力特徵縮放校準：修復 schema.py 中遺漏的全局縮放步驟。依據 SPEC.md 第 2.2 節，將 Gravity_X/Y/Z 分量進行 ÷10.0 處理。
    - 修正影響：此修正移除了因數值放大導致的虛假變異數（原本放大 100 倍），恢復了 ClinicalQualityGate 在低訊噪比環境下的攔截能力

3. 生物回饋算法與 UI 狀態設計
姿勢相似度引擎：實作基於歐幾里德距離的映射公式：$Score_{sim} = \max(0, 100 - Distance \times 15)$。
綜合評分體系：確立臨床評分加權比重為 品質 (40%) + 相似度 (60%)。
UI 顏色狀態設計：定義紅、黃、綠三色引導邏輯，並成功驗證 S10 數據在標準動作下能觸發 GREEN (🟢) 訊號。

4. 批量驗收報告 (S1–S10 Batch Assessment)
執行全量受試者數據測試，驗證系統在真實數據流下的表現：

| 受試者群組 | 品質合格率 (平均) | 臨床診斷結論 |
|-------------|--------------------|---------------|
| 低活動組 (S01–S04) | 3.5% ~ 16.5% | 🔴 成功攔截：有效排除治療師集群效應導致的無效訊號。 |
| 高活動組 (S05–S10) | 17.7% ~ 25.3% | 🟡 狀態辨識：僅在真實動態復健區間啟動推論，排除靜態休息片段。 |
| 黃金標準 (S10) | 20.6% | ✅ 基準效應：合格率為 S01 的 6 倍，證明具備極佳的雜訊抑制能力。 |

5. 技術規格變動 (SPEC.md Update)
    - 更新 8.5 節：定義姿勢相似度演算法與公式。
    - 更新 8.6 節：正式納入 UI 顏色狀態設計與臨床提示訊息準則。

##  2026/07/11  研發歷程紀錄：演算法魯棒性優化（Stage 6 — DTW 相似度）
### 1. 核心問題識別
- 現象紀錄：舊版動作相似度比對採用歐幾里德距離，屬點對點的剛性計算。受試者只要節奏或速度與 S10 黃金範本相差約 0.05 秒，分數即暴跌並誤觸黃燈（🟡）。
- 臨床意涵：此類誤判懲罰的是「節奏差異」而非「姿勢錯誤」，與復健生物回饋的目的相悖——動作正確但速度略慢的患者不應被系統判定為做錯。

### 2. 技術決策與實作
- 新增 `dtw_distance(seq_a, seq_b, radius=16)`：純 numpy 手刻 DTW，採 Sakoe-Chiba band 限制（radius=16 samples ≈ 0.32 秒 @ 50Hz），local cost 取平方差、最終距離開根號，與原歐幾里德距離維持同一尺度。
- `RealTimeBiofeedbackEngine.calculate_similarity()` 改用 `dtw_distance()` 取代 `np.linalg.norm()`；`__init__()` 新增可調參數 `dtw_radius`（預設 16）。

### 3. 免重新校準的理由（關鍵設計決策）
- 對角線路徑恆為 DTW 搜尋空間中的合法解，故 DTW 距離恆 ≤ 對應的歐幾里德距離。
- 由此推得評分公式的 `× 15` 係數與 GREEN/YELLOW 門檻皆無需重新校準，`ClinicalQualityGate` 完全未動。此性質讓 Stage 5 的批量驗收數字得以直接沿用，避免整套臨床門檻重新驗證的成本。

### 4. 驗證與未竟事項
- 已執行 `python main.py` 端到端測試，S01–S04 合格率仍明顯低於 S10，攔截行為與 Stage 5 基準一致。
- 尚未涵蓋：多標籤模型適配性測試。Label 7（手臂前舉）以外的其他復健動作標籤，其即時推論與相似度比對尚未驗證。

##  2026/07/11  研發歷程紀錄：系統整合與 UI 橋接（Stage 7）
### 1. 核心問題識別
- 系統輸出（品質報告、AI 預測、相似度分數）僅能在終端機以文字呈現，未與任何前端畫面接軌，無法作為真正的「生物回饋」介面。

### 2. 技術實作
- 新增 `ui_bridge.py`：以 `asyncio` + `websockets` 建立 WebSocket 伺服器（預設埠 8765），模擬指定受試者的逐幀串流，並將每個視窗的推論結果即時 broadcast 給所有已連線 client（JSON 格式，見 SPEC.md §9.5）。支援 `--subject`／`--port`／`--speed`／`--realtime` 參數。
- 設計原則：完全重用 `schema.py` 既有的 `load_and_preprocess_subject`、`ClinicalQualityGate`、`RealTimeBiofeedbackEngine`、`ACTIVITY_LABELS`，不修改任何既有推論邏輯，確保橋接層不引入行為偏移。
- 新增 `frontend/demo.html`：自包含 HTML/JS 最小展示前端（無框架、無 build tooling），以瀏覽器原生 WebSocket API 連線，顯示紅/黃/綠燈號與 final_score／similarity／AI 辨識動作。
- 依賴新增 `websockets`（寫入 `requirements.txt`，保留其原有 UTF-16 編碼）。

### 3. 驗證項目
- 伺服器啟動流程（STEP 1–4）正常。
- 單一 client 收發正常；雙 client 同時連線收到相同 broadcast。
- client 異常斷線時伺服器不崩潰，並正確清理連線集合。
- `python main.py` 既有驗收流程不受影響。

### 4. 尚未涵蓋
- 正式的 3D 前端視覺組件（`demo.html` 僅為驗證串接用的 2D 展示頁）。
- 50Hz 真實速率下的長時間延遲與穩定性壓力測試（留待 Stage 8）。

##  2026/07/12  研發歷程紀錄：訓練與推論一致性重大修復（Fix 3）
本次為專案至今最隱蔽的一個 Bug，紀錄重點在於「如何被發現」而非僅是「如何修復」。

### 1. 現象
- 模型從未預測過 Label 1、5、6、7——12 類中有 4 類完全消失。
- 真實 Label 7 視窗有 98% 以上被誤判為 Label 10。
- `main.py` 報告中的「AI 辨識 L7 次數」因此恆為 0，而 Label 7 正是黃金範本所使用的動作。

### 2. 根因
對照訓練用 notebook（`development_history/20260404_Project_Rehab_Optimization.ipynb`）逐步比對後，發現 `load_and_preprocess_subject()` 遺漏兩個關鍵步驟：
1. Acc_X/Y/Z/Magnitude 的**受試者級 `StandardScaler` Z-score**。
2. FFT 能量的 `log1p(x) / 5.0` 縮放。

兩者都只存在於 SPEC.md 與訓練 notebook 的文件記載中，Week 5 產品化遷移時未真正落實到程式碼——典型的 Train/Serving Skew。

### 3. 順序陷阱（本次最重要的教訓）
- FFT 能量必須以**原始尺度**的 Magnitude 計算，因此 Z-score 一定要排在 FFT 廣播**之後**。
- 實驗中曾將順序顛倒（Z-score 排在 FFT 之前），結果整體準確率反而崩落至約 5–48%。
- 危險之處在於：顛倒版本「看起來像修好了」（4 個消失的類別確實重新出現），實際上比修復前更糟。**唯一能揭穿它的是對照 notebook 的基準準確率（S10=0.9029、S02=0.8430）**。
- 結論：涉及特徵管線的修復，必須以數值基準而非「現象消失」作為驗收標準。

### 4. 修復內容
- `load_and_preprocess_subject()` 補回正確順序的 Z-score 與 FFT 縮放。
- `extract_window_fft_energy()` 回傳值已含 `log1p / 5.0` 縮放。
- 新增 `extract_golden_template(X, y, target_label=7)`，修正 `main.py`／`ui_bridge.py` 原本直接取 `X[0]` 當黃金範本的問題（`X[0]` 未必是 Label 7 動作）。

### 5. 修復後驗證

| 受試者 | 準確率 | 與 notebook 記錄 |
|--------|--------|------------------|
| S02 | 0.8430 | 完全吻合 |
| S07 | 0.9045 | 完全吻合 |
| S09 | 0.9065 | 完全吻合 |
| S10 | 0.9029 | 完全吻合 |

全受試者平均準確率回復至 92.6%。

### 6. 影響範圍與已知限制
- 僅影響 Acc/Mag 與 FFT 兩條特徵路徑；`ClinicalQualityGate`（Grav_Y）與 Stage 6 的 DTW 相似度計算使用獨立路徑，不受影響，Stage 5–7 的品質閘門驗收數字維持有效。
- 已知限制：Z-score 為受試者級統計量，需該受試者完整視窗集合才能計算，僅適用於目前「預先載入整個受試者、重播模擬即時串流」的架構。若未來接上真正逐筆即時的穿戴式硬體，需另外設計線上／滾動標準化策略。

##  2026/08/30  研發歷程紀錄：臨床串流防護、緩衝區復原與測試基礎建設（Stage 8 — A/D/F）
Stage 8 依風險排序拆為 A（髒數據防護）、B（推論阻塞）、C（記憶體成長）、D（緩衝區復原）、F（回歸測試）。本次完成 A、D、F 三項——此三項互相咬合且屬臨床安全層級，優先於效能議題。

### 1. 髒數據防護（A）
- **核心問題**：`ClinicalQualityGate.get_quality_report()` 缺少 NaN 防護。當感測器噪音產生 NaN 時 `var_y` 為 NaN，而 `NaN < MIN_SAFE_LIMIT` 在 IEEE 754 下**恆為 False**，導致髒數據被判定為「品質良好」並直接送入模型推論——品質閘門在最需要攔截的情況下完全失效。
- 品質閘門新增 `np.isfinite(var_y)` 檢查，採 fail-safe 原則：無法判定品質時一律攔截。
- 新增 `RealTimeStreamProcessor.validate_frame()`，於影格進入緩衝區**之前**攔截維度錯誤、NaN/Inf、非數值型別與超出生理範圍的資料。
- 新增 `SensorStreamError` / `DirtyFrameError` / `SensorDisconnectedError` 例外階層；連續 25 幀髒數據（0.5 秒 @ 50Hz）升級判定為感測器斷訊。
- **設計要點**：邊界檢查採**標準化後尺度**（Z-score 50 個標準差），而非沿用文件中的 50 m/s² 原始物理門檻。原因是進入串流處理器的資料已經過 `load_and_preprocess_subject()` 標準化（Acc/Mag 為 Z-score、Grav 已 ÷10.0、FFT 已 log1p/5.0），若沿用原始物理門檻，該檢查將完全失效。

### 2. 緩衝區復原（D）
- **核心問題**：髒影格一旦進入 `deque`，後續最多 `window_size`（128）步的視窗都會沿用被污染的緩衝，系統會輸出「看起來正常但基於污染資料」的評分。原 `ui_bridge.py` 的例外處理只印出錯誤就繼續，造成例外的資料仍殘留在緩衝區內。
- 新增 `reset_buffer()`，攔截髒影格時同步清空 deque 與步長計數器，強制重新累積完整的乾淨視窗。
- `process_live_frame()` 將例外轉為紅燈結果而非向外拋出，使長時間串流不因單一壞封包而中斷；並僅在「乾淨→髒」的狀態轉換點輸出，避免雜訊期間的洪水式推播。
- `ui_bridge.py` 的 except 區塊補上 `engine.reset_buffer()`。
- 結果新增 `reason` 欄位（`OK` / `LOW_QUALITY` / `DIRTY_DATA` / `DISCONNECTED`），`status` 維持原值，因此 `main.py` 與 `demo.html` 無需任何改動。
- 新增 `get_health_stats()`，回傳總影格數、髒數據比例、緩衝區重置次數等指標，供後續長時間壓力測試監控使用。

### 3. 回歸測試基礎建設（F）
- 專案在此之前**沒有任何自動化測試**。Fix 2、Fix 3 這類致命修正僅存在於文件記載，缺乏程式碼層級的保護。
- 新增 `tests/`，共 72 項測試，涵蓋預處理修正、品質閘門、串流處理器、DTW 數學性質、引擎狀態機與模型準確率基準。
- **Fix 3 順序陷阱的精確不變量**：本次找到一個免載入 TensorFlow 即可攔截順序錯誤的判定方式——若 Z-score 被誤排到 FFT 之前，儲存於第 7 欄的能量值會「恰好等於」以最終輸出（已標準化）的 Magnitude 重算的結果。直接斷言兩者必須不同，即可在 3 秒內攔截 ROADMAP 記載「必須靠 notebook 基準準確率才抓得出來」的 Bug。
- **準確率基準測試**標記為 `slow`，對照訓練 notebook 的 S02／S07／S09／S10 數值，缺少 TensorFlow 或模型檔時自動 skip。

### 4. 突變測試驗證（測試品質的驗收）
通過但無法失敗的測試沒有價值。逐一將四個已修復的回歸重新注入程式碼，確認測試如實失敗：

| 注入的回歸 | 失敗測試數 |
|-------------|------------|
| 移除 Fix 2 的重力 ÷10.0 | 2 |
| Z-score 提前至 FFT 之前 | 7（快速測試與準確率基準雙層攔截）|
| 移除品質閘門 NaN 防護 | 4 |
| 移除髒影格緩衝區清空 | 3 |

### 5. cp950 輸出編碼修正
- **現象**：`python main.py > out.log` 會在印出第一位受試者結果時崩潰（`UnicodeEncodeError: 'cp950' codec can't encode character`），批量驗收報告因此永遠無法完整輸出。
- **根因**：Windows 繁中環境 locale 預設為 cp950。標準輸出被重導至管道或檔案時，stdout 不再是主控台而退回 locale 編碼，訊息中的 ✅、⚠️、❌、🔌 等字元無法編碼。掃描後確認問題橫跨 `main.py`、`inference_test.py` 與 `schema.py` 三處；`schema.py` 的字元位於品質閘門與斷訊訊息中，會一路送到前端燈號顯示，因此不能以「移除 emoji」的方式解決。
- **修復**：新增 `console.py` 提供 `enable_utf8_output()`，於各入口點將 stdout/stderr 設為 UTF-8。直接在主控台執行時 Python 本就使用 Windows Unicode API，此設定為無害的 no-op；被重導時才生效。另加上 `errors="replace"` 作為第二道保險。回歸測試以子行程強制 `PYTHONIOENCODING=cp950` 重現，並含對照組證明測試具鑑別力。

### 6. 專案維護
- 刪除 `inference_test.py`：呼叫 `RealTimeBiofeedbackEngine` 時缺少 `model` 參數，已無法執行；其驗證範圍由 `tests/test_biofeedback_engine.py` 涵蓋。
- Git 歷史瘦身：217MB 的 `data_raw/` 資料集曾被 commit 進版控，雖已於先前移除工作區檔案，blob 仍留在歷史中使 `.git` 達 95MB。以 `git filter-repo` 清除後降至 **7.7MB**，21 個 commit 全數保留。操作前建立完整 bundle 備份並實測還原。

### 7. 驗收確認
- `python main.py` 端到端通過，Stage 5 品質合格率完全保留（S01=3.5%、S02=5.1%、S03=16.5%、S04=15.1%、S10=20.6%）。
- `ui_bridge.py` 實測 WebSocket 連線，正常推播含 `reason` 欄位的結果。
- 全套 72 項測試通過。

### 8. 後續（Stage 8 剩餘）
- **B — 推論阻塞 event loop**：`model.predict()` 為同步呼叫，跑在 asyncio 主迴圈上，每次推論會凍結整個 WebSocket 伺服器。規劃以 `asyncio.to_thread()` 隔離。
- **C — 記憶體成長**：`model.predict()` 在迴圈中重複呼叫會累積 function trace，為已知的記憶體成長來源。規劃改用 `model(input_data, training=False)`。
- **壓力測試腳本**：長時間循環全量 S1–S10，以 `get_health_stats()` 監控緩衝區穩定性。
- **交付文件**：部署指南與技術白皮書。

##  2026/08/31  研發歷程紀錄：即時推論路徑最佳化（Stage 8 — C）
### 1. 原規劃與實測的落差
- 原 Roadmap 記載 C 應改用 `model(input_data, training=False)`，理由是「`predict()` 在迴圈中重複呼叫會累積 function trace」——這是 Keras 的一般性通則。
- 動手前先做實測，結果**推翻了此規劃**：在本專案的 Keras 3.13.2 環境下，`model(x, training=False)` 反而是四個候選方案中最慢的。
- 教訓：套用框架通則前必須先量測。此專案的 Fix 3 已有一次「文件記載與實際行為不符」的教訓，效能決策同樣不能靠假設。

### 2. 四方案實測（單筆 `(1, 128, 8)` 輸入，已預熱，150 次取中位數）

| 方案 | 延遲中位數 | p95 | 2000 次後 RSS | 輸出一致性 |
|------|-----------|-----|---------------|-----------|
| `predict()`（原實作） | 94.91 ms | 212.65 ms | +5.5 MB | 基準 |
| `model(x, training=False)` | 145.64 ms | 420.59 ms | +0.0 MB | 完全一致 |
| **`predict_on_batch()`（採用）** | **2.49 ms** | **3.38 ms** | **+0.0 MB** | 完全一致 |
| `tf.function()` 包裝 | 2.57 ms | 3.59 ms | +0.0 MB | 完全一致 |

- 根本原因：`predict()` 為批次導向 API，每次呼叫都會建立 data adapter 與 tf.function 追蹤。這些開銷攤提在大批次上可忽略，但用於「單一視窗、高頻重複」的即時推論時完全主導了執行時間。
- `predict_on_batch()` 與 `tf.function()` 包裝效能相當，選前者是因為不需額外程式碼與輸入簽章維護。

### 3. 安全性驗證
- **數值一致性**：四個方案輸出逐位元相同（最大絕對差異 `0.000e+00`，argmax 全部一致）。此為採用的前提——臨床系統的效能改善不得以任何預測行為變動為代價。
- **端到端**：`python main.py` 由 **256 秒降至 44 秒**（5.8 倍），且輸出與修改前**逐行完全相同**（以 diff 驗證）。
- **回歸守門**：新增 `test_uses_predict_on_batch_not_predict`，stub model 分別計數兩條推論路徑，斷言 `predict()` 呼叫次數為 0。經突變測試確認：改回 `predict()` 時該測試如實失敗。

### 4. 對 B 的連帶影響
- B（以 `asyncio.to_thread()` 隔離推論）的急迫性建立在「推論很慢」之上。原本 94.91 ms 的同步推論跑在 asyncio 主迴圈上確實嚴重：以 `--speed 20` 計算，每 64 ms 產生一個視窗，阻塞佔比約 148%——伺服器根本追不上串流。
- C 完成後推論降至 2.49 ms，同一條件下阻塞佔比降至約 4%；真實 50Hz 播放時更低於 0.2%。
- **決策**：B 改列為「待重新評估」。先做 B 等於把 95 ms 的問題搬進執行緒藏起來而非解決它；C 先行才是正確順序。是否仍需引入執行緒的複雜度，待實測 event loop 實際延遲後再定。

### 5. 未竟事項
- `predict()` 的記憶體增量在觀察窗內呈遞減趨勢（每 500 次推論：+3.6 → +4.4 → +5.4 → +5.5 MB 累計），尚未確認是趨於平穩或持續線性成長。此問題不影響本次決策（38 倍加速本身已足夠成立），但若要為 Stage 8 的 Memory Leak 驗證目標留下正式證據，仍需補做長時間量測。

##  2026/09/06  研發歷程紀錄：串流重建缺陷與驗收報表修正（稽核 B1／B3）
### 1. B1 的發現與嚴重性升級
- 全面稽核時發現 `main.py` 與 `ui_bridge.py` 以 `X.reshape(-1, 8)` 產生「模擬即時串流」，但 `X` 是**已 50% 重疊**的視窗集合。實測 S01：
  - 串流長度 70,144 步，為真實訊號 35,136 步的 **2.00 倍**——每個樣本重複出現兩次
  - 每 128 步在時間軸倒退 64 步，接縫處 Grav_Y 跳變達視窗內部的 **53 倍**
- 動手修正前再次量測，發現問題比原先評估的更嚴重：引擎重新切窗後，**恰好 50% 的視窗橫跨兩個原始視窗**，使第 8 維 FFT 能量在單一視窗內出現兩個不同值。而訓練資料中該欄永遠是整個視窗的單一常數（實測 200 個訓練視窗，相異值數皆為 1）。
- 這是與 Fix 3 同類的 **train/serving skew**：模型有一半的推論建立在訓練時從未見過的輸入形態上。

### 2. 修法的約束：為何只能逐視窗
- 直覺的修法是「重建連續訊號」（取首個視窗 + 後續各視窗的後半段）。但這只能修掉 2 倍膨脹與時間不連續，**修不掉 FFT skew**——重新切窗後仍會橫跨兩個原始視窗。
- 根本原因：FFT 能量是**視窗級**特徵，且必須以**原始尺度**的 Magnitude 計算（Fix 3 的順序陷阱）。逐幀串流中的 Magnitude 已經過 Z-score，引擎無法自行重算。
- 結論：在現行架構下，逐視窗評估是唯一能餵入正確輸入形態的作法。

### 3. 實作
- 自 `process_live_frame()` 抽出 `evaluate_window()`，供串流路徑與批量／重播路徑共用，確保判定邏輯完全一致（抽出後以同一視窗比對兩條路徑，結果逐位元相同）。
- `main.py` 改為逐視窗評估；`ui_bridge.stream_subject()` 改為逐視窗推播，並維持真實時間節奏（相鄰視窗相距 stride 幀，`--realtime` 下間隔 1.28 秒）。
- **取捨**：`RealTimeStreamProcessor` 的逐幀緩衝區不再被生產路徑使用。此為經評估後的決定——正確的輸入形態優先於「展示串流緩衝」；該類別仍由 `tests/test_stream_processor.py` 完整覆蓋，並保留為未來接入真實硬體的 API。

### 4. B3：讓驗收報表可判讀
- `main.py` 一直載入 `y_test` 卻從不使用。「AI 辨識 L7 次數」是模型預測 Label 7 的**原始次數**，無從判斷正確與否——S08 顯示 17 次為全體最高，但看不出是辨識能力強還是誤判多。
- 改以真實標籤計算 **L7 召回率**：S08 實際有 15 個 L7 視窗通過閘門，召回率 93.3%，確認是前者。
- 無 L7 視窗通過閘門時顯示「—」而非 0.0%，避免與「全部辨識錯誤」混淆（S01／S02 屬此情形）。

### 5. 驗收基準變更

| | 舊基準 | 新基準 |
|---|---|---|
| S01 總視窗數 | 1095 | **548**（等於真實預處理視窗數）|
| S10 總視窗數 | 1049 | **525** |
| S01 品質合格率 | 3.5% | 5.5% |
| S10 品質合格率 | 20.6% | 22.5% |
| L7 欄位 | 預測次數（無從判讀）| 召回率 |

- 模型準確率基準（S02=0.8430 等）**不受影響**——該測試直接使用預處理視窗，未經攤平路徑。此為修正正確性的佐證：真正的模型行為沒有改變，改變的是評估管線餵給它什麼。
- 端到端：`ui_bridge --realtime` 視窗推播中位數 1.279 秒（理論 1.280 秒，達成率 100.1%），抖動由 0.236 秒降至 0.064 秒——較少但較大的 sleep 反而更穩定。
- `main.py` 執行時間由 44 秒降至 19 秒（視窗數減半）。

### 6. 教訓
- 「模擬即時串流」這個名稱掩蓋了實作與真實訊號的落差。稽核時若只看程式碼是否會出錯，不會發現任何問題——`reshape(-1, 8)` 語法完全正確、跑起來也不會拋例外。**是量測資料的物理性質（長度膨脹倍率、接縫跳變幅度、特徵形態分布）才暴露出來的。**
- 這與 Fix 3 的教訓一致：涉及特徵管線的正確性，必須以數值驗證，不能只看程式碼是否合理。
