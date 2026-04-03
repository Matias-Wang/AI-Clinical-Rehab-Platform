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