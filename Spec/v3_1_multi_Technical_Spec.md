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