# inference_test.py (Step 5 最終驗證版)
from schema import ClinicalQualityGate, RealTimeBiofeedbackEngine
import numpy as np

# 1. 建立模擬的黃金範本 (應對應你從 S10 提取的 128x8 矩陣)
# 這裡為了演示，我們生成一個具有週期性的模擬數據
t = np.linspace(0, 1, 128)
mock_golden = np.zeros((128, 8))
mock_golden[:, 5] = np.sin(2 * np.pi * t) * 0.1  # 模擬標準的 Grav_Y 擺動幅度

# 2. 初始化引擎
gate = ClinicalQualityGate(mock_golden)
engine = RealTimeBiofeedbackEngine(gate, mock_golden)

print("🧪 執行 Step 5：高品質成功案例測試...")

for i in range(128):
    # 逐幀餵入數據，並加入微量噪音模擬真實感測器
    dynamic_frame = mock_golden[i] + np.random.normal(0, 0.001, 8)
    
    result = engine.process_live_frame(dynamic_frame)
    
    if result:
        print(f"\n--- [視窗分析結果：動態達標] ---")
        print(f"🔹 狀態: {result['status']}")
        print(f"🎨 UI 顏色: {result['ui_color']}")
        print(f"📈 綜合得分: {result['score']:.1f}")
        print(f"🎯 姿態相似度: {result['similarity']:.1f}")
        print(f"💬 臨床建議: {result['msg']}")