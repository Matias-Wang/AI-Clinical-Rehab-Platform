# main.py (Week 5 最終驗收報告版)
from tensorflow.keras.models import load_model
from schema import (
    load_and_preprocess_subject, 
    ClinicalQualityGate, 
    RealTimeBiofeedbackEngine,
    ACTIVITY_LABELS
)
import numpy as np

def run_batch_assessment():
    print("====================================================")
    print("   AI 臨床復健平台 - Week 5 產品化批量驗收報告   ")
    print("====================================================\n")

    # 1. 載入 v3 模型
    model_v3 = load_model("models/clinical_rehab_model_v3.keras")

    # 2. 準備黃金範本 (S10 Label 7)
    df_s10_X, _ = load_and_preprocess_subject(10, "data/")
    golden_template = df_s10_X[0] # 取第一個標準視窗作為物理標竿

    # 3. 初始化引擎 (預留 10 位受試者的統計字典)
    report_summary = []

    # 4. 遍歷 S1 - S10
    for sid in range(1, 11):
        X_test, y_test = load_and_preprocess_subject(sid, "data/")
        if X_test is None: continue

        gate = ClinicalQualityGate(golden_template)
        engine = RealTimeBiofeedbackEngine(gate, golden_template, model_v3)
        
        halt_count = 0
        proceed_count = 0
        correct_label_count = 0 # 針對 Label 7 的辨識統計
        
        # 模擬連續影格流
        flat_stream = X_test.reshape(-1, 8) 
        for i in range(len(flat_stream)):
            result = engine.process_live_frame(flat_stream[i])
            if result:
                if result['status'] == "HALT":
                    halt_count += 1
                else:
                    proceed_count += 1
                    # 統計 AI 是否正確辨識出該片段為 Label 7
                    if result['predict_label'] == 7:
                        correct_label_count += 1
        
        # 計算統計指標
        total_windows = halt_count + proceed_count
        pass_rate = (proceed_count / total_windows) * 100 if total_windows > 0 else 0
        
        report_summary.append({
            "sid": sid,
            "total": total_windows,
            "pass_rate": pass_rate,
            "halt": halt_count,
            "label7_hits": correct_label_count
        })
        print(f"✅ S{sid:02} 測試完成。品質合格率: {pass_rate:.1f}%")

    # 5. 印出最終驗收大表
    print("\n" + "="*60)
    print(f"{'受試者':<8} | {'總視窗數':<8} | {'品質合格率':<10} | {'AI 辨識 L7 次數':<10}")
    print("-" * 60)
    for r in report_summary:
        print(f"S{r['sid']:02}{'':<6} | {r['total']:<10} | {r['pass_rate']:>8.1f}%{'':<3} | {r['label7_hits']:<10}")
    print("="*60)
    print("註：S1-S4 預期合格率極低（<30%），符合臨床品質閘門攔截標準。")

if __name__ == "__main__":
    run_batch_assessment()