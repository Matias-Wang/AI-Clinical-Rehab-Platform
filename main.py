# main.py (Week 5 最終驗收報告版)
# Stage 8：必須先設定 UTF-8 輸出，避免輸出被重導時因 cp950 而崩潰
from console import enable_utf8_output

enable_utf8_output()

import sys

import numpy as np
from tensorflow.keras.models import Model, load_model

from schema import (
    ClinicalQualityGate,
    RealTimeBiofeedbackEngine,
    extract_golden_template,
    load_and_preprocess_subject,
)

# <使用者自訂變數>（請勿更動）
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGAENTA = "\033[95m"
RESET = "\033[0m"

MODEL_PATH = "models/clinical_rehab_model_v3.keras"
DATA_FOLDER = "data/"
GOLDEN_SUBJECT_ID = 10
GOLDEN_TARGET_LABEL = 7


def log_step(index: int, message: str) -> None:
    """輸出步驟啟動訊息。

    步驟訊息一律送往 stderr，驗收報表則留在 stdout。如此
    `python main.py > report.txt` 得到的是乾淨的報表，不會混入進度訊息，
    這也讓報表輸出可作為穩定的回歸基準（見 tests/test_acceptance_parity.py）。

    Parameters
    ----------
    index : int
        步驟序號。
    message : str
        步驟說明文字。
    """
    print(f"{GREEN}STEP {index}:{message}{RESET}", file=sys.stderr)


def log_error(index: int, error: Exception) -> None:
    """輸出步驟錯誤訊息。

    Parameters
    ----------
    index : int
        發生錯誤的步驟序號。
    error : Exception
        捕捉到的例外。
    """
    print(f"{RED}STEP {index} ERROR:{error}{RESET}", file=sys.stderr)


def assess_subject(
    subject_id: int,
    golden_template: np.ndarray,
    model: Model,
) -> dict | None:
    """對單一受試者執行完整的即時推論模擬並統計結果。

    Parameters
    ----------
    subject_id : int
        受試者編號 (1-10)。
    golden_template : numpy.ndarray
        shape (128, 8) 的黃金範本。
    model : keras.Model
        已載入的 v3 推論模型。

    Returns
    -------
    dict or None
        含 sid、total、pass_rate、halt、label7_hits 的統計字典；
        找不到該受試者資料時回傳 None。
    """
    X_test, y_test = load_and_preprocess_subject(subject_id, DATA_FOLDER)
    if X_test is None:
        return None

    gate = ClinicalQualityGate(golden_template)
    engine = RealTimeBiofeedbackEngine(gate, golden_template, model)

    halt_count = 0
    proceed_count = 0
    correct_label_count = 0  # 針對 Label 7 的辨識統計

    # 模擬連續影格流
    flat_stream = X_test.reshape(-1, 8)
    for i in range(len(flat_stream)):
        result = engine.process_live_frame(flat_stream[i])
        if result:
            if result["status"] == "HALT":
                halt_count += 1
            else:
                proceed_count += 1
                # 統計 AI 是否正確辨識出該片段為 Label 7
                if result["predict_label"] == 7:
                    correct_label_count += 1

    total_windows = halt_count + proceed_count
    pass_rate = (proceed_count / total_windows) * 100 if total_windows > 0 else 0

    return {
        "sid": subject_id,
        "total": total_windows,
        "pass_rate": pass_rate,
        "halt": halt_count,
        "label7_hits": correct_label_count,
    }


def print_report(report_summary: list[dict]) -> None:
    """印出最終驗收大表。

    Parameters
    ----------
    report_summary : list of dict
        各受試者的統計字典列表。
    """
    print("\n" + "=" * 60)
    print(f"{'受試者':<8} | {'總視窗數':<8} | {'品質合格率':<10} | {'AI 辨識 L7 次數':<10}")
    print("-" * 60)
    for r in report_summary:
        print(
            f"S{r['sid']:02}{'':<6} | {r['total']:<10} | "
            f"{r['pass_rate']:>8.1f}%{'':<3} | {r['label7_hits']:<10}"
        )
    print("=" * 60)
    print("註：S1-S4 預期合格率極低（<30%），符合臨床品質閘門攔截標準。")


def run_batch_assessment() -> int:
    """執行 S1–S10 批量驗收測試。

    Returns
    -------
    int
        全部順利完成回傳 0，發生無法繼續的錯誤則回傳 1。
    """
    print("====================================================")
    print("   AI 臨床復健平台 - Week 5 產品化批量驗收報告   ")
    print("====================================================\n")

    try:
        log_step(1, f"載入 v3 模型 {MODEL_PATH}")
        model_v3 = load_model(MODEL_PATH)
    except Exception as e:
        log_error(1, e)
        return 1

    try:
        log_step(2, f"準備黃金範本（S{GOLDEN_SUBJECT_ID:02} Label {GOLDEN_TARGET_LABEL}）")
        golden_X, golden_y = load_and_preprocess_subject(
            GOLDEN_SUBJECT_ID, DATA_FOLDER
        )
        if golden_X is None:
            raise ValueError(f"找不到受試者 S{GOLDEN_SUBJECT_ID:02} 的資料")
        golden_template = extract_golden_template(
            golden_X, golden_y, target_label=GOLDEN_TARGET_LABEL
        )
    except Exception as e:
        log_error(2, e)
        return 1

    log_step(3, "開始遍歷 S01–S10 執行即時推論模擬")
    report_summary = []
    for sid in range(1, 11):
        try:
            stats = assess_subject(sid, golden_template, model_v3)
        except Exception as e:
            # 單一受試者失敗不應中斷整批驗收，記錄後繼續下一位
            log_error(3, f"S{sid:02} 評估失敗：{e}")
            continue

        if stats is None:
            print(f"{YELLOW}STEP 3:S{sid:02} 無可用資料，略過{RESET}", file=sys.stderr)
            continue

        report_summary.append(stats)
        print(f"✅ S{sid:02} 測試完成。品質合格率: {stats['pass_rate']:.1f}%")

    try:
        log_step(4, "彙整驗收報表")
        print_report(report_summary)
    except Exception as e:
        log_error(4, e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_batch_assessment())
