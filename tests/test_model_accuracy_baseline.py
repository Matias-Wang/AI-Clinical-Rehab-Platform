"""Fix 3 的最終防線：模型準確率基準測試。

ROADMAP 明確記錄，Z-score 與 FFT 的順序顛倒時「整體準確率反而崩到 ~5-48%，
看起來像修好了實際上更糟，必須靠對照 notebook 的基準準確率才抓出來」。
特徵層的測試無法涵蓋這種情況，因此保留一組端到端準確率斷言。

本模組需要 TensorFlow 與已訓練模型，執行較慢；標記為 slow，
可用 `pytest -m "not slow"` 略過。
"""

import numpy as np
import pytest

from conftest import MODEL_PATH, NOTEBOOK_BASELINE_ACCURACY, load_subject_cached

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model_v3():
    """載入 v3.1_multi 模型；缺少 TensorFlow 或模型檔時略過。"""
    keras = pytest.importorskip(
        "tensorflow.keras.models", reason="未安裝 TensorFlow，略過準確率基準測試。"
    )
    if not MODEL_PATH.exists():
        pytest.skip(f"找不到模型檔 {MODEL_PATH}，略過準確率基準測試。")
    return keras.load_model(str(MODEL_PATH))


@pytest.mark.parametrize("subject_id", sorted(NOTEBOOK_BASELINE_ACCURACY))
def test_accuracy_matches_notebook_baseline(model_v3, subject_id):
    """各受試者準確率必須吻合訓練 notebook 的記錄值。

    容差設為 ±1%：順序顛倒等 train/serving skew 會造成數十個百分點的落差，
    遠超此容差，因此能被穩定攔截。
    """
    X, y = load_subject_cached(subject_id)
    if X is None:
        pytest.skip(f"找不到 S{subject_id:02} 的資料檔。")

    predictions = np.argmax(model_v3.predict(X, verbose=0), axis=1)
    accuracy = float(np.mean(predictions == y))
    expected = NOTEBOOK_BASELINE_ACCURACY[subject_id]

    assert accuracy == pytest.approx(expected, abs=0.01), (
        f"S{subject_id:02} 準確率 {accuracy:.4f} 偏離 notebook 基準 {expected:.4f}，"
        "預處理管線可能再次出現 train/serving skew（參見 ROADMAP Fix 3）。"
    )


def test_all_twelve_labels_are_predictable(model_v3):
    """Fix 3 的核心症狀回歸：不得再有整個類別從未被預測。

    修復前，Label 1、5、6、7 在 12 類中完全消失。
    """
    predicted_labels = set()
    for subject_id in sorted(NOTEBOOK_BASELINE_ACCURACY):
        X, _ = load_subject_cached(subject_id)
        if X is None:
            continue
        predicted_labels.update(
            np.argmax(model_v3.predict(X, verbose=0), axis=1).tolist()
        )

    missing = set(range(1, 13)) - predicted_labels
    assert not missing, (
        f"標籤 {sorted(missing)} 從未被預測，症狀與 Fix 3 修復前一致。"
    )


def test_label_7_is_recognised(model_v3):
    """Label 7（手臂前舉）為黃金範本標籤，其召回率不得為 0。

    修復前 main.py 報告的「AI 辨識 L7 次數」恆為 0，
    真實 L7 視窗有 98% 以上被誤判為 Label 10。
    """
    X, y = load_subject_cached(10)
    if X is None:
        pytest.skip("找不到 S10 的資料檔。")

    predictions = np.argmax(model_v3.predict(X, verbose=0), axis=1)
    l7_mask = y == 7
    assert l7_mask.any(), "S10 應包含 Label 7 視窗。"

    recall = float(np.mean(predictions[l7_mask] == 7))
    assert recall > 0.5, (
        f"Label 7 召回率僅 {recall:.2%}，Fix 3 的 train/serving skew 可能已回退。"
    )
