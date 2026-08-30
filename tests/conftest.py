"""pytest 共用設定與 fixture。

讓測試可直接 `import schema`（將專案根目錄加入 sys.path），
並提供跨測試共用的受試者資料快取，避免每個測試重複跑預處理管線。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_FOLDER = str(PROJECT_ROOT / "data")
MODEL_PATH = PROJECT_ROOT / "models" / "clinical_rehab_model_v3.keras"

# ROADMAP「Fix 3」記錄的基準準確率，來源為訓練用對照 notebook
# development_history/20260404_Project_Rehab_Optimization.ipynb
NOTEBOOK_BASELINE_ACCURACY = {2: 0.8430, 7: 0.9045, 9: 0.9065, 10: 0.9029}

_subject_cache: dict[int, tuple] = {}


def load_subject_cached(subject_id: int) -> tuple:
    """載入並快取指定受試者的預處理結果。

    Parameters
    ----------
    subject_id : int
        受試者編號 (1-10)。

    Returns
    -------
    tuple
        (X, y)，X shape 為 (n_windows, 128, 8)。
    """
    if subject_id not in _subject_cache:
        from schema import load_and_preprocess_subject

        _subject_cache[subject_id] = load_and_preprocess_subject(
            subject_id, DATA_FOLDER
        )
    return _subject_cache[subject_id]


@pytest.fixture(scope="session")
def subject_s10() -> tuple:
    """S10（黃金標準受試者）的預處理資料。"""
    X, y = load_subject_cached(10)
    if X is None:
        pytest.skip("找不到 S10 資料檔，略過需要真實資料的測試。")
    return X, y


@pytest.fixture(scope="session")
def golden_template(subject_s10) -> np.ndarray:
    """由 S10 Label 7 視窗擷取的黃金範本 (128, 8)。"""
    from schema import extract_golden_template

    X, y = subject_s10
    return extract_golden_template(X, y, target_label=7)


@pytest.fixture
def clean_frame(golden_template) -> np.ndarray:
    """一筆通過所有完整性檢查的乾淨影格 (8,)。"""
    return golden_template[0].copy()
