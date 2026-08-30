"""dtw_distance() 的數學性質測試（Stage 6 回歸）。

ROADMAP 主張「DTW 距離恆 ≤ 歐幾里德距離，因此評分公式與門檻無需重新校準」。
此模組把該主張釘死為可執行的測試。
"""

import numpy as np
import pytest

from schema import dtw_distance


class TestDtwProperties:
    """DTW 的基本數學性質。"""

    def test_identical_sequences_have_zero_distance(self):
        """完全相同的序列距離為 0。"""
        seq = np.sin(np.linspace(0, 4 * np.pi, 128))
        assert dtw_distance(seq, seq) == pytest.approx(0.0)

    def test_distance_is_symmetric(self):
        """DTW 距離應對稱。"""
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=128), rng.normal(size=128)
        assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))

    def test_distance_is_non_negative(self):
        """距離不得為負。"""
        rng = np.random.default_rng(8)
        for _ in range(5):
            a, b = rng.normal(size=128), rng.normal(size=128)
            assert dtw_distance(a, b) >= 0.0

    def test_dtw_never_exceeds_euclidean(self):
        """核心主張：對角線恆為合法路徑，故 DTW ≤ 歐幾里德距離。

        這是 Stage 6 未重新校準評分係數與 GREEN/YELLOW 門檻的唯一依據。
        """
        rng = np.random.default_rng(11)
        for _ in range(20):
            a, b = rng.normal(size=128), rng.normal(size=128)
            euclidean = float(np.linalg.norm(a - b))
            assert dtw_distance(a, b) <= euclidean + 1e-9

    def test_dtw_tolerates_phase_shift_better_than_euclidean(self):
        """相位平移下 DTW 應明顯優於歐幾里德距離（Stage 6 的動機）。"""
        t = np.linspace(0, 4 * np.pi, 128)
        reference = np.sin(t)
        shifted = np.sin(t + 0.4)  # 節奏落後，非動作品質變差

        euclidean = float(np.linalg.norm(reference - shifted))
        assert dtw_distance(reference, shifted) < euclidean

    def test_radius_zero_equals_euclidean(self):
        """radius=0 時搜尋空間退化為對角線，應等同歐幾里德距離。"""
        rng = np.random.default_rng(13)
        a, b = rng.normal(size=64), rng.normal(size=64)
        assert dtw_distance(a, b, radius=0) == pytest.approx(
            float(np.linalg.norm(a - b))
        )
