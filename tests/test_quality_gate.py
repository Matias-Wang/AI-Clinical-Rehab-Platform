"""ClinicalQualityGate 的臨床安全行為測試（Stage 8 - A）。

重點在於 fail-safe：任何無法判定品質的情況都必須攔截，
絕不能讓髒數據穿透閘門進入模型推論。
"""

import numpy as np
import pytest

from schema import ClinicalQualityGate


@pytest.fixture
def gate(golden_template) -> ClinicalQualityGate:
    """以 S10 黃金範本初始化的品質閘門。"""
    return ClinicalQualityGate(golden_template)


def make_window(grav_y: np.ndarray) -> np.ndarray:
    """建立一個只控制 Grav_Y（第 5 欄）的 (128, 8) 測試視窗。

    Parameters
    ----------
    grav_y : numpy.ndarray
        長度 128 的 Grav_Y 序列。

    Returns
    -------
    numpy.ndarray
        shape (128, 8) 的視窗。
    """
    window = np.zeros((128, 8), dtype=np.float64)
    window[:, 5] = grav_y
    return window


class TestNaNGuard:
    """Stage 8 修正：NaN/Inf 必須被攔截，而非被判定為品質良好。"""

    def test_nan_window_is_rejected(self, gate):
        """含 NaN 的視窗必須攔截。

        修正前 var_y 為 NaN，而 `NaN < MIN_SAFE_LIMIT` 恆為 False，
        會讓髒數據被判定為「品質良好」並送入模型。
        """
        window = make_window(np.full(128, np.nan))
        is_valid, score, msg = gate.get_quality_report(window)

        assert is_valid is False
        assert score == 0.0
        assert "NaN" in msg

    def test_partially_nan_window_is_rejected(self, gate):
        """只要視窗內有任一 NaN 即應攔截。"""
        grav_y = np.linspace(-1.0, 1.0, 128)
        grav_y[64] = np.nan
        is_valid, _, _ = gate.get_quality_report(make_window(grav_y))
        assert is_valid is False

    def test_inf_window_is_rejected(self, gate):
        """含 Inf 的視窗必須攔截（變異數會變成 NaN 或 Inf）。"""
        grav_y = np.zeros(128)
        grav_y[10] = np.inf
        is_valid, _, _ = gate.get_quality_report(make_window(grav_y))
        assert is_valid is False

    def test_batch_window_with_nan_is_rejected(self, gate):
        """3 維批次輸入同樣需受 NaN 防護。"""
        batch = np.zeros((4, 128, 8))
        batch[:, :, 5] = np.linspace(-1.0, 1.0, 128)
        batch[2, 30, 5] = np.nan
        is_valid, _, _ = gate.get_quality_report(batch)
        assert is_valid is False


class TestThresholdBehaviour:
    """驗收基準：門檻邏輯本身不得因 Stage 8 的修改而變動。"""

    def test_static_window_is_halted(self, gate):
        """靜態（變異數趨近 0）視窗應被攔截並給出偏低分數。"""
        is_valid, score, _ = gate.get_quality_report(make_window(np.zeros(128)))
        assert is_valid is False
        assert score == pytest.approx(0.0)

    def test_dynamic_window_passes(self, gate):
        """變異數明顯高於門檻的視窗應通過。"""
        grav_y = np.sin(np.linspace(0, 4 * np.pi, 128)) * 0.5
        is_valid, score, _ = gate.get_quality_report(make_window(grav_y))
        assert is_valid is True
        assert score == pytest.approx(100.0)

    def test_score_is_capped_at_100(self, gate):
        """分數上限必須為 100。"""
        grav_y = np.sin(np.linspace(0, 8 * np.pi, 128)) * 10.0
        _, score, _ = gate.get_quality_report(make_window(grav_y))
        assert score <= 100.0

    def test_threshold_constants_unchanged(self, gate):
        """門檻常數為 Stage 5 驗收數據的基礎，變動即需重新驗收。"""
        assert gate.MIN_SAFE_LIMIT == 0.0005
        assert gate.GOLDEN_VAR_LIMIT == 0.001595

    def test_real_s10_pass_rate_matches_acceptance_baseline(self, subject_s10, gate):
        """S10 的視窗通過率應維持在 ROADMAP 記錄的量級（約 20%）。"""
        X, _ = subject_s10
        passed = sum(
            1 for w in X if gate.get_quality_report(w)[0]
        )
        pass_rate = passed / len(X) * 100

        assert 10.0 < pass_rate < 35.0, (
            f"S10 品質合格率 {pass_rate:.1f}% 偏離驗收基準（約 20.6%）。"
        )
