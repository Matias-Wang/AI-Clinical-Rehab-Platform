"""預處理管線的致命修正回歸測試（Fix 2 / Fix 3）。

ROADMAP.md 記錄了兩個曾讓整套系統邏輯空轉的 bug：

- Fix 2：`gravity_values / 10.0` 遺失，使變異數放大 100 倍、品質閘門完全失效。
- Fix 3：Acc/Mag 受試者級 Z-score 與 FFT `log1p(x)/5.0` 縮放遺失，導致 12 類中
  有 4 類從未被預測；且兩者順序顛倒時準確率會崩到 5-48%，卻「看起來像修好了」。

這些修正只存在於文件與紀錄裡，缺乏測試釘死；本模組即為此而建。
"""

import numpy as np
import pytest

from schema import extract_window_fft_energy, load_and_preprocess_subject
from conftest import DATA_FOLDER


class TestFix2GravityScaling:
    """Fix 2：重力分量必須執行全局縮放 ÷10.0。"""

    def test_gravity_vector_norm_is_scaled_to_unit_range(self, subject_s10):
        """重力向量長度應落在 1.0 附近（9.8 m/s² ÷ 10）而非 9.8 附近。"""
        X, _ = subject_s10
        gravity = X[:, :, 4:7].reshape(-1, 3)
        mean_norm = float(np.linalg.norm(gravity, axis=1).mean())

        # 若 ÷10.0 被移除，此值會回到 ~9.3，測試立即失敗
        assert 0.5 < mean_norm < 1.5, (
            f"重力向量長度 {mean_norm:.4f} 不在 ÷10.0 後的預期範圍，"
            "Fix 2 的全局縮放可能已遺失。"
        )

    def test_grav_y_variance_matches_quality_gate_thresholds(self, subject_s10):
        """Grav_Y 變異數的量級必須與品質閘門門檻 (0.0005) 相容。"""
        X, _ = subject_s10
        var_y = np.var(X[:, :, 5], axis=1)

        # 縮放正確時，S10 應同時存在「低於門檻的靜態視窗」與「高於門檻的動態視窗」，
        # 品質閘門才有鑑別力；若縮放遺失，變異數放大 100 倍會使幾乎全數通過。
        assert (var_y < 0.0005).any(), "S10 應存在被攔截的靜態視窗。"
        assert (var_y > 0.0005).any(), "S10 應存在通過閘門的動態視窗。"


class TestFix3ZScoreAndFFT:
    """Fix 3：Acc/Mag 受試者級 Z-score、FFT 縮放，以及兩者的先後順序。"""

    def test_acc_and_magnitude_are_subject_level_zscored(self, subject_s10):
        """前 4 欄（Acc_X/Y/Z、Magnitude）應在受試者層級標準化為 mean≈0、std≈1。"""
        X, _ = subject_s10
        acc_mag = X[:, :, :4].reshape(-1, 4)

        np.testing.assert_allclose(acc_mag.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(acc_mag.std(axis=0), 1.0, atol=1e-6)

    def test_gravity_and_fft_columns_are_not_zscored(self, subject_s10):
        """Z-score 只能作用於前 4 欄，重力與 FFT 欄位必須維持原本縮放。"""
        X, _ = subject_s10
        untouched = X[:, :, 4:].reshape(-1, 4)

        # 重力欄位平均值明顯偏離 0（存在固定的重力方向偏置），
        # FFT 欄位恆為正值，兩者都不可能是標準化後的結果
        assert np.abs(untouched.mean(axis=0)).max() > 0.05, (
            "第 4-7 欄看起來被標準化了，Z-score 的作用範圍可能被誤擴大。"
        )
        assert (X[:, :, 7] > 0).all(), "FFT 能量經 log1p 後應恆為正值。"

    def test_fft_is_computed_before_zscore_not_after(self, subject_s10):
        """順序陷阱守門：FFT 能量必須以「原始尺度」Magnitude 計算。

        這是 Fix 3 最難察覺的部分。若 Z-score 被誤排到 FFT 廣播之前，
        儲存於第 7 欄的能量值會「恰好等於」用最終輸出（已標準化）的
        Magnitude 重算出來的能量。此處直接斷言兩者必須不同。
        """
        X, _ = subject_s10
        stored_energy = X[:, 0, 7]
        recomputed_from_scaled = np.array(
            [extract_window_fft_energy(window) for window in X]
        )

        assert not np.allclose(stored_energy, recomputed_from_scaled), (
            "第 7 欄的 FFT 能量等同於用已標準化的 Magnitude 重算的結果，"
            "代表 Z-score 被錯誤地排在 FFT 廣播之前（ROADMAP Fix 3 的順序陷阱）。"
        )

    def test_fft_energy_stays_in_raw_scale_regime(self, subject_s10):
        """以原始尺度計算的能量平均值明顯高於用標準化尺度計算的結果。"""
        X, _ = subject_s10
        stored_mean = float(X[:, 0, 7].mean())
        scaled_mean = float(
            np.mean([extract_window_fft_energy(window) for window in X])
        )

        assert stored_mean > scaled_mean, (
            f"儲存的 FFT 能量均值 {stored_mean:.4f} 未高於標準化尺度重算值 "
            f"{scaled_mean:.4f}，FFT 與 Z-score 的順序可疑。"
        )

    def test_extract_window_fft_energy_applies_log1p_scaling(self):
        """extract_window_fft_energy() 回傳值必須已含 log1p(x)/5.0 縮放。"""
        rng = np.random.default_rng(0)
        window = rng.normal(loc=9.8, scale=1.0, size=(128, 8))

        sig = window[:, 3]
        fft_vals = np.fft.rfft(sig)
        raw_energy = np.sum(np.abs(fft_vals[1:]) ** 2) / len(sig)
        expected = np.log1p(raw_energy) / 5.0

        assert extract_window_fft_energy(window) == pytest.approx(expected)


class TestPipelineContract:
    """預處理管線的對外契約（維度與標籤）。"""

    def test_output_shape_is_8d(self, subject_s10):
        """Fix 1 回歸：輸出必須為 8 維特徵，不得退回 7 維。"""
        X, y = subject_s10
        assert X.ndim == 3
        assert X.shape[1] == 128
        assert X.shape[2] == 8, "特徵維度不是 8，Fix 1 的維度對齊可能已回退。"
        assert len(X) == len(y)

    def test_all_windows_are_finite(self, subject_s10):
        """預處理輸出不得含 NaN/Inf，否則會直接穿透下游品質閘門。"""
        X, _ = subject_s10
        assert np.all(np.isfinite(X))

    def test_missing_subject_returns_none_pair(self):
        """受試者檔案不存在時應回傳 (None, None) 而非拋出例外。"""
        X, y = load_and_preprocess_subject(999, DATA_FOLDER)
        assert X is None and y is None
