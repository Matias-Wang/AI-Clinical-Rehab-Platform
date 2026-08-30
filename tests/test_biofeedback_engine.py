"""RealTimeBiofeedbackEngine 的端到端狀態機測試（Stage 8 - A/D）。

使用 stub 模型，不需載入 TensorFlow，確保髒數據與斷訊情境
能被轉換為前端可理解的紅燈狀態，而非讓整條串流崩潰。
"""

import numpy as np
import pytest

from schema import ClinicalQualityGate, RealTimeBiofeedbackEngine


class StubModel:
    """固定回傳 Label 7 的假模型，用於隔離模型推論以外的邏輯。"""

    def __init__(self) -> None:
        self.call_count = 0

    def predict(self, input_data: np.ndarray, verbose: int = 0) -> np.ndarray:
        """回傳 one-hot 於索引 7 的 (1, 13) 機率陣列。"""
        self.call_count += 1
        probs = np.zeros((1, 13))
        probs[0, 7] = 1.0
        return probs


@pytest.fixture
def engine(golden_template) -> RealTimeBiofeedbackEngine:
    """以 S10 黃金範本與 stub 模型組成的即時引擎。"""
    gate = ClinicalQualityGate(golden_template)
    return RealTimeBiofeedbackEngine(gate, golden_template, StubModel())


def dynamic_frames(n: int) -> np.ndarray:
    """產生 Grav_Y 具足夠變異數、可通過品質閘門的影格序列。"""
    frames = np.zeros((n, 8))
    frames[:, 5] = np.sin(np.linspace(0, 8 * np.pi, n)) * 0.5
    return frames


class TestDirtyDataHandling:
    """髒數據不得使引擎拋出例外，必須轉為紅燈結果。"""

    def test_dirty_frame_returns_red_halt_instead_of_raising(self, engine):
        """單一髒影格應回傳 HALT/RED，並標註 DIRTY_DATA 原因。"""
        result = engine.process_live_frame(np.full(8, np.nan))

        assert result is not None
        assert result["status"] == "HALT"
        assert result["ui_color"] == "RED"
        assert result["reason"] == "DIRTY_DATA"
        assert result["predict_label"] is None

    def test_dirty_frames_do_not_flood_output(self, engine):
        """連續髒影格只在狀態轉換點輸出一次，避免洪水式推播。"""
        bad = np.full(8, np.nan)
        first = engine.process_live_frame(bad)
        followups = [engine.process_live_frame(bad) for _ in range(5)]

        assert first is not None
        assert all(r is None for r in followups)

    def test_model_is_never_called_on_dirty_data(self, engine):
        """髒數據絕不可進入模型推論。"""
        for _ in range(10):
            engine.process_live_frame(np.full(8, np.nan))
        assert engine.model.call_count == 0

    def test_disconnect_escalates_to_red_with_reason(self, engine):
        """連續髒影格達門檻應輸出 DISCONNECTED 紅燈。"""
        bad = np.full(8, np.nan)
        results = [
            engine.process_live_frame(bad)
            for _ in range(engine.disconnect_dirty_frames)
        ]
        last = results[-1]

        assert last is not None
        assert last["reason"] == "DISCONNECTED"
        assert last["ui_color"] == "RED"


class TestRecoveryAfterDirtyData:
    """D：串流在髒數據後必須能自行復原。"""

    def test_engine_recovers_and_resumes_prediction(self, engine):
        """插入髒影格後，累積滿新的乾淨視窗即應恢復正常推論。"""
        frames = dynamic_frames(400)
        for frame in frames[:100]:
            engine.process_live_frame(frame)

        engine.process_live_frame(np.full(8, np.nan))
        assert len(engine.buffer) == 0

        results = [engine.process_live_frame(f) for f in frames[100:300]]
        proceeded = [r for r in results if r and r["status"] == "PROCEED"]

        assert proceeded, "髒數據後引擎未能恢復輸出，復原機制失效。"
        assert proceeded[0]["predict_label"] == 7

    def test_no_window_spans_the_dirty_frame(self, engine):
        """復原後輸出的視窗不得含任何無效值。"""
        frames = dynamic_frames(400)
        for frame in frames[:120]:
            engine.process_live_frame(frame)
        engine.process_live_frame(np.full(8, np.inf))

        for frame in frames[120:]:
            engine.process_live_frame(frame)

        assert np.all(np.isfinite(np.array(engine.buffer)))


class TestCleanPathUnchanged:
    """迴歸保護：乾淨路徑的輸出格式與行為不得改變。"""

    def test_result_keys_are_stable(self, engine):
        """PROCEED 結果必須包含既有消費端依賴的所有欄位。"""
        results = [r for r in map(engine.process_live_frame, dynamic_frames(300)) if r]
        proceed = next(r for r in results if r["status"] == "PROCEED")

        assert set(proceed) == {
            "status", "ui_color", "msg", "score",
            "similarity", "predict_label", "reason",
        }
        assert proceed["ui_color"] in {"GREEN", "YELLOW"}
        assert 0 <= proceed["score"] <= 100

    def test_static_input_still_halts_with_low_quality(self, engine):
        """靜態輸入仍應以 LOW_QUALITY 原因被攔截。"""
        results = [r for r in map(engine.process_live_frame, np.zeros((300, 8))) if r]

        assert results
        assert all(r["status"] == "HALT" for r in results)
        assert all(r["reason"] == "LOW_QUALITY" for r in results)
