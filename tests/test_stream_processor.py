"""RealTimeStreamProcessor 的髒數據攔截與緩衝區復原測試（Stage 8 - A/D）。

對應 ROADMAP Stage 8 的「感測器斷訊、髒數據、硬體噪音」防護需求。
"""

import numpy as np
import pytest

from schema import (
    STREAM_GUARD,
    DirtyFrameError,
    RealTimeStreamProcessor,
    SensorDisconnectedError,
)


@pytest.fixture
def processor() -> RealTimeStreamProcessor:
    """預設參數（window=128、stride=64）的串流處理器。"""
    return RealTimeStreamProcessor()


def clean_frames(n: int) -> np.ndarray:
    """產生 n 筆合法的 8D 影格。"""
    rng = np.random.default_rng(42)
    return rng.normal(scale=0.5, size=(n, 8))


class TestWindowEmission:
    """乾淨資料下的視窗輸出節奏不得因 Stage 8 修改而改變。"""

    def test_no_window_before_buffer_is_full(self, processor):
        """緩衝區未滿 128 筆前不得輸出視窗。"""
        for frame in clean_frames(127):
            ready, window = processor.push_data(frame)
            assert ready is False and window is None

    def test_first_window_emitted_at_128_frames(self, processor):
        """第 128 筆影格應觸發第一個視窗。"""
        frames = clean_frames(128)
        for frame in frames[:-1]:
            processor.push_data(frame)

        ready, window = processor.push_data(frames[-1])
        assert ready is True
        assert window.shape == (128, 8)

    def test_subsequent_windows_emitted_every_stride(self, processor):
        """緩衝區填滿後，每 64 筆（stride）輸出一個視窗。"""
        emitted = [
            i for i, frame in enumerate(clean_frames(320))
            if processor.push_data(frame)[0]
        ]
        assert emitted == [127, 191, 255, 319]


class TestFrameValidation:
    """A：髒影格必須在進入緩衝區前就被攔截。"""

    @pytest.mark.parametrize(
        "bad_frame, label",
        [
            (np.full(8, np.nan), "全 NaN"),
            (np.array([0.1, np.nan, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), "單點 NaN"),
            (np.array([0.1, np.inf, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), "Inf"),
            (np.zeros(7), "維度不足 (7D)"),
            (np.zeros(9), "維度過多 (9D)"),
            (np.array([1e6, 0, 0, 0, 0, 0, 0, 0]), "數值爆表"),
        ],
    )
    def test_dirty_frames_are_rejected(self, processor, bad_frame, label):
        """各類髒影格都應拋出 DirtyFrameError。"""
        with pytest.raises(DirtyFrameError):
            processor.push_data(bad_frame)

    def test_rejected_frame_never_enters_buffer(self, processor):
        """被拒絕的影格不得留在緩衝區內。"""
        with pytest.raises(DirtyFrameError):
            processor.push_data(np.full(8, np.nan))

        assert len(processor.buffer) == 0
        assert not any(
            not np.all(np.isfinite(row)) for row in processor.buffer
        )

    def test_non_numeric_frame_is_rejected(self, processor):
        """無法轉為數值的輸入（例如字串）也應被攔截。"""
        with pytest.raises(DirtyFrameError):
            processor.push_data(["a", "b", "c", "d", "e", "f", "g", "h"])

    def test_clean_frame_within_sanity_limit_passes(self, processor):
        """接近但未超過門檻的數值應視為合法。"""
        frame = np.full(8, STREAM_GUARD["SANITY_ABS_LIMIT"] - 0.1)
        ready, _ = processor.push_data(frame)
        assert ready is False
        assert len(processor.buffer) == 1


class TestBufferRecovery:
    """D：髒影格必須觸發緩衝區清空，避免污染後續視窗。"""

    def test_dirty_frame_flushes_accumulated_buffer(self, processor):
        """已累積的乾淨資料在遇到髒影格時應整段丟棄。"""
        for frame in clean_frames(100):
            processor.push_data(frame)
        assert len(processor.buffer) == 100

        with pytest.raises(DirtyFrameError):
            processor.push_data(np.full(8, np.nan))

        assert len(processor.buffer) == 0, "髒影格後緩衝區未清空，後續視窗會被污染。"
        assert processor.new_data_counter == 0
        assert processor.buffer_resets == 1

    def test_recovery_requires_a_full_clean_window(self, processor):
        """復原後必須重新累積滿 128 筆乾淨資料才會再輸出視窗。"""
        for frame in clean_frames(127):
            processor.push_data(frame)

        with pytest.raises(DirtyFrameError):
            processor.push_data(np.full(8, np.nan))

        # 清空後再推 127 筆仍不應輸出
        frames = clean_frames(128)
        for frame in frames[:-1]:
            ready, _ = processor.push_data(frame)
            assert ready is False

        ready, window = processor.push_data(frames[-1])
        assert ready is True
        assert np.all(np.isfinite(window)), "復原後的視窗仍含無效值。"

    def test_reset_buffer_is_idempotent(self, processor):
        """重複呼叫 reset_buffer() 不應出錯。"""
        processor.reset_buffer()
        processor.reset_buffer()
        assert len(processor.buffer) == 0
        assert processor.buffer_resets == 2


class TestDisconnectEscalation:
    """A：連續髒影格應升級為感測器斷訊。"""

    def test_disconnect_raised_after_threshold(self, processor):
        """連續髒影格達門檻時改拋 SensorDisconnectedError。"""
        threshold = STREAM_GUARD["DISCONNECT_DIRTY_FRAMES"]
        bad = np.full(8, np.nan)

        for _ in range(threshold - 1):
            with pytest.raises(DirtyFrameError):
                processor.push_data(bad)

        with pytest.raises(SensorDisconnectedError):
            processor.push_data(bad)

    def test_clean_frame_resets_consecutive_counter(self, processor):
        """中間出現乾淨影格即應重置連續計數，不誤判為斷訊。"""
        bad = np.full(8, np.nan)
        for _ in range(5):
            with pytest.raises(DirtyFrameError):
                processor.push_data(bad)
        assert processor.consecutive_dirty == 5

        processor.push_data(clean_frames(1)[0])
        assert processor.consecutive_dirty == 0


class TestHealthStats:
    """長時間運行的健康度統計正確性。"""

    def test_stats_track_dirty_ratio_and_resets(self, processor):
        """統計數字應如實反映髒數據比例與重置次數。"""
        for frame in clean_frames(10):
            processor.push_data(frame)
        for _ in range(3):
            with pytest.raises(DirtyFrameError):
                processor.push_data(np.full(8, np.nan))

        stats = processor.get_health_stats()
        assert stats["total_frames"] == 13
        assert stats["dirty_frames"] == 3
        assert stats["dirty_ratio"] == pytest.approx(3 / 13)
        assert stats["buffer_resets"] == 3

    def test_buffer_never_exceeds_window_size(self, processor):
        """緩衝區為 bounded deque，長時間串流不得無限成長。"""
        for frame in clean_frames(2000):
            processor.push_data(frame)
            assert len(processor.buffer) <= processor.window_size
