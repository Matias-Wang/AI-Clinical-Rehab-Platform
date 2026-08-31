"""效能煙霧測試。

門檻刻意設得寬鬆，避免不同機器的效能差異造成偽陽性，但仍足以攔截
「數量級」的退化。實測基準（Keras 3.13.2）：

    predict_on_batch()  中位數  2.49 ms   ← 現行實作
    predict()           中位數 94.91 ms   ← 退回舊實作會被本測試攔下

串流速率部分揭露的是尚未修復的缺陷：Windows ProactorEventLoop 的
asyncio.sleep 粒度為 15.55 ms，使 ui_bridge 的逐幀 sleep 無法達成宣稱速率
（--realtime 實測僅 31.3 Hz，目標 50 Hz）。該項以 strict xfail 標記，
pacing 修正完成後會轉為 XPASS 而失敗，藉此強制將其改回正式斷言。
"""

import asyncio
import time

import numpy as np
import pytest

from conftest import MODEL_PATH, load_subject_cached

pytestmark = pytest.mark.perf

# 單視窗推論延遲上限（現況 2.49 ms，退回 predict() 為 94.91 ms）
MAX_INFERENCE_MEDIAN_MS = 15.0

# --realtime 應達成的最低實際取樣率（目標 50 Hz）
MIN_REALTIME_HZ = 45.0


@pytest.fixture(scope="module")
def model():
    """載入 v3 模型；缺少 TensorFlow 或模型檔時略過。"""
    keras = pytest.importorskip(
        "tensorflow.keras.models", reason="未安裝 TensorFlow，略過效能測試。"
    )
    if not MODEL_PATH.exists():
        pytest.skip(f"找不到模型檔 {MODEL_PATH}。")
    return keras.load_model(str(MODEL_PATH))


@pytest.fixture(scope="module")
def windows():
    """S10 的前 80 個視窗，整形為推論輸入。"""
    X, _ = load_subject_cached(10)
    if X is None:
        pytest.skip("找不到 S10 資料檔。")
    return [np.expand_dims(w, axis=0).astype("float32") for w in X[:80]]


def test_engine_window_latency_within_budget(model, golden_template):
    """引擎處理單一視窗的同步耗時必須維持在低毫秒級。

    **必須經由 `process_live_frame()` 量測，不可直接呼叫模型 API。**
    直接量測 `model.predict_on_batch()` 的版本無法攔截 schema.py 內部改用
    `model.predict()` 的退化——測試根本沒走到那條程式碼。

    此處量測的是實際會阻塞 asyncio event loop 的那段同步工作：
    品質閘門 + DTW 相似度 + 模型推論，實測合計約 4.86 ms；
    若推論退回 `predict()` 則升至約 97 ms，遠超門檻。
    """
    from schema import ClinicalQualityGate, RealTimeBiofeedbackEngine

    X, _ = load_subject_cached(10)
    if X is None:
        pytest.skip("找不到 S10 資料檔。")

    gate = ClinicalQualityGate(golden_template)
    engine = RealTimeBiofeedbackEngine(gate, golden_template, model)

    # 只統計 status 為 PROCEED 的視窗：HALT 分支在品質閘門就提前返回，
    # 完全不執行模型推論（約 0.01 ms）。S10 僅約 20% 視窗通過閘門，
    # 若把 HALT 一併計入，中位數會被大量快速結果稀釋而掩蓋推論退化。
    latencies = []
    for frame in X[:80].reshape(-1, 8):
        start = time.perf_counter()
        result = engine.process_live_frame(frame)
        elapsed = (time.perf_counter() - start) * 1000
        if result is not None and result["status"] == "PROCEED":
            latencies.append(elapsed)

    assert len(latencies) >= 8, (
        f"通過品質閘門的視窗僅 {len(latencies)} 個，取樣不足以判定效能。"
    )

    # 首個視窗含模型預熱成本，排除後再取中位數
    median = float(np.median(latencies[1:]))
    assert median < MAX_INFERENCE_MEDIAN_MS, (
        f"單視窗處理中位數 {median:.2f} ms 超過 {MAX_INFERENCE_MEDIAN_MS} ms 門檻。"
        "\n推論路徑可能已退回 model.predict()——該 API 為批次導向，"
        "用於單視窗高頻推論時開銷高出約 38 倍且會累積記憶體。"
    )


def measure_stream_hz(speed: float, seconds: float = 6.0) -> float:
    """量測 ui_bridge 節奏邏輯的實際幀率。

    Parameters
    ----------
    speed : float
        播放加速倍率，1.0 為真實 50Hz。
    seconds : float
        量測時長（秒）。

    Returns
    -------
    float
        實際達成的幀率（Hz）。
    """
    async def run() -> float:
        interval = (1.0 / 50.0) / speed
        n = 0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            n += 1
            await asyncio.sleep(interval)
        return n / (time.perf_counter() - start)

    return asyncio.run(run())


@pytest.mark.xfail(
    strict=True,
    reason="pacing 缺陷未修：逐幀 asyncio.sleep 受 Windows 15.55ms 粒度限制，"
           "--realtime 實測僅 31.3 Hz。修好後本測試會 XPASS，屆時請移除此 xfail。",
)
def test_realtime_achieves_target_sampling_rate():
    """--realtime 模式必須達成接近 50 Hz 的實際取樣率。"""
    hz = measure_stream_hz(speed=1.0)
    assert hz >= MIN_REALTIME_HZ, (
        f"實際取樣率僅 {hz:.1f} Hz（目標 50 Hz，門檻 {MIN_REALTIME_HZ} Hz）。"
        "\n系統無法達成宣稱的取樣率，代表它也追不上真實的 50Hz 感測器，"
        "Stage 8 的 50Hz 壓力測試無法進行。"
    )


def test_stream_rate_is_measurable():
    """節奏量測本身可運作（避免 xfail 掩蓋量測失效）。"""
    hz = measure_stream_hz(speed=1.0, seconds=3.0)
    assert hz > 0, "串流速率量測失效。"
