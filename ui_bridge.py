"""ui_bridge.py — 即時資料橋接層 (Stage 7)
# python ui_bridge.py --subject 10 --port 8765 --speed 20
透過 WebSocket 將 RealTimeBiofeedbackEngine 的即時推論結果
（ui_color、final_score 等）即時推播給前端展示頁面
（frontend/demo.html）。
"""

from console import enable_utf8_output

# Stage 8：必須早於任何輸出，避免伺服器以 subprocess 啟動（stdout 為管道）
# 時，燈號訊息中的 emoji 觸發 cp950 UnicodeEncodeError 而中斷服務。
enable_utf8_output()

import argparse
import asyncio
import json

import numpy as np
from tensorflow.keras.models import load_model
from websockets.asyncio.server import ServerConnection, broadcast, serve

from schema import (
    ACTIVITY_LABELS,
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

# 串流落後超過此秒數即重設時間軸，不再嘗試追趕（見 stream_subject()）
MAX_CATCHUP_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    """解析命令列參數。

    Returns
    -------
    argparse.Namespace
        包含 subject、port、speed、realtime 四個欄位。
    """
    parser = argparse.ArgumentParser(description="AI 臨床復健平台 - 即時資料橋接伺服器")
    parser.add_argument("--subject", type=int, default=10, help="模擬串流的受試者編號 (預設 10)")
    parser.add_argument(
        "--port", type=int, default=8765, help="WebSocket 監聽埠 (預設 8765)"
    )
    parser.add_argument(
        "--speed", type=float, default=20.0, help="播放加速倍率，1.0 為真實 50Hz (預設 20.0)"
    )
    parser.add_argument(
        "--realtime", action="store_true", help="以真實 50Hz 播放 (等同 --speed 1.0)"
    )
    return parser.parse_args()


def build_broadcast_payload(
    result: dict, subject_id: int, frame_index: int
) -> str:
    """將 process_live_frame() 的結果轉為前端可用的 JSON 字串。

    Parameters
    ----------
    result : dict
        RealTimeBiofeedbackEngine.process_live_frame() 的回傳值。
    subject_id : int
        目前模擬串流的受試者編號。
    frame_index : int
        目前視窗在串流中的序號（每次重播歸零）。

    Returns
    -------
    str
        可直接透過 WebSocket 傳送的 JSON 字串。
    """
    predict_label = result["predict_label"]
    payload = {
        "status": result["status"],
        "ui_color": result["ui_color"],
        "msg": result["msg"],
        "score": float(result["score"]),
        "similarity": float(result["similarity"]),
        "predict_label": int(predict_label) if predict_label is not None else None,
        "predict_label_name": ACTIVITY_LABELS.get(int(predict_label))
        if predict_label is not None
        else None,
        "reason": result.get("reason", "OK"),
        "subject_id": subject_id,
        "frame_index": frame_index,
    }
    return json.dumps(payload, ensure_ascii=False)


async def stream_subject(
    engine: RealTimeBiofeedbackEngine,
    windows: np.ndarray,
    subject_id: int,
    speed: float,
    clients: set[ServerConnection],
) -> None:
    """依真實時間節奏逐視窗評估並 broadcast 結果給所有 client。

    重播跑到結尾後會自動從頭開始，讓展示頁面可以持續看到燈號變化。

    Stage 8 B1：改為逐視窗推播，取代原本「攤平重疊視窗後逐幀 push_data」
    的作法。原作法把已 50% 重疊的視窗攤平當成連續訊號，使引擎重新切窗後
    有 50% 的視窗橫跨兩個原始視窗——第 8 維 FFT 能量因此出現兩個不同值，
    而該特徵在訓練資料中永遠是整個視窗的單一常數，構成 train/serving skew。
    FFT 為視窗級特徵且須以原始尺度的 Magnitude 計算（見 Fix 3），在已標準化
    的逐幀串流中無法重建，故逐視窗評估是唯一能餵入正確輸入形態的作法。

    時間節奏維持真實：相鄰視窗相距 stride 幀，故每個視窗間隔
    `stride / 50 / speed` 秒（--realtime 下為 1.28 秒）。

    Parameters
    ----------
    engine : RealTimeBiofeedbackEngine
        已初始化的即時推論引擎。
    windows : numpy.ndarray
        預處理後的視窗集合，shape 為 (n_windows, 128, 8)。
    subject_id : int
        受試者編號，僅用於標記推播訊息。
    speed : float
        播放加速倍率，1.0 代表真實 50Hz。
    clients : set[ServerConnection]
        目前已連線的 WebSocket client 集合。
    """
    loop = asyncio.get_running_loop()
    window_interval = (engine.stride / 50.0) / speed
    frame_index = 0
    windows_elapsed = 0
    timeline_start = loop.time()

    while True:
        for window in windows:
            try:
                result = engine.evaluate_window(window)
                payload = build_broadcast_payload(result, subject_id, frame_index)
                if clients:
                    broadcast(clients, payload)
                frame_index += 1
            except Exception as e:
                # 單一視窗評估失敗不應中斷整條串流，記錄後繼續下一個視窗
                print(f"{RED}STEP 4 ERROR:{e}{RESET}")

            # Stage 8 (A1)：以絕對時間軸排程，取代逐次 sleep(window_interval)。
            # Windows 的 ProactorEventLoop 計時器粒度為 15.55 ms，任何小於此值
            # 的 sleep 都會被拉長到 15.55 ms，逐次 sleep 會使誤差持續累積：
            # --realtime 原本只跑到 31.3 Hz（目標 50 Hz），系統比即時還慢。
            #
            # 改以「每個視窗的目標時刻」計算延遲：超前才 sleep，落後則立即
            # 處理下一個視窗進行追趕，粒度誤差因此攤平為抖動而非累積延遲。
            windows_elapsed += 1
            delay = timeline_start + windows_elapsed * window_interval - loop.time()

            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # 落後時不 sleep，但仍讓出控制權，避免追趕期間餓死其他任務
                await asyncio.sleep(0)

                if delay < -MAX_CATCHUP_SECONDS:
                    # 落後過多（系統暫停、行程被凍結等）時重設時間軸，
                    # 避免以最高速率無止境追趕一段永遠補不回來的落差。
                    timeline_start = (
                        loop.time() - windows_elapsed * window_interval
                    )


async def handle_client(
    connection: ServerConnection, clients: set[ServerConnection]
) -> None:
    """處理單一 WebSocket client 的連線生命週期。

    Parameters
    ----------
    connection : ServerConnection
        新建立的 client 連線。
    clients : set[ServerConnection]
        所有已連線 client 的共用集合，用於 broadcast。
    """
    clients.add(connection)
    print(f"{CYAN}STEP 5:新 Client 連線，目前連線數 {len(clients)}{RESET}")
    try:
        async for _ in connection:
            pass  # demo client 不會送任何訊息，僅需維持連線並偵測斷線
    except Exception as e:
        print(f"{RED}STEP 5 ERROR:{e}{RESET}")
    finally:
        clients.discard(connection)
        print(f"{CYAN}STEP 5:Client 斷線，目前連線數 {len(clients)}{RESET}")


async def main() -> None:
    """啟動即時資料橋接伺服器的主流程。"""
    args = parse_args()
    speed = 1.0 if args.realtime else args.speed

    try:
        print(f"{GREEN}STEP 1:載入模型 {MODEL_PATH}{RESET}")
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"{RED}STEP 1 ERROR:{e}{RESET}")
        return

    try:
        print(f"{GREEN}STEP 2:準備黃金範本與受試者 S{args.subject:02} 模擬串流{RESET}")
        golden_X, golden_y = load_and_preprocess_subject(10, DATA_FOLDER)
        golden_template = extract_golden_template(golden_X, golden_y, target_label=7)
        X_subject, _ = load_and_preprocess_subject(args.subject, DATA_FOLDER)
        if X_subject is None:
            raise ValueError(f"找不到受試者 S{args.subject:02} 的資料")
        # Stage 8 B1：直接使用預處理視窗，不再攤平為逐幀串流
    except Exception as e:
        print(f"{RED}STEP 2 ERROR:{e}{RESET}")
        return

    gate = ClinicalQualityGate(golden_template)
    engine = RealTimeBiofeedbackEngine(gate, golden_template, model)
    clients: set[ServerConnection] = set()

    try:
        print(f"{GREEN}STEP 3:啟動 WebSocket 伺服器於 ws://localhost:{args.port}{RESET}")
        async with serve(
            lambda connection: handle_client(connection, clients),
            "localhost",
            args.port,
        ):
            print(
                f"{GREEN}STEP 4:開始模擬受試者 S{args.subject:02} "
                f"即時串流 (speed={speed}x){RESET}"
            )
            await stream_subject(engine, X_subject, args.subject, speed, clients)
    except Exception as e:
        print(f"{RED}STEP 3 ERROR:{e}{RESET}")


if __name__ == "__main__":
    # Stage 8 (A10)：Ctrl+C 原本會讓 asyncio.run() 直接拋出 KeyboardInterrupt
    # 並印出整段 traceback。臨床展示情境下這既不美觀也讓人誤以為系統崩潰，
    # 故改為輸出明確的關閉訊息後正常結束。
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{CYAN}STEP 6:收到中斷訊號，伺服器已關閉{RESET}")
