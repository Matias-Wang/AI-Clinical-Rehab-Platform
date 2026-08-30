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


def parse_args() -> argparse.Namespace:
    """解析命令列參數。

    Returns
    -------
    argparse.Namespace
        包含 subject、port、speed、realtime 四個欄位。
    """
    parser = argparse.ArgumentParser(description="AI 臨床復健平台 - 即時資料橋接伺服器")
    parser.add_argument("--subject", type=int, default=10, help="模擬串流的受試者編號 (預設 10)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket 監聽埠 (預設 8765)")
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
    flat_stream: np.ndarray,
    subject_id: int,
    speed: float,
    clients: set[ServerConnection],
) -> None:
    """持續模擬受試者感測器串流，並將每個視窗結果 broadcast 給所有 client。

    串流跑到結尾後會自動從頭重播，讓展示頁面可以持續看到燈號變化。

    Parameters
    ----------
    engine : RealTimeBiofeedbackEngine
        已初始化的即時推論引擎。
    flat_stream : numpy.ndarray
        攤平後的逐幀感測器資料，shape 為 (n_frames, 8)。
    subject_id : int
        受試者編號，僅用於標記推播訊息。
    speed : float
        播放加速倍率，1.0 代表真實 50Hz。
    clients : set[ServerConnection]
        目前已連線的 WebSocket client 集合。
    """
    frame_interval = (1.0 / 50.0) / speed
    frame_index = 0

    while True:
        for frame in flat_stream:
            try:
                result = engine.process_live_frame(frame)
                if result is not None:
                    payload = build_broadcast_payload(result, subject_id, frame_index)
                    if clients:
                        broadcast(clients, payload)
                    frame_index += 1
            except Exception as e:
                # Stage 8：非預期例外時必須丟棄整段緩衝區。
                # 若只印錯誤就繼續，造成例外的資料已殘留在 deque 中，
                # 後續最多 window_size 步的視窗都會沿用被污染的緩衝。
                print(f"{RED}STEP 4 ERROR:{e}{RESET}")
                engine.reset_buffer()
                print(f"{YELLOW}STEP 4:已清空串流緩衝區，等待重新累積乾淨視窗{RESET}")

            await asyncio.sleep(frame_interval)


async def handle_client(connection: ServerConnection, clients: set[ServerConnection]) -> None:
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
        flat_stream = X_subject.reshape(-1, 8)
    except Exception as e:
        print(f"{RED}STEP 2 ERROR:{e}{RESET}")
        return

    gate = ClinicalQualityGate(golden_template)
    engine = RealTimeBiofeedbackEngine(gate, golden_template, model)
    clients: set[ServerConnection] = set()

    try:
        print(f"{GREEN}STEP 3:啟動 WebSocket 伺服器於 ws://localhost:{args.port}{RESET}")
        async with serve(
            lambda connection: handle_client(connection, clients), "localhost", args.port
        ):
            print(f"{GREEN}STEP 4:開始模擬受試者 S{args.subject:02} 即時串流 (speed={speed}x){RESET}")
            await stream_subject(engine, flat_stream, args.subject, speed, clients)
    except Exception as e:
        print(f"{RED}STEP 3 ERROR:{e}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
