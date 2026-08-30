"""終端機輸出編碼設定（Stage 8）。

問題背景
--------
Windows 繁體中文環境的 locale 預設編碼為 cp950。當 Python 的標準輸出被重導
到管道或檔案時（例如 `python main.py > report.log`、CI 收集輸出、或以
subprocess 啟動服務），stdout 不再是主控台而會退回 locale 編碼，此時本專案
訊息中的 `✅`、`⚠️`、`❌`、`🔌` 等字元無法以 cp950 編碼，會直接拋出
UnicodeEncodeError 並中斷整支程式。

實際案例：`python main.py > out.log` 會在印出第一位受試者結果時崩潰，
批量驗收報告因此永遠無法完整輸出。

解法
----
在各入口點啟動時將 stdout/stderr 重新設定為 UTF-8。直接在主控台執行時
Python 本就使用 Windows Unicode API，此設定為無害的 no-op；被重導時則
確保輸出為 UTF-8 而非 cp950。另外統一加上 errors="replace"，即使未來
出現無法編碼的字元也只會顯示替代符號，而不會讓臨床系統整個中斷。
"""

import sys
from typing import TextIO


def _reconfigure_stream(stream: TextIO | None) -> bool:
    """嘗試將單一輸出串流重新設定為 UTF-8。

    Parameters
    ----------
    stream : typing.TextIO or None
        要設定的串流，通常為 sys.stdout 或 sys.stderr。
        以 pythonw 執行或被測試框架取代時可能為 None 或不支援 reconfigure。

    Returns
    -------
    bool
        成功設定為 UTF-8 回傳 True；串流不支援設定則回傳 False。
    """
    if stream is None or not hasattr(stream, "reconfigure"):
        return False

    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
        return True
    except (ValueError, OSError):
        # 串流已關閉或不允許變更編碼時，退而求其次只放寬錯誤處理，
        # 至少確保不會因單一字元而讓整支程式崩潰。
        try:
            stream.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
        return False


def enable_utf8_output() -> bool:
    """將 stdout 與 stderr 設定為 UTF-8，避免 cp950 編碼錯誤。

    應在每個可執行入口點的最開頭呼叫（早於任何 print 與可能輸出訊息的
    第三方套件匯入）。本函式為冪等操作，重複呼叫不會產生副作用。

    Returns
    -------
    bool
        stdout 與 stderr 皆成功設定為 UTF-8 時回傳 True。
    """
    stdout_ok = _reconfigure_stream(sys.stdout)
    stderr_ok = _reconfigure_stream(sys.stderr)
    return stdout_ok and stderr_ok
