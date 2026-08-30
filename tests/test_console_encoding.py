"""cp950 輸出編碼回歸測試（Stage 8）。

Windows 繁中環境 locale 為 cp950，輸出被重導至管道時，訊息中的
emoji（✅ ⚠️ ❌ 🔌）會觸發 UnicodeEncodeError 使程式中斷。
本模組以子行程強制 PYTHONIOENCODING=cp950 重現該情境，
確保 console.enable_utf8_output() 確實解除此限制。
"""

import io
import os
import subprocess
import sys

import pytest

from conftest import PROJECT_ROOT
from console import enable_utf8_output

# 專案訊息中實際使用、且無法以 cp950 編碼的字元
NON_CP950_CHARS = ["✅", "⚠️", "❌", "🔌", "🧪", "📈"]


def run_child(code: str) -> subprocess.CompletedProcess:
    """在強制 cp950 的環境下以子行程執行程式碼，stdout 為管道。

    Parameters
    ----------
    code : str
        要執行的 Python 程式碼。

    Returns
    -------
    subprocess.CompletedProcess
        子行程執行結果，stdout/stderr 以 UTF-8 解碼。
    """
    env = dict(os.environ, PYTHONIOENCODING="cp950", PYTHONPATH=str(PROJECT_ROOT))
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )


class TestCp950Regression:
    """重現並驗證 cp950 崩潰情境。"""

    def test_child_without_fix_crashes(self):
        """對照組：未套用修正時，cp950 下輸出 emoji 必定崩潰。

        此測試證明下方的修正測試確實有鑑別力，而非恆真。
        """
        result = run_child("print('✅ 測試完成')")

        assert result.returncode != 0
        assert "UnicodeEncodeError" in result.stderr

    def test_child_with_fix_succeeds(self):
        """套用 enable_utf8_output() 後，同樣的輸出應正常完成。"""
        result = run_child(
            "from console import enable_utf8_output\n"
            "enable_utf8_output()\n"
            "print('✅ 測試完成')\n"
        )

        assert result.returncode == 0, f"仍然崩潰：{result.stderr}"
        assert "✅ 測試完成" in result.stdout

    @pytest.mark.parametrize("char", NON_CP950_CHARS)
    def test_all_project_symbols_are_printable(self, char):
        """專案實際使用的每個非 cp950 字元都必須能安全輸出。"""
        result = run_child(
            "from console import enable_utf8_output\n"
            "enable_utf8_output()\n"
            f"print({char!r})\n"
        )

        assert result.returncode == 0, f"字元 {char!r} 仍造成崩潰：{result.stderr}"
        assert char in result.stdout


class TestEntryPointsAreProtected:
    """每個可執行入口點都必須在輸出前套用修正。"""

    @pytest.mark.parametrize("entry_point", ["main.py", "ui_bridge.py"])
    def test_entry_point_calls_enable_utf8_output(self, entry_point):
        """入口點原始碼中必須呼叫 enable_utf8_output()。"""
        source = io.open(PROJECT_ROOT / entry_point, encoding="utf-8").read()
        assert "enable_utf8_output()" in source, (
            f"{entry_point} 未套用 UTF-8 輸出設定，重導輸出時會因 cp950 崩潰。"
        )

    def test_ui_bridge_help_runs_under_cp950(self):
        """ui_bridge.py 的參數說明在 cp950 管道下應可正常輸出。"""
        result = run_child(
            "import sys; sys.argv = ['ui_bridge.py', '--help']\n"
            "exec(open('ui_bridge.py', encoding='utf-8').read())\n"
        )
        assert result.returncode == 0, f"--help 崩潰：{result.stderr}"
        assert "--subject" in result.stdout


class TestHelperBehaviour:
    """enable_utf8_output() 本身的行為契約。"""

    def test_is_idempotent(self):
        """重複呼叫不得產生副作用或例外。"""
        first = enable_utf8_output()
        second = enable_utf8_output()
        assert first == second

    def test_handles_stream_without_reconfigure(self, monkeypatch):
        """串流不支援 reconfigure（例如被測試框架取代）時應安全略過。"""
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        assert enable_utf8_output() is False

    def test_handles_none_stream(self, monkeypatch):
        """以 pythonw 執行時 stdout 可能為 None，不得拋出例外。"""
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)
        assert enable_utf8_output() is False
