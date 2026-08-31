"""驗收輸出平價測試。

執行 `python main.py` 並與 `tests/golden/main_acceptance.txt` 逐行比對。

這是本專案最強的回歸守門員：Fix 2（重力 ÷10.0 遺漏）與 Fix 3（Z-score 與
FFT 縮放遺漏）這兩個曾讓系統邏輯空轉的缺陷，都會直接改變這份輸出。
任何行為變更都會在此被攔下，強迫變更者明確更新基準，而非讓數字默默漂移。

基準更新方式：確認變更為預期行為後，重新執行
    python main.py > tests/golden/main_acceptance.txt
並於 CHANGELOG 明確記錄「基準變更」及其原因。
"""

import io
import os
import subprocess
import sys

import pytest

from _support import PROJECT_ROOT, strip_noise

pytestmark = pytest.mark.acceptance

GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "main_acceptance.txt"


@pytest.fixture(scope="module")
def actual_output() -> list[str]:
    """實際執行 main.py 並回傳濾除雜訊後的輸出行。"""
    if not (PROJECT_ROOT / "models" / "clinical_rehab_model_v3.keras").exists():
        pytest.skip("找不到模型檔，略過驗收平價測試。")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True, cwd=str(PROJECT_ROOT), env=env,
        encoding="utf-8", errors="replace", timeout=1800,
    )
    assert result.returncode == 0, (
        f"main.py 執行失敗 (exit={result.returncode})：\n{result.stderr[-2000:]}"
    )
    return strip_noise(result.stdout)


@pytest.fixture(scope="module")
def golden_output() -> list[str]:
    """讀取 golden 基準行。"""
    if not GOLDEN_PATH.exists():
        pytest.skip(f"找不到基準檔 {GOLDEN_PATH}。")
    return strip_noise(io.open(GOLDEN_PATH, encoding="utf-8").read())


def test_output_matches_golden_line_by_line(actual_output, golden_output):
    """main.py 輸出必須與基準逐行完全相同。"""
    assert len(actual_output) == len(golden_output), (
        f"輸出行數不符：實際 {len(actual_output)} 行，基準 {len(golden_output)} 行。\n"
        "若為預期的行為變更，請更新 tests/golden/main_acceptance.txt 並記錄於 CHANGELOG。"
    )

    diffs = [(i, g, a) for i, (g, a) in enumerate(zip(golden_output, actual_output), 1)
             if g != a]
    if diffs:
        detail = "\n".join(
            f"  第 {i} 行:\n    基準: {g!r}\n    實際: {a!r}" for i, g, a in diffs[:8]
        )
        pytest.fail(
            f"驗收輸出與基準有 {len(diffs)} 行不符：\n{detail}\n\n"
            "這代表系統行為已改變。若為預期變更，請更新基準並於 CHANGELOG 記錄原因。"
        )


def test_all_ten_subjects_present(actual_output):
    """報告必須涵蓋 S01–S10 全部受試者。"""
    joined = "\n".join(actual_output)
    missing = [f"S{i:02}" for i in range(1, 11) if f"S{i:02}" not in joined]
    assert not missing, f"驗收報告缺少受試者：{missing}"


def test_output_is_utf8_safe(actual_output):
    """輸出必須可安全編碼為 UTF-8（cp950 修正的端到端佐證）。"""
    for line in actual_output:
        line.encode("utf-8")
