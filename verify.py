"""驗證閘門單一入口。

本專案的缺陷有共同模式：文件寫著一套、程式碼跑著另一套，而且沒有東西會發現。
Fix 2（重力縮放遺漏）、Fix 3（Z-score 與 FFT 順序）、cp950 崩潰、SPEC.md 記載
不存在的 API——這些都在無人察覺的情況下存活了很久。

本腳本將所有自動化檢查收攏為單一入口，分為兩層：

    python verify.py          每次修改後執行（約 15 秒）
    python verify.py --full   commit 前執行（約 1-3 分鐘）

回傳非零 exit code 代表驗證失敗，可直接接上 git hook 或 CI。
"""

import argparse
import subprocess
import sys

from console import enable_utf8_output

enable_utf8_output()

# <使用者自訂變數>（請勿更動）
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGAENTA = "\033[95m"
RESET = "\033[0m"

# (層級名稱, pytest -m 選擇式, 說明)
QUICK_LAYERS = [
    ("單元與回歸測試", "not slow and not acceptance and not perf and not convention",
     "髒數據防護、緩衝區復原、DTW 性質、預處理修正"),
    ("規範與一致性檢查", "convention",
     "CLAUDE.md 規範、文件漂移、notebook 相容性"),
]

FULL_LAYERS = QUICK_LAYERS + [
    ("模型準確率基準", "slow", "對照訓練 notebook 的 S02/S07/S09/S10 準確率"),
    ("驗收輸出平價", "acceptance", "main.py 輸出與 golden 基準逐行比對"),
    ("效能煙霧測試", "perf", "推論延遲與串流速率"),
]


def parse_args() -> argparse.Namespace:
    """解析命令列參數。

    Returns
    -------
    argparse.Namespace
        含 full 與 quiet 兩個欄位。
    """
    parser = argparse.ArgumentParser(
        description="AI 臨床復健平台 - 驗證閘門",
        epilog="每次修改後執行 verify.py；commit 前執行 verify.py --full。",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="執行完整驗證，含準確率基準、驗收平價與效能測試",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="只輸出各層結果摘要"
    )
    return parser.parse_args()


def run_layer(index: int, name: str, selector: str, quiet: bool) -> tuple[bool, str]:
    """執行單一驗證層。

    Parameters
    ----------
    index : int
        層級序號，用於 STEP 輸出。
    name : str
        層級名稱。
    selector : str
        傳給 pytest -m 的標記選擇式。
    quiet : bool
        是否隱藏 pytest 的詳細輸出。

    Returns
    -------
    tuple of (bool, str)
        (是否通過, pytest 最後一行摘要)。
    """
    cmd = [sys.executable, "-m", "pytest", "-m", selector, "--tb=short", "-q"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=3600
        )
    except subprocess.TimeoutExpired as e:
        print(f"{RED}STEP {index} ERROR:{e}{RESET}")
        return False, "逾時"

    lines = [ln for ln in result.stdout.split("\n") if ln.strip()]
    summary = lines[-1] if lines else "（無輸出）"

    if result.returncode != 0:
        if not quiet:
            print(result.stdout)
        return False, summary
    return True, summary


def main() -> int:
    """執行驗證閘門並回傳 exit code。

    Returns
    -------
    int
        全部通過回傳 0，否則回傳 1。
    """
    args = parse_args()
    layers = FULL_LAYERS if args.full else QUICK_LAYERS
    mode = "完整驗證 (--full)" if args.full else "快速驗證"

    print(f"{BLUE}{'=' * 62}{RESET}")
    print(f"{BLUE}  AI 臨床復健平台 - 驗證閘門（{mode}）{RESET}")
    print(f"{BLUE}{'=' * 62}{RESET}")

    results = []
    for index, (name, selector, description) in enumerate(layers, start=1):
        print(f"\n{GREEN}STEP {index}:{name} — {description}{RESET}")
        try:
            passed, summary = run_layer(index, name, selector, args.quiet)
        except Exception as e:
            print(f"{RED}STEP {index} ERROR:{e}{RESET}")
            passed, summary = False, str(e)

        results.append((name, passed, summary))
        mark = f"{GREEN}通過{RESET}" if passed else f"{RED}失敗{RESET}"
        print(f"  {mark}  {summary}")

    print(f"\n{BLUE}{'=' * 62}{RESET}")
    failed = [name for name, passed, _ in results if not passed]
    for name, passed, summary in results:
        icon = f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"
        print(f"  {icon} {name:<20} {summary}")
    print(f"{BLUE}{'=' * 62}{RESET}")

    if failed:
        print(f"{RED}驗證失敗：{'、'.join(failed)}{RESET}")
        if not args.full:
            print(f"{YELLOW}提示：commit 前請執行 python verify.py --full{RESET}")
        return 1

    print(f"{GREEN}全部通過。{RESET}")
    if not args.full:
        print(f"{YELLOW}提示：commit 前請執行 python verify.py --full{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
