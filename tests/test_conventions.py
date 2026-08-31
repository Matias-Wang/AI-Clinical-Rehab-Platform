"""CLAUDE.md 程式碼規範的自動化檢查。

CLAUDE.md 訂有明確且可機器驗證的規範，但目前僅靠人工遵循：

- 所有函式必須標註參數與回傳型別
- Docstring 使用 NumPy Style，註解語言為繁體中文
- 每行長度限制 88 字元（Black 標準）
- 要 print 每個步驟的啟動：`print(f"{GREEN}STEP 1:xxx{RESET}")`
- try...except 的 except 區塊要 print 錯誤：`print(f"{RED}STEP 1 ERROR:{e}{RESET}")`

本模組將上述規範轉為可執行斷言，避免規範隨時間流於形式。
"""

import ast
import io
import re

import pytest

from _support import PROJECT_ROOT

pytestmark = pytest.mark.convention

# 受規範約束的生產程式碼（tests/ 自身與 notebook 不在此列）
SOURCE_FILES = ["schema.py", "main.py", "ui_bridge.py", "console.py", "verify.py"]

# 可執行入口點：必須具備 STEP 輸出與例外處理
ENTRY_POINTS = ["main.py", "ui_bridge.py", "verify.py"]

MAX_LINE_LENGTH = 88


def existing_sources() -> list[str]:
    """回傳實際存在的受檢查檔案清單。"""
    return [f for f in SOURCE_FILES if (PROJECT_ROOT / f).exists()]


def parse(path: str) -> tuple[ast.Module, str]:
    """讀取並解析指定檔案。"""
    text = io.open(PROJECT_ROOT / path, encoding="utf-8").read()
    return ast.parse(text), text


def functions_of(tree: ast.Module) -> list[ast.FunctionDef]:
    """取出模組內所有函式與方法定義。"""
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


@pytest.mark.parametrize("path", existing_sources())
class TestTypeHintsAndDocstrings:
    """型別註解與 docstring 覆蓋率。"""

    def test_all_functions_have_type_hints(self, path):
        """所有函式必須標註參數與回傳型別。"""
        tree, _ = parse(path)
        bad = []
        for fn in functions_of(tree):
            args = [a for a in fn.args.args if a.arg not in ("self", "cls")]
            missing_args = [a.arg for a in args if a.annotation is None]
            if missing_args or fn.returns is None:
                detail = []
                if missing_args:
                    detail.append(f"參數 {missing_args}")
                if fn.returns is None:
                    detail.append("回傳值")
                bad.append(f"  {path}:{fn.lineno} {fn.name}() 缺 {'、'.join(detail)}")

        assert not bad, (
            f"{path} 有函式缺少型別註解（CLAUDE.md：所有函式必須標註參數與回傳型別）：\n"
            + "\n".join(bad)
        )

    def test_all_functions_have_docstrings(self, path):
        """所有函式必須具備 docstring。"""
        tree, _ = parse(path)
        bad = [f"  {path}:{fn.lineno} {fn.name}()"
               for fn in functions_of(tree) if ast.get_docstring(fn) is None]
        assert not bad, (
            f"{path} 有函式缺少 docstring（CLAUDE.md：使用 NumPy Style 註解）：\n"
            + "\n".join(bad)
        )


@pytest.mark.parametrize("path", existing_sources())
def test_line_length_within_limit(path):
    """行長度不得超過 88 字元（Black Formatter 標準）。"""
    _, text = parse(path)
    bad = [f"  {path}:{i} （{len(ln)} 字元）"
           for i, ln in enumerate(text.split("\n"), 1) if len(ln) > MAX_LINE_LENGTH]
    assert not bad, (
        f"{path} 有 {len(bad)} 行超過 {MAX_LINE_LENGTH} 字元：\n" + "\n".join(bad[:15])
    )


@pytest.mark.parametrize("path", [p for p in ENTRY_POINTS
                                  if (PROJECT_ROOT / p).exists()])
class TestEntryPointConventions:
    """入口點的步驟輸出與例外處理規範。"""

    def test_has_step_progress_output(self, path):
        """入口點必須以 STEP 格式輸出每個步驟的啟動。"""
        _, text = parse(path)
        # 步驟編號可為字面數字，也可為迴圈變數（f-string 佔位符）；
        # 且色碼前可能有 \n 等前綴，故不要求緊接 print(f" 開頭。
        steps = re.findall(
            r'\{(?:GREEN|CYAN|BLUE|YELLOW)\}STEP\s*(?:\d+|\{[^}]+\})\s*[:：]',
            text,
        )
        assert steps, (
            f"{path} 未使用 CLAUDE.md 規定的步驟輸出格式：\n"
            '  print(f"{GREEN}STEP 1:xxxxxxxxx{RESET}")'
        )

    def test_every_except_block_prints_error(self, path):
        """每個 except 區塊都必須以 STEP N ERROR 格式輸出錯誤。"""
        tree, text = parse(path)
        lines = text.split("\n")
        bad = []
        for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
            body = "\n".join(lines[handler.lineno - 1: handler.end_lineno])
            pattern = r'print\(f"\{RED\}STEP\s*(?:\d+|\{[^}]+\})\s*ERROR'
            if not re.search(pattern, body):
                bad.append(f"  {path}:{handler.lineno}")

        assert not bad, (
            f"{path} 有 except 區塊未輸出錯誤（CLAUDE.md：except 要 print 每個 error）：\n"
            + "\n".join(bad)
            + '\n格式：print(f"{RED}STEP 1 ERROR:{e}{RESET}")'
        )
