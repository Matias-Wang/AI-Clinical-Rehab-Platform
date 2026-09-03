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


def _catches_only(handler: ast.ExceptHandler, names: set[str]) -> bool:
    """判斷 except 子句是否只捕捉指定的例外型別。

    Parameters
    ----------
    handler : ast.ExceptHandler
        要檢查的 except 子句。
    names : set of str
        視為「非錯誤」的例外型別名稱。

    Returns
    -------
    bool
        該子句捕捉的型別全部落在 names 內時回傳 True。
    """
    node = handler.type
    if node is None:
        return False
    targets = node.elts if isinstance(node, ast.Tuple) else [node]
    caught = {n.id for n in targets if isinstance(n, ast.Name)}
    return bool(caught) and caught <= names


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
        """每個 except 區塊都必須以 STEP N ERROR 格式輸出錯誤。

        允許透過輔助函式輸出：只要該函式自身的內容符合規定格式即可。
        把錯誤輸出抽成 `log_error()` 這類函式比在每個 except 內複製一份
        print 更好，檢查器不應懲罰這種寫法。
        """
        tree, text = parse(path)
        lines = text.split("\n")
        pattern = r'print\(f"\{RED\}STEP\s*(?:\d+|\{[^}]+\})\s*ERROR'

        # 先找出模組內「本身就會輸出 STEP ERROR」的輔助函式
        error_loggers = set()
        for fn in functions_of(tree):
            body = "\n".join(lines[fn.lineno - 1: fn.end_lineno])
            if re.search(pattern, body):
                error_loggers.add(fn.name)

        bad = []
        for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
            # KeyboardInterrupt / SystemExit 屬正常控制流（使用者主動中止），
            # 不是錯誤，不應強制以 ERROR 格式輸出。
            if _catches_only(handler, {"KeyboardInterrupt", "SystemExit"}):
                continue

            body = "\n".join(lines[handler.lineno - 1: handler.end_lineno])
            if re.search(pattern, body):
                continue
            called = {n.func.id for n in ast.walk(handler)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            if called & error_loggers:
                continue
            bad.append(f"  {path}:{handler.lineno}")

        assert not bad, (
            f"{path} 有 except 區塊未輸出錯誤（CLAUDE.md：except 要 print 每個 error）：\n"
            + "\n".join(bad)
            + '\n格式：print(f"{RED}STEP 1 ERROR:{e}{RESET}")'
        )
