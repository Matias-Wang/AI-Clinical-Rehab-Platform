"""訓練 notebook 的相容性守門。

`development_history/*.ipynb` 透過 `import schema` 與生產程式碼共用資料管線。
若 `schema.py` 移除或改名任何被 notebook 引用的符號，訓練流程會直接壞掉，
而這在一般測試中完全看不出來——notebook 不會被 pytest 執行。

本模組解析 notebook 原始碼，抽出所有 `schema.X` 參照並斷言其存在。

已實測確認的依賴（不得移除）：
    align_coordinates              20260404_Optimization（Fix 3 基準的對照 notebook）
    get_final_training_data        20260222_Consistency_Test
    get_all_subjects_for_analysis  20260228 / 20260307 / 20260403 / 20260404
    PHYSICAL_CONSTANTS             20260228
    ACTIVITY_LABELS                20260307 / 20260403
    ClinicalQualityGate            20260404

注意 `align_coordinates` 與 `get_final_training_data` 在生產程式碼中確實
沒有呼叫點，靜態掃描極易誤判為死碼。此測試即為防止該誤判造成的破壞。
"""

import ast
import io
import json
import re

import pytest

from _support import PROJECT_ROOT, module_symbols

pytestmark = pytest.mark.convention

NOTEBOOK_DIR = PROJECT_ROOT / "development_history"
SCHEMA = PROJECT_ROOT / "schema.py"

# 即使當下的 notebook 掃描結果有變，這些符號仍不得消失。
# 明列於此，避免 notebook 資料夾缺席時失去保護。
PINNED_SYMBOLS = [
    "align_coordinates",
    "get_final_training_data",
    "get_all_subjects_for_analysis",
    "load_and_preprocess_subject",
    "PHYSICAL_CONSTANTS",
    "ACTIVITY_LABELS",
    "ClinicalQualityGate",
]


def notebook_sources() -> dict[str, str]:
    """讀取所有 notebook 的程式碼儲存格內容。

    Returns
    -------
    dict
        {notebook 檔名: 合併後的程式碼字串}。
    """
    out = {}
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        doc = json.load(io.open(path, encoding="utf-8"))
        out[path.name] = "\n".join(
            "".join(c.get("source", []))
            for c in doc.get("cells", []) if c.get("cell_type") == "code"
        )
    return out


def referenced_symbols(source: str) -> set[str]:
    """抽出程式碼中所有 `schema.X` 形式的參照。

    以 AST 解析而非正規表示式：notebook 內的註解（「# 重新載入 schema.py」）
    與字串字面值（「請檢查 schema.py 是否存檔成功」）都含有 `schema.` 字樣，
    字串比對會把副檔名 `py` 誤判為符號名稱。AST 只看真正的屬性存取。

    Parameters
    ----------
    source : str
        notebook 的程式碼儲存格內容。

    Returns
    -------
    set of str
        被參照的 schema 符號名稱集合；程式碼無法解析時回傳空集合。
    """
    # Jupyter magic 與 shell 指令不是合法 Python，解析前先移除
    code = "\n".join("" if re.match(r"\s*[%!]", ln) else ln
                     for ln in source.split("\n"))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    return {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "schema"
    }


@pytest.mark.parametrize("symbol", PINNED_SYMBOLS)
def test_pinned_symbol_still_exists(symbol):
    """釘選符號必須持續存在於 schema.py。

    這些符號部分在生產程式碼中無呼叫點，容易被靜態掃描誤判為死碼而刪除。
    """
    assert symbol in module_symbols(SCHEMA), (
        f"`{symbol}` 已從 schema.py 消失，訓練 notebook 會直接壞掉。\n"
        "此符號可能在生產程式碼中無呼叫點，但 development_history/ 的 notebook "
        "有使用。請勿以「死碼」為由移除。"
    )


class TestNotebookReferences:
    """實際掃描 notebook 內容進行比對。"""

    def test_notebook_dir_present_or_skip(self):
        """notebook 目錄未納入版控，缺席時後續測試會 skip。"""
        if not NOTEBOOK_DIR.is_dir():
            pytest.skip("development_history/ 不存在（未納入版控），略過實際掃描。")
        assert any(NOTEBOOK_DIR.glob("*.ipynb")), "development_history/ 內沒有 notebook。"

    def test_all_referenced_symbols_exist(self):
        """每個 notebook 參照的 schema 符號都必須存在。"""
        if not NOTEBOOK_DIR.is_dir():
            pytest.skip("development_history/ 不存在，略過。")

        actual = module_symbols(SCHEMA)
        missing = []
        for name, source in notebook_sources().items():
            for sym in sorted(referenced_symbols(source)):
                if sym not in actual:
                    missing.append(f"  {name} → schema.{sym}")

        assert not missing, (
            "訓練 notebook 參照了 schema.py 中不存在的符號：\n"
            + "\n".join(missing)
            + "\n請恢復該符號，或同步更新 notebook。"
        )

    def test_pipeline_entry_points_are_stable(self):
        """notebook 共用的管線進入點簽章不得改變參數名稱。

        notebook 多以 `schema.get_all_subjects_for_analysis()` 無參數呼叫，
        依賴其 folder_path 預設值指向實際資料夾。
        """
        if not NOTEBOOK_DIR.is_dir():
            pytest.skip("development_history/ 不存在，略過。")

        # 以 AST 取預設值，而非比對簽章字串：函式簽章可能因格式化而換行，
        # 正規表示式會因此失效並讓測試在無人察覺下失去保護力。
        tree = ast.parse(io.open(SCHEMA, encoding="utf-8").read())
        target = next(
            (n for n in tree.body
             if isinstance(n, ast.FunctionDef)
             and n.name == "get_all_subjects_for_analysis"),
            None,
        )
        assert target is not None, "schema.py 找不到 get_all_subjects_for_analysis()。"

        arg_names = [a.arg for a in target.args.args]
        assert "folder_path" in arg_names, (
            "get_all_subjects_for_analysis() 已無 folder_path 參數，"
            "notebook 的無參數呼叫可能失效。"
        )

        # 預設值由後往前對齊參數列表
        defaults = dict(zip(arg_names[len(arg_names) - len(target.args.defaults):],
                            target.args.defaults))
        node = defaults.get("folder_path")
        assert isinstance(node, ast.Constant), (
            "folder_path 已無字面預設值，notebook 的無參數呼叫會失敗。"
        )
        assert (PROJECT_ROOT / node.value).is_dir(), (
            f"get_all_subjects_for_analysis() 的預設資料夾 '{node.value}' 不存在，"
            "notebook 的無參數呼叫會失敗。"
        )
