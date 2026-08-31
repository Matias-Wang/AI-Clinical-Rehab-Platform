"""文件與程式碼一致性檢查。

本專案最主要的缺陷模式是「文件寫著一套、程式碼跑著另一套」：

- SPEC.md 第 5 節曾長期記載 `add_sample()`、`get_window()`、`process_window()`
  三個程式碼中根本不存在的方法
- ARCHITECTURE.md 與 README.md 的樹狀圖曾列出 `architecture/`、`Spec/`
  兩個不存在的目錄

這類漂移不會讓任何程式失敗，只會誤導接手者——Fix 3 正是這樣產生的。
本模組將「文件所述必須真實存在」自動化為可執行的斷言。
"""

import io
import re

import pytest

from _support import PROJECT_ROOT, class_methods, module_symbols

pytestmark = pytest.mark.convention

SCHEMA = PROJECT_ROOT / "schema.py"

# 樹狀圖中不代表真實單一路徑的標記（範圍、省略、註解）
TREE_SKIP = ("~", "...", "*", "省略")


def schema_api_names() -> set[str]:
    """schema.py 的所有頂層符號與類別方法名稱。"""
    names = module_symbols(SCHEMA)
    for cls in list(names):
        names |= class_methods(SCHEMA, cls)
    return names


def spec_documented_symbols() -> list[tuple[str, int]]:
    """從 SPEC.md 第 5 節表格抽出被記載的 API 名稱。

    Returns
    -------
    list of (str, int)
        (符號名稱, 行號) 的列表。
    """
    text = io.open(PROJECT_ROOT / "SPEC.md", encoding="utf-8").read()
    start = text.index("## 5. 資料管線介面")
    end = text.index("## 6. ", start)
    section = text[:start].count("\n"), text[start:end]

    found = []
    for offset, line in enumerate(section[1].split("\n")):
        if not line.startswith("|"):
            continue
        for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(", line):
            found.append((m.group(1), section[0] + offset + 1))
    return found


def doc_tree_paths(doc: str) -> list[tuple[str, int]]:
    """從 md 的「資料夾架構」章節抽出被列出的檔案／目錄名稱。

    必須限縮於資料夾架構章節：這兩份文件另有模組依賴圖與驗證架構圖也使用
    相同的樹狀符號，但其節點是模組名稱與敘述文字，不是檔案路徑。

    Parameters
    ----------
    doc : str
        文件檔名（ARCHITECTURE.md 或 README.md）。

    Returns
    -------
    list of (str, int)
        (路徑名稱, 行號) 的列表。
    """
    heading = {"ARCHITECTURE.md": "## 8. 資料夾架構",
               "README.md": "## 專案架構 (Project Structure)"}[doc]
    text = io.open(PROJECT_ROOT / doc, encoding="utf-8").read()
    start = text.index(heading)
    base_line = text[:start].count("\n")
    section = text[start:]
    fence = section.index("```")
    section = section[fence + 3: section.index("```", fence + 3)]
    base_line += section[:0].count("\n") + text[start:start + fence].count("\n") + 1

    out = []
    for offset, line in enumerate(section.split("\n")):
        m = re.match(r"^[│\s]*[├└]──\s+([^\s#]+)", line)
        if not m:
            continue
        name = m.group(1).rstrip("/")
        if any(s in name for s in TREE_SKIP) or not name:
            continue
        out.append((name, base_line + offset + 1))
    return out


class TestSpecApiExists:
    """SPEC.md 記載的 API 必須真實存在於 schema.py。"""

    def test_section_5_is_parseable(self):
        """確保解析邏輯有抓到東西，避免測試因解析失敗而空過。"""
        assert len(spec_documented_symbols()) >= 10, (
            "SPEC.md 第 5 節解析不到足夠的 API 名稱，解析邏輯可能已與文件格式脫節。"
        )

    def test_every_documented_symbol_exists(self):
        """逐一比對文件記載的符號是否存在於程式碼。"""
        actual = schema_api_names()
        missing = [(name, ln) for name, ln in spec_documented_symbols()
                   if name not in actual]
        assert not missing, (
            "SPEC.md 記載了 schema.py 中不存在的 API：\n"
            + "\n".join(f"  SPEC.md:{ln}  `{name}()`" for name, ln in missing)
            + "\n請修正文件以符合實際程式碼，或補上缺少的實作。"
        )


class TestDocTreePathsExist:
    """文件樹狀圖列出的路徑必須真實存在。"""

    @pytest.mark.parametrize("doc", ["ARCHITECTURE.md", "README.md"])
    def test_tree_is_parseable(self, doc):
        """確保樹狀圖解析有效。"""
        assert len(doc_tree_paths(doc)) >= 5, f"{doc} 的樹狀圖解析不到足夠條目。"

    @pytest.mark.parametrize("doc", ["ARCHITECTURE.md", "README.md"])
    def test_every_tree_entry_exists(self, doc):
        """樹狀圖中的每個名稱都應能在專案中找到對應檔案或目錄。"""
        missing = []
        for name, line in doc_tree_paths(doc):
            direct = PROJECT_ROOT / name
            if direct.exists():
                continue
            if any(PROJECT_ROOT.rglob(name)):
                continue
            missing.append((name, line))

        assert not missing, (
            f"{doc} 的資料夾架構列出了不存在的路徑：\n"
            + "\n".join(f"  {doc}:{ln}  {name}" for name, ln in missing)
            + "\n請更新樹狀圖以反映實際專案結構。"
        )


class TestCrossDocConsistency:
    """跨文件的關鍵數值必須一致。"""

    def test_quality_gate_thresholds_match_code(self):
        """SPEC.md 記載的品質閘門門檻必須與 schema.py 實際值相同。"""
        code = io.open(SCHEMA, encoding="utf-8").read()
        spec = io.open(PROJECT_ROOT / "SPEC.md", encoding="utf-8").read()

        for const, value in (("MIN_SAFE_LIMIT", "0.0005"),
                             ("GOLDEN_VAR_LIMIT", "0.001595")):
            assert f"{const} = {value}" in code, (
                f"schema.py 的 {const} 已不是 {value}，"
                "此為 Stage 5 驗收數據的基礎，變動需重新驗收。"
            )
            assert value in spec, f"SPEC.md 未記載 {const} 的實際值 {value}。"
