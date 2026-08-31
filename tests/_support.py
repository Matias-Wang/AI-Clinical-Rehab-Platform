"""測試共用工具：輸出過濾與原始碼符號解析。

供驗收平價、文件一致性與 notebook 相容性等檢查重用，
避免各測試模組各自複製一份雜訊過濾規則而逐漸走樣。
"""

import ast
import io
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TensorFlow 與 absl 在匯入及推論時輸出的訊息，與程式邏輯無關，
# 且內容隨執行環境（CPU 指令集、驅動版本）變動，比對前必須濾除。
NOISE_PATTERN = re.compile(
    r"oneDNN|TF-TRT|cpu_feature_guard|To enable the following|tensorflow/core"
    r"|external/local_xla|WARNING|warnings\.warn|self\._warn|absl::"
    r"|I0000|W0000|E0000"
)


def strip_noise(text: str) -> list[str]:
    """濾除框架雜訊並去除頭尾空行，回傳可供比對的行列表。

    Parameters
    ----------
    text : str
        程式的原始輸出。

    Returns
    -------
    list of str
        濾除雜訊後的行列表，每行已去除行尾換行。
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()
             if not NOISE_PATTERN.search(ln)]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def module_symbols(path: Path) -> set[str]:
    """以 AST 解析模組，取出其頂層公開符號名稱。

    採 AST 而非字串搜尋，避免註解或字串內容造成誤判。

    Parameters
    ----------
    path : pathlib.Path
        要解析的 .py 檔路徑。

    Returns
    -------
    set of str
        頂層函式、類別與常數的名稱集合。
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def class_methods(path: Path, class_name: str) -> set[str]:
    """取出指定類別的方法名稱。

    Parameters
    ----------
    path : pathlib.Path
        要解析的 .py 檔路徑。
    class_name : str
        目標類別名稱。

    Returns
    -------
    set of str
        該類別（含其基底類別為本檔內類別時不遞迴）的方法名稱集合。
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()
