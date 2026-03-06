from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_apply_patch_module():
    module_path = Path(__file__).resolve().parents[1] / "seimei" / "editing" / "apply_patch.py"
    spec = importlib.util.spec_from_file_location("seimei_editing_apply_patch", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_apply_patch_accepts_unprefixed_context_lines(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    patch_text = """*** Begin Patch
*** Update File: sample.txt
@@
alpha
-beta
+beta fixed
gamma
*** End Patch
"""

    patch_module.apply_patch_to_workspace(patch_text, tmp_path)
    assert target.read_text(encoding="utf-8") == "alpha\nbeta fixed\ngamma\n"


def test_apply_patch_falls_back_to_core_replacement_when_context_drifts(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    target = tmp_path / "sample.txt"
    target.write_text("line one\ntarget value\nline three\n", encoding="utf-8")

    patch_text = """*** Begin Patch
*** Update File: sample.txt
@@
 stale context above
-target value
+updated value
 stale context below
*** End Patch
"""

    patch_module.apply_patch_to_workspace(patch_text, tmp_path)
    assert target.read_text(encoding="utf-8") == "line one\nupdated value\nline three\n"
