from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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


def test_apply_patch_supports_line_range_fallback_format(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    patch_text = """*** Begin Patch
*** Update File: sample.txt
@@2-2
beta fixed
*** End Patch
"""

    patch_module.apply_patch_to_workspace(patch_text, tmp_path)
    assert target.read_text(encoding="utf-8") == "alpha\nbeta fixed\ngamma\n"


def test_apply_patch_supports_edit_blocks_with_original_line_numbers(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

    patch_text = """*** Begin Patch
*** Update File: sample.txt
<EDIT replace=2-3>
beta fixed
gamma fixed
</EDIT>
<EDIT insert=4>
inserted before delta
</EDIT>
*** End Patch
"""

    patch_module.apply_patch_to_workspace(patch_text, tmp_path)
    assert target.read_text(encoding="utf-8") == (
        "alpha\n"
        "beta fixed\n"
        "gamma fixed\n"
        "inserted before delta\n"
        "delta\n"
    )


def test_apply_patch_edit_blocks_allow_delete_only_and_append(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    patch_text = """*** Begin Patch
*** Update File: sample.txt
<EDIT replace=2-2>
</EDIT>
<EDIT insert=4>
four
</EDIT>
*** End Patch
"""

    patch_module.apply_patch_to_workspace(patch_text, tmp_path)
    assert target.read_text(encoding="utf-8") == "one\nthree\nfour\n"


def test_apply_patch_rejects_non_update_operations(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    patch_text = """*** Begin Patch
*** Add File: sample.txt
+hello
*** End Patch
"""

    with pytest.raises(patch_module.PatchParseError):
        patch_module.apply_patch_to_workspace(patch_text, tmp_path)


def test_apply_patch_rejects_overlapping_line_ranges(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    patch_text = """*** Begin Patch
*** Update File: sample.txt
<EDIT replace=2-3>
X
</EDIT>
<EDIT replace=3-4>
Y
</EDIT>
*** End Patch
"""

    with pytest.raises(patch_module.PatchApplyError):
        patch_module.apply_patch_to_workspace(patch_text, tmp_path)


def test_apply_patch_rejects_mixed_edit_and_line_range_hunks(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    patch_text = """*** Begin Patch
*** Update File: sample.txt
<EDIT replace=2-2>
two fixed
</EDIT>
@@3
inserted
*** End Patch
"""

    with pytest.raises(patch_module.PatchParseError):
        patch_module.apply_patch_to_workspace(patch_text, tmp_path)


def test_apply_patch_rejects_empty_insertion_chunk(tmp_path: Path) -> None:
    patch_module = _load_apply_patch_module()
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    patch_text = """*** Begin Patch
*** Update File: sample.txt
<EDIT insert=2>
</EDIT>
*** End Patch
"""

    with pytest.raises(patch_module.PatchApplyError):
        patch_module.apply_patch_to_workspace(patch_text, tmp_path)
