import argparse
import asyncio
import csv
import json
import random
import shutil
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from seimei import load_run_messages, seimei
from seimei.editing import PatchApplyError, PatchParseError, apply_patch_to_workspace

# v6 notes (vs v5):
# - Simplified trials: no-knowledge runs followed by knowledge-enabled runs.
# - Knowledge selection uses seimei's built-in sampling (knowledge_search_config).
# - Knowledge pool auto-updates after scoring and is persisted to knowledge_v6.csv.

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
PATCH_DIR = EXP_DIR / "patch_files"
DEFAULT_DATASET_PATH = EXP_DIR / "dataset.json"
DEFAULT_RESULT_PATH = EXP_DIR / "train_v6_results.json"
DEFAULT_LLM_MODEL_NAME = "/workspace/gpt-oss-20b"
DEFAULT_LLM_URL = "https://94sownu2ebkgwm-8000.proxy.runpod.net/v1"  # Set None if you use openai model
DEFAULT_RM_URL = None
#DEFAULT_RM_URL = "https://oyl94a4yv16q5y-8000.proxy.runpod.net/rmsearch"
DEFAULT_BATCH_SIZE = 100
DEFAULT_TOP_N_SAMPLE_KLG = 5
DEFAULT_DISTRIBUTION_DECAY_RATE = 0.5
DEFAULT_RANDOM_KLG_SAMPLING_RATE = 0.2
DEFAULT_KLG_SAMPLE_MODE = "llm"
DEFAULT_N_NO_KLG_TRIALS = 3
DEFAULT_N_KLG_TRIALS = 7
DEFAULT_KLG_POOL_LOAD_PATH = EXP_DIR / "knowledge_v6.csv"
DEFAULT_ENABLE_UPDATE_KLG_POOL = True
DEFAULT_FINAL_KLG_POOL_SAVE_PATH = EXP_DIR / "knowledge_v6.csv"
WORKSPACE_ROOT = EXP_DIR / "_workspace_copies"

BASE_SYSTEM_PROMPT_LIST = [
    "Act as a senior Fortran plasma physicist: inspect the local repo, reason about the magnetic-field terms, "
    "edit the source carefully, and summarize the exact patches you applied.",
    "Work like a responsible HPC debugger—diff the relevant modules, trace the control flow, and document the "
    "precise code edits that resolve the regression.",
    "Channel a gyrokinetic code maintainer: read the bug report, open the Fortran files, add or remove lines "
    "surgically, then explain why the change restores the missing physics.",
    "Be a cautious tokamak simulation engineer who tests hypotheses against the source, edits with apply_patch, "
    "and double-checks each assumption before writing the final note.",
    "Think like a numerical physicist reviewing electromagnetic solvers: inspect coefficients, restore missing "
    "operators, and narrate the fix with references to specific routines.",
    "Operate as a debugging lead: outline the failure mode, open the file, add the missing calls or loops, and "
    "describe the scientific impact of the change.",
    "Take the role of a patch surgeon—identify the minimal diff required, keep the edits consistent with coding "
    "style, and justify how the fix affects simulations.",
    "Behave as an HPC maintainer who validates interface contracts, reinstates removed code paths, and records "
    "the before/after physics effect.",
    "Think as a code-review mentor: reproduce the bug mentally, craft the Fortran changes step by step, and "
    "document what each edit re-enables.",
    "Act like an integration engineer ensuring GKV regressions are fixed; reason about boundary conditions, "
    "tweak the loops, and clearly explain the resulting behavior.",
]

KLG_SYSTEM_PROMPT_LIST = [
    "Treat each knowledge snippet as a mini patch review: restate the code cue, inspect the matching lines, and explain how to adjust them.",
    "Use the knowledge hints as guardrails by quoting the routine or loop they mention, checking that context, and guiding the edit there.",
    "Think like a reviewer handing you TODO comments—translate each snippet into a concrete Fortran action and report the result.",
    "Anchor every move to the knowledge cue: name the variables it references, open that block, and describe the precise change.",
    "Consider the knowledge text mandatory checkpoints; for each, cite the routine, verify current behavior, and note the edit to make.",
    "Behave as a cautious maintainer who paraphrases the knowledge, inspects the code around it, and ties conclusions directly back.",
    "Use the knowledge to prioritize lines: mention the snippet, map it to the workspace, and describe the instrumentation or edit you will run.",
    "Let the knowledge drive your micro-plan—quote it, echo the relevant arrays or flags, and keep reasoning tethered to that instruction.",
    "Imagine the knowledge as diff hunks; state the intended change, verify the file, and reason about its impact before moving on.",
    "Weave the knowledge cues into your debugging narrative by mirroring their language and pointing to the exact Fortran constructs involved.",
]

DEFAULT_KNOWLEDGE_POOL: List[Dict[str, Any]] = [
    {
        "id": "code_ls_inventory",
        "agent": "code_act",
        "step": None,
        "text": "Use `ls -a` to inventory the repo before touching files so you know which modules, patches, or scripts exist.",
        "tags": ["shell", "ls", "context"],
    },
    {
        "id": "code_pwd_confirm",
        "agent": "code_act",
        "step": None,
        "text": "Run `pwd` and confirm you are inside the experiment workspace before editing paths.",
        "tags": ["shell", "pwd", "sanity-check"],
    },
    {
        "id": "code_rg_search",
        "agent": "code_act",
        "step": None,
        "text": "Run `rg -n \"keyword\"` across the repo to locate definitions or uses before assuming semantics.",
        "tags": ["shell", "rg", "search"],
    },
    {
        "id": "code_rg_fortran_files",
        "agent": "code_act",
        "step": None,
        "text": "List Fortran sources with `rg --files -g '*.f90'` to scope the search surface.",
        "tags": ["shell", "rg", "inventory"],
    },
    {
        "id": "code_rg_module",
        "agent": "code_act",
        "step": None,
        "text": "Search for module or routine names with `rg -n \"module_name\" -g '*.f90'` to find declarations.",
        "tags": ["shell", "rg", "fortran"],
    },
    {
        "id": "code_rg_call_sites",
        "agent": "code_act",
        "step": None,
        "text": "Use `rg -n \"call routine_name\" -g '*.f90'` to find call sites and verify control flow.",
        "tags": ["shell", "rg", "call-graph"],
    },
    {
        "id": "code_rg_flags",
        "agent": "code_act",
        "step": None,
        "text": "Track a config flag via `rg -n \"flag_name\"` to see where branches diverge.",
        "tags": ["shell", "rg", "config"],
    },
    {
        "id": "code_head_preview",
        "agent": "code_act",
        "step": None,
        "text": "Call `head -n 40 path/to/file` to see file headers, module names, and includes.",
        "tags": ["shell", "head", "preview"],
    },
    {
        "id": "code_tail_logs",
        "agent": "code_act",
        "step": None,
        "text": "Use `tail -n 40` on build or run logs to spot the most recent error lines.",
        "tags": ["shell", "tail", "sanity-check"],
    },
    {
        "id": "code_wc_loc",
        "agent": "code_act",
        "step": None,
        "text": "`wc -l file.f90` gives a quick sense of file size before deep edits.",
        "tags": ["shell", "wc", "metrics"],
    },
    {
        "id": "code_sed_context",
        "agent": "code_act",
        "step": None,
        "text": "Use `sed -n '120,200p' file.f90` or `nl -ba file.f90 | sed -n '120,200p'` to view numbered context.",
        "tags": ["shell", "sed", "preview"],
    },
    {
        "id": "code_diff_versions",
        "agent": "code_act",
        "step": None,
        "text": "Use `git diff -U3 path/to/file` to compare local edits with expected changes.",
        "tags": ["shell", "git", "diff"],
    },
    {
        "id": "code_git_status",
        "agent": "code_act",
        "step": None,
        "text": "Run `git status -sb` to confirm which files changed before and after patching.",
        "tags": ["shell", "git", "status"],
    },
    {
        "id": "code_rg_preproc",
        "agent": "code_act",
        "step": None,
        "text": "Search preprocessor guards with `rg -n \"#if|#ifdef\" -g '*.[Ff]90'` to check compile-time branches.",
        "tags": ["shell", "rg", "preprocessor"],
    },
    {
        "id": "code_rg_parameters",
        "agent": "code_act",
        "step": None,
        "text": "Find constants with `rg -n \"parameter\" -g '*.f90'` to locate default values.",
        "tags": ["shell", "rg", "constants"],
    },
    {
        "id": "code_python_find_defs",
        "agent": "code_act",
        "step": None,
        "text": "Use a short Python regex to list subroutine/module names so you can map the file structure quickly.",
        "tags": ["python", "parsing", "fortran"],
    },
    {
        "id": "code_python_find_assign",
        "agent": "code_act",
        "step": None,
        "text": "Write a Python snippet to scan for assignments to a target variable and print line numbers.",
        "tags": ["python", "search", "variables"],
    },
    {
        "id": "code_python_extract_block",
        "agent": "code_act",
        "step": None,
        "text": "Use Python to extract and print a specific block (between two markers) when the file is large.",
        "tags": ["python", "parsing", "context"],
    },
    {
        "id": "code_python_call_graph",
        "agent": "code_act",
        "step": None,
        "text": "Use Python to scan `call` statements across files and build a quick caller list for a routine.",
        "tags": ["python", "call-graph", "analysis"],
    },
    {
        "id": "code_python_compare_files",
        "agent": "code_act",
        "step": None,
        "text": "Use Python's difflib to compare two versions of a source file and isolate the minimal delta.",
        "tags": ["python", "diff", "analysis"],
    },
    {
        "id": "code_rg_runtime_params",
        "agent": "code_act",
        "step": None,
        "text": "Search namelist or input parameters with `rg -n \"&\" -g '*.nml'` or `rg -n \"namelist\"` to trace runtime flags.",
        "tags": ["shell", "rg", "runtime"],
    },
    {
        "id": "code_find_config_files",
        "agent": "code_act",
        "step": None,
        "text": "List relevant configs with `rg --files -g '*.nml' -g '*.in'` so you know where inputs live.",
        "tags": ["shell", "rg", "config"],
    },
    {
        "id": "code_rg_todo",
        "agent": "code_act",
        "step": None,
        "text": "Use `rg -n \"TODO|FIXME\"` to see if there are hints about known issues.",
        "tags": ["shell", "rg", "context"],
    },
    {
        "id": "code_python_symbol_counts",
        "agent": "code_act",
        "step": None,
        "text": "Use Python to count how often a symbol appears per file to prioritize where to inspect first.",
        "tags": ["python", "metrics", "analysis"],
    },
    {
        "id": "code_checksum_validate",
        "agent": "code_act",
        "step": None,
        "text": "Compute `md5sum path/to/file` before and after edits to confirm you are modifying the intended file.",
        "tags": ["shell", "md5sum", "integrity"],
    },
    {
        "id": "reasoning_refocus_think_1",
        "agent": "think",
        "step": None,
        "text": "This reasoning path seems off. Step back, restate the objective, and try a different approach.",
        "tags": ["reasoning", "course-correct"],
    },
    {
        "id": "reasoning_refocus_think_2",
        "agent": "think",
        "step": None,
        "text": "You're leaning on an assumption without evidence. Verify it against the repo or logs before continuing.",
        "tags": ["reasoning", "verification"],
    },
    {
        "id": "reasoning_refocus_think_3",
        "agent": "think",
        "step": None,
        "text": "If the current plan isn't yielding results, widen the hypothesis and validate with file evidence.",
        "tags": ["reasoning", "course-correct"],
    },
    {
        "id": "reasoning_refocus_code_act_1",
        "agent": "code_act",
        "step": None,
        "text": "The file you investigated seems unrelated. Look for other files that are more relevant to the issue.",
        "tags": ["code_act", "course-correct"],
    },
    {
        "id": "reasoning_refocus_code_act_2",
        "agent": "code_act",
        "step": None,
        "text": "No signal from that file. Broaden the `rg` search or pivot to config/namelist inputs.",
        "tags": ["code_act", "search", "course-correct"],
    },
    {
        "id": "investigate_src_gkvp_advnc",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_advnc.f90. This file is about RHS evaluation and time advance via Runge-Kutta-Gill.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_bndry",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_bndry.f90. This file is about MPI boundary communications in z, v_l, and mu.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_clock",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_clock.f90. This file is about elapsed time measurements and timers.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_colli",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_colli.f90. This file is about the collision operator.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_colliimp",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_colliimp.f90. This file is about the implicit collision solver.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_colliimp_soa",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_colliimp_SoA.f90. This file is about the implicit collision solver in SoA layout.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_dtc",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_dtc.f90. This file is about time-step size control.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_exb",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_exb.f90. This file is about the E x B term.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_f0_56_advnc_tune_nec1",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_f0.56_advnc_tune_nec1.f90. This file is about tuned f0.56/NEC1 advance logic with flux-surface and field-line averages.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_f0_56_bndry_tune_nec1",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_f0.56_bndry_tune_nec1.f90. This file is about a tuned boundary/utilities variant for f0.56/NEC1.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_f0_56_colli_tune_nifs",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_f0.56_colli_tune_nifs.f90. This file is about a tuned collision operator for f0.56/NIFS.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_f0_56_exb_tune2r_0813",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_f0.56_exb_tune2r_0813.f90. This file is about a tuned E x B term implementation.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_f0_56_fft_fftw_tune2r_0813",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_f0.56_fft_fftw_tune2r_0813.f90. This file is about a tuned FFT module for E x B using SSL2.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_f0_56_zfilter_tune_nec1",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_f0.56_zfilter_tune_nec1.f90. This file is about a tuned z-filter/utilities variant for f0.56/NEC1.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_fft_fftw",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_fft_fftw.f90. This file is about the FFTW-based FFT module for E x B.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_fileio_fortran",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_fileio_fortran.f90. This file is about Fortran binary output I/O.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_fileio_netcdf",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_fileio_netcdf.f90. This file is about NetCDF output I/O.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_fileio_zarr",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_fileio_zarr.f90. This file is about the Zarr output backend in the file I/O module.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_fld",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_fld.f90. This file is about the field solver.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_freq",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_freq.f90. This file is about linear growth rates and real frequencies without shear flows.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_geom",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_geom.f90. This file is about geometric constants.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_header",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_header.f90. This file is about common headers and shared parameters for the fluxtube code.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_igs",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_igs.f90. This file is about magnetic field components and metric coefficients from MEUDAS or G-EQDSK using IGS.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_intgrl",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_intgrl.f90. This file is about flux-surface/field-line averages and velocity-space integrals.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_main",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_main.f90. This file is about the main program entry for GKV+.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_mpienv",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_mpienv.f90. This file is about MPI environment headers and settings.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_out",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_out.f90. This file is about data writing outputs.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_ring",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_ring.f90. This file is about ring-geometry helpers and elliptic integral utilities.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_set",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_set.f90. This file is about reading namelist parameters and setting I/O.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_shearflow",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_shearflow.f90. This file is about the shearflow convection term.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_tips",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_tips.f90. This file is about shared utility helpers and tips.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_trans",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_trans.f90. This file is about entropy transfer diagnostics.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_vmecbzx",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_vmecbzx.f90. This file is about VMEC-based magnetic geometry using the BZX code.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_vmecin",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_vmecin.f90. This file is about VMEC-based magnetic field and metric coefficients.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_src_gkvp_zfilter",
        "agent": "code_act",
        "step": None,
        "text": "Investigate src/gkvp_zfilter.f90. This file is about filtering in z to reduce high-kz oscillations.",
        "tags": ["code_act", "inspect", "src"],
    },
    {
        "id": "investigate_run_makefile",
        "agent": "code_act",
        "step": None,
        "text": "Investigate run/Makefile. This file is about build configuration, compiler flags, and library selection.",
        "tags": ["code_act", "inspect", "run"],
    },
    {
        "id": "investigate_run_gkvp_namelist",
        "agent": "code_act",
        "step": None,
        "text": "Investigate run/gkvp_namelist. This file is about runtime parameters and namelist settings.",
        "tags": ["code_act", "inspect", "run"],
    },
    {
        "id": "investigate_run_shoot",
        "agent": "code_act",
        "step": None,
        "text": "Investigate run/shoot. This file is about batch job submission scripting for multi-run sweeps.",
        "tags": ["code_act", "inspect", "run"],
    },
    {
        "id": "investigate_run_subq",
        "agent": "code_act",
        "step": None,
        "text": "Investigate run/sub.q. This file is about HPC queue submission settings and environment setup.",
        "tags": ["code_act", "inspect", "run"],
    },
    {
        "id": "investigate_lib_gkvp_math_matrix",
        "agent": "code_act",
        "step": None,
        "text": "Investigate lib/gkvp_math_MATRIX.f90. This file is about mathematical functions using the MATRIX/MPP library.",
        "tags": ["code_act", "inspect", "lib"],
    },
    {
        "id": "investigate_lib_gkvp_math_mklnag",
        "agent": "code_act",
        "step": None,
        "text": "Investigate lib/gkvp_math_MKLNAG.f90. This file is about mathematical functions using MKL/NAG-style libraries.",
        "tags": ["code_act", "inspect", "lib"],
    },
    {
        "id": "investigate_lib_gkvp_math_ssl2",
        "agent": "code_act",
        "step": None,
        "text": "Investigate lib/gkvp_math_SSL2.f90. This file is about mathematical functions using the SSL2 library.",
        "tags": ["code_act", "inspect", "lib"],
    },
    {
        "id": "investigate_lib_gkvp_math_portable",
        "agent": "code_act",
        "step": None,
        "text": "Investigate lib/gkvp_math_portable.f90. This file is about portable mathematical helper functions.",
        "tags": ["code_act", "inspect", "lib"],
    },
    {
        "id": "answer_clear_from_agents",
        "agent": "answer",
        "step": None,
        "text": "From the agent output obtained in previous steps, give a clear final answer.",
        "tags": ["answer", "clarity"],
    },
]


def _normalize_pool_ids(pool: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(pool, start=1):
        new_entry = dict(entry)
        raw_id = entry.get("id")
        try:
            new_id = int(raw_id)
        except (TypeError, ValueError):
            new_id = idx
        new_entry["id"] = new_id
        normalized.append(new_entry)
    return normalized


def _load_knowledge_pool_csv(path: Path) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    if not path.exists():
        return pool
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("knowledge") or row.get("text") or "").strip()
                agent = str(row.get("agent") or "").strip()
                if not text or not agent:
                    continue
                tags_raw = row.get("tags")
                tags: List[str] = []
                if tags_raw not in (None, ""):
                    try:
                        parsed_tags = json.loads(tags_raw)
                        if isinstance(parsed_tags, list):
                            tags = [str(tag) for tag in parsed_tags]
                        elif isinstance(parsed_tags, str):
                            tags = [parsed_tags]
                    except json.JSONDecodeError:
                        tags = [tag.strip() for tag in str(tags_raw).split(",") if tag.strip()]
                step_value = row.get("step")
                step: Optional[int] = None
                if step_value not in (None, ""):
                    try:
                        step = int(step_value)
                    except (TypeError, ValueError):
                        step = None
                entry: Dict[str, Any] = {
                    "id": row.get("id"),
                    "agent": agent,
                    "text": text,
                    "tags": tags,
                }
                if step is not None:
                    entry["step"] = step
                pool.append(entry)
    except OSError as exc:
        print(f"[train_v6] Failed to load knowledge pool from {path}: {exc}")
        return []
    return pool


DEFAULT_KNOWLEDGE_POOL = _normalize_pool_ids(DEFAULT_KNOWLEDGE_POOL)
if DEFAULT_KLG_POOL_LOAD_PATH is not None:
    loaded_pool = _load_knowledge_pool_csv(Path(DEFAULT_KLG_POOL_LOAD_PATH))
    if loaded_pool:
        DEFAULT_KNOWLEDGE_POOL = _normalize_pool_ids(loaded_pool)
    else:
        print(
            "[train_v6] Using built-in knowledge pool; "
            f"failed to load from {DEFAULT_KLG_POOL_LOAD_PATH}"
        )

SCORING_SYSTEM_PROMPT = "Return only JSON."

KNOWLEDGE_SYSTEM_PROMPT = (
    "You provide concise, reusable advice (1-3 short lines) that nudges the agent back onto a reliable "
    "reasoning path without giving away the final answer."
)

KNOWLEDGE_SELECTION_SYSTEM_PROMPT = (
    "You rank reusable knowledge snippets from the pool. Respond only with JSON."
)

KNOWLEDGE_UPDATE_SYSTEM_PROMPT = (
    "You revise reusable knowledge snippets to improve future reasoning. Respond only with JSON."
)

RUN_ID_CACHE: Dict[str, str] = {}

# ---------- NEW: durable run cache + incremental saving ----------
CACHE_SCHEMA_VERSION = 2


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str):
        return getattr(self._streams[0], name)


def _setup_log_output(*, save_log: bool, log_dir: Optional[Path], prefix: str):
    should_log = bool(save_log) or not sys.stdout.isatty()
    if not should_log:
        return None
    resolved_dir = log_dir or (EXP_DIR / "_logs")
    resolved_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = resolved_dir / f"{prefix}_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)
    print(f"[{prefix}] Logging to {log_path}")
    return (log_file, original_stdout, original_stderr)


def _close_log_output(state) -> None:
    if not state:
        return
    log_file, original_stdout, original_stderr = state
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_json_size(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return 10**9


def _compact_msg_history(history: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Keep a bounded msg_history as a fallback if run logs are missing.
    Prefer not to bloat the results file.
    """
    if not isinstance(history, list) or not history:
        return None
    if len(history) > 160:
        return None
    # Bound total JSON size (rough)
    if _safe_json_size(history) > 250_000:
        return None
    compact: List[Dict[str, Any]] = []
    for msg in history:
        if isinstance(msg, dict):
            compact.append(dict(msg))
    return compact or None


@dataclass
class EvalCheckpoint:
    output_path: Path
    eval_entries: List[Dict[str, Any]]
    run_cache: Dict[str, Dict[str, Any]]
    lock: asyncio.Lock

    @staticmethod
    def make_run_key(entry_id: str, run_name: str) -> str:
        return f"{entry_id}::{run_name}"

    def get_cached(self, *, entry_id: str, run_name: str) -> Optional[Dict[str, Any]]:
        key = self.make_run_key(entry_id, run_name)
        cached = self.run_cache.get(key)
        if not isinstance(cached, dict):
            return None
        # Return a shallow copy so callers can safely mutate fields like run_id normalization.
        return dict(cached)

    async def record_run(
        self,
        *,
        entry_id: str,
        dataset_index: int,
        run_name: str,
        orchestrator_log_dir: Path,
        result: Dict[str, Any],
    ) -> None:
        """
        Persist minimal run outputs immediately after each orchestrator call.
        This enables resuming by skipping runs that already exist in run_cache.
        """
        key = self.make_run_key(entry_id, run_name)
        normalize_result_run_id(result, Path(orchestrator_log_dir))
        payload: Dict[str, Any] = {
            "entry_id": entry_id,
            "dataset_index": int(dataset_index),
            "run_name": run_name,
            "run_id": str(result.get("run_id") or "").strip(),
            "output": result.get("output", ""),
            "saved_at": _now_iso(),
        }
        msg_history = _compact_msg_history(result.get("msg_history"))
        if msg_history is not None:
            payload["msg_history"] = msg_history

        async with self.lock:
            # Avoid unnecessary rewrites if identical cached entry exists.
            prev = self.run_cache.get(key)
            if isinstance(prev, dict) and prev.get("run_id") == payload.get("run_id") and prev.get("output") == payload.get("output"):
                return
            self.run_cache[key] = payload
            # Must save after every run ends (user requirement).
            save_eval_entries(self.eval_entries, self.output_path, run_cache=self.run_cache)


class LLM_Request:
    def __init__(self, max_concurrent: int) -> None:
        self.max_concurrent = max(int(max_concurrent), 1)
        self.queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._closed = False

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self.max_concurrent)
        ]

    async def request(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._closed:
            raise RuntimeError("LLM_Request is closed.")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self.queue.put((func, args, kwargs, future))
        return await future

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            await self.queue.put(None)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

    async def _worker(self, _worker_id: int) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            func, args, kwargs, future = item
            if future.cancelled():
                self.queue.task_done()
                continue
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            else:
                if not future.done():
                    future.set_result(result)
            finally:
                self.queue.task_done()


async def _run_llm_request(
    llm_request: Optional[LLM_Request],
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if llm_request is None:
        return await func(*args, **kwargs)
    return await llm_request.request(func, *args, **kwargs)


def _sanitize_id_component(text: str, *, limit: int = 48) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_", " "}) else " " for ch in text)
    collapsed = "_".join(cleaned.split())
    return collapsed[:limit]


def extract_problem_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("problem") or entry.get("Question") or "").strip()


def extract_reference_answer(entry: Dict[str, Any]) -> str:
    return str(entry.get("answer") or entry.get("CorrectAnswer") or "").strip()


def extract_expected_difference(entry: Dict[str, Any]) -> str:
    return str(entry.get("expected_simulation_result_difference") or "").strip()


def build_problem_prompt(problem: str, expected_difference: str) -> str:
    lines = [
        "You are debugging the gyrokinetic plasma simulation workspace. "
        "Edit the local Fortran sources to resolve the regression described below.",
    ]
    if problem:
        lines.append(f"Problem statement:\n{problem}")

    '''
    if expected_difference:
        lines.append(
            "Expected simulation behavior after applying a correct fix:\n"
            f"{expected_difference}"
        )
    '''

    lines.append(
        "Apply minimal, precise code edits to resolve the issue. Reference the exact files and routines you touch, "
        "and explain why the fix restores the intended physics before concluding."
    )
    return "\n\n".join(line for line in lines if line.strip())


def build_task_prompt(entry: Dict[str, Any]) -> str:
    problem = extract_problem_text(entry)
    if problem:
        return build_problem_prompt(problem, extract_expected_difference(entry))
    csv_path = str(entry.get("CSVPath") or "").strip()
    question = str(entry.get("Question") or "").strip()
    if csv_path and question:
        return f"Analyze inside {csv_path} and answer the question below:\n\n{question}"
    return question or problem


def format_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


class PatchWorkspaceManager:
    def __init__(self, workspace: Path, patch_dir: Path) -> None:
        self.workspace = workspace.resolve()
        self.patch_dir = patch_dir.resolve()

    def resolve_patch_path(self, dataset_entry: Dict[str, Any], dataset_index: int) -> Path:
        custom_field = str(
            dataset_entry.get("patch_file")
            or dataset_entry.get("patch_path")
            or ""
        ).strip()
        if custom_field:
            candidate = self._coerce_patch_path(Path(custom_field))
            if candidate:
                return candidate
            raise FileNotFoundError(f"Could not find patch file '{custom_field}' for sample {dataset_index}")

        default_name = f"patch{dataset_index}.txt"
        default_path = (self.patch_dir / default_name).resolve()
        if default_path.exists():
            return default_path
        raise FileNotFoundError(
            f"No patch file found for dataset index {dataset_index} under {self.patch_dir}"
        )

    def _coerce_patch_path(self, candidate: Path) -> Optional[Path]:
        search_paths: List[Path] = []
        if candidate.is_absolute():
            search_paths.append(candidate)
        else:
            search_paths.append(self.patch_dir / candidate)
            search_paths.append(self.workspace / candidate)
            search_paths.append(candidate)
        for option in search_paths:
            resolved = option.resolve()
            if resolved.exists():
                return resolved
        return None

    @contextmanager
    def apply_for_problem(self, dataset_entry: Dict[str, Any], dataset_index: int):
        patch_path = self.resolve_patch_path(dataset_entry, dataset_index)
        patch_text = patch_path.read_text(encoding="utf-8")
        touched_paths = self._collect_touched_paths(patch_text)
        backups = self._snapshot_files(touched_paths)
        try:
            apply_patch_to_workspace(patch_text, self.workspace)
        except (PatchApplyError, PatchParseError) as exc:
            rel_name = self._relative_patch_name(patch_path)
            raise RuntimeError(f"Failed to apply patch {rel_name}: {exc}") from exc
        try:
            yield
        finally:
            self._restore_files(backups)

    def _relative_patch_name(self, patch_path: Path) -> str:
        try:
            return patch_path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(patch_path)

    def _collect_touched_paths(self, patch_text: str) -> Iterable[str]:
        markers = (
            "*** Update File:",
            "*** Add File:",
            "*** Delete File:",
            "*** Move to:",
        )
        seen: Dict[str, None] = {}
        for raw_line in patch_text.splitlines():
            stripped = raw_line.strip()
            for marker in markers:
                if stripped.startswith(marker):
                    rel_path = stripped[len(marker) :].strip()
                    if rel_path and rel_path not in seen:
                        seen[rel_path] = None
                    break
        return tuple(seen.keys())

    def _snapshot_files(self, rel_paths: Iterable[str]) -> Dict[Path, Optional[bytes]]:
        backups: Dict[Path, Optional[bytes]] = {}
        for rel_path in rel_paths:
            absolute = self._resolve_within_workspace(rel_path)
            if absolute in backups:
                continue
            backups[absolute] = absolute.read_bytes() if absolute.exists() else None
        return backups

    def _resolve_within_workspace(self, rel_path: str) -> Path:
        candidate = (self.workspace / Path(rel_path)).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:  # pragma: no cover - defensive
            raise AssertionError(f"Patch references a path outside the workspace: {rel_path}") from exc
        return candidate

    def _restore_files(self, backups: Dict[Path, Optional[bytes]]) -> None:
        for path, content in backups.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


@dataclass(frozen=True)
class WorkspaceHandle:
    workspace: Path
    patch_manager: PatchWorkspaceManager
    orchestrator: Any


def _build_orchestrator(args: argparse.Namespace, workspace: Path):
    prior_cwd = Path.cwd()
    os.chdir(workspace)
    try:
        return seimei(
            # agent_config=[{"file_path": "seimei/agents/code_act.py"}],
            agent_config=[{"name": "code_act"}, {"name": "think"}, {"name": "answer"}],
            llm_config={"base_url": args.llm_url, "model":args.llm_model_name},
            rm_config={"base_url": args.rm_url},
            allow_code_exec=True,
            agent_log_head_lines=1,
            max_tokens_per_question=80000,
        )
    finally:
        os.chdir(prior_cwd)


def _prepare_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    src_dest = workspace / "src"
    run_dest = workspace / "run"
    if src_dest.exists():
        shutil.rmtree(src_dest)
    if run_dest.exists():
        shutil.rmtree(run_dest)
    shutil.copytree(REPO_ROOT / "src", src_dest)
    shutil.copytree(REPO_ROOT / "run", run_dest)


def build_workspace_pool(args: argparse.Namespace, count: int) -> List[WorkspaceHandle]:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    handles: List[WorkspaceHandle] = []
    for idx in range(count):
        workspace = WORKSPACE_ROOT / f"ws_{idx:02d}"
        _prepare_workspace(workspace)
        patch_manager = PatchWorkspaceManager(workspace, PATCH_DIR)
        orchestrator = _build_orchestrator(args, workspace)
        handles.append(WorkspaceHandle(workspace=workspace, patch_manager=patch_manager, orchestrator=orchestrator))
    return handles


async def run_orchestrator_with_patch(
    orchestrator,
    patch_manager: PatchWorkspaceManager,
    *,
    dataset_entry: Dict[str, Any],
    dataset_index: int,
    messages: Sequence[Dict[str, Any]],
    run_name: str,
    knowledge_config: Dict[str, Any],
    knowledge_search_config: Optional[Sequence[Dict[str, Any]]] = None,
    knowledge_search_mode: Optional[str] = None,
    checkpoint: Optional[EvalCheckpoint] = None,
    entry_id: Optional[str] = None,
    llm_request: Optional[LLM_Request] = None,
) -> Dict[str, Any]:
    """
    NEW behavior:
    - If checkpoint has a cached run for (entry_id, run_name), return it and skip inference.
    - Otherwise, run inference, then immediately persist the result via save_eval_entries.
    """
    resolved_entry_id = entry_id or compute_entry_id(dataset_entry, dataset_index)
    if checkpoint is not None:
        cached = checkpoint.get_cached(entry_id=resolved_entry_id, run_name=run_name)
        if cached:
            # Ensure minimal keys exist for downstream code.
            cached.setdefault("run_id", cached.get("run_id") or "")
            cached.setdefault("output", cached.get("output") or "")
            if "msg_history" in cached:
                cached["msg_history"] = cached.get("msg_history")
            return cached

    knowledge_kwargs = split_knowledge_args(knowledge_config)
    if knowledge_search_config is not None:
        knowledge_kwargs["knowledge_search_config"] = knowledge_search_config
    if knowledge_search_mode:
        knowledge_kwargs["knowledge_search_mode"] = knowledge_search_mode
        
    knowledge_kwargs["agent_search_mode"] = "klg"

    try:
        with patch_manager.apply_for_problem(dataset_entry, dataset_index):
            result = await _run_llm_request(
                llm_request,
                orchestrator,
                messages=messages,
                run_name=run_name,
                **knowledge_kwargs,
                workspace=patch_manager.workspace,
            )

        if checkpoint is not None:
            await checkpoint.record_run(
                entry_id=resolved_entry_id,
                dataset_index=dataset_index,
                run_name=run_name,
                orchestrator_log_dir=Path(orchestrator.log_dir),
                result=result,
            )
        return result
    finally:
        _prepare_workspace(patch_manager.workspace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v6 by comparing no-knowledge vs knowledge-enabled trials."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the dataset JSON used for evaluation.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="Where to store evaluation traces and scores.",
    )
    parser.add_argument(
        "--llm-model-name",
        default=DEFAULT_LLM_MODEL_NAME,
        help="Name of model in LLM endpoint.",
    )
    parser.add_argument(
        "--llm-url",
        default=DEFAULT_LLM_URL,
        help="LLM endpoint passed to the orchestrator.",
    )
    parser.add_argument(
        "--rm-url",
        default=DEFAULT_RM_URL,
        help="Reward model search endpoint passed to the orchestrator.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of problems to process concurrently.",
    )
    parser.add_argument(
        "--n-no-klg-trials",
        type=int,
        default=DEFAULT_N_NO_KLG_TRIALS,
        help="How many trials to run without knowledge injection.",
    )
    parser.add_argument(
        "--n-klg-trials",
        type=int,
        default=DEFAULT_N_KLG_TRIALS,
        help="How many trials to run with knowledge injection.",
    )
    parser.add_argument(
        "--top-n-sample-klg",
        type=int,
        default=DEFAULT_TOP_N_SAMPLE_KLG,
        help="Top-N knowledge candidates to consider per step before sampling.",
    )
    parser.add_argument(
        "--distribution-decay-rate",
        type=float,
        default=DEFAULT_DISTRIBUTION_DECAY_RATE,
        help="Decay rate for the ranked sampling distribution.",
    )
    parser.add_argument(
        "--random-klg-sampling-rate",
        type=float,
        default=DEFAULT_RANDOM_KLG_SAMPLING_RATE,
        help="Chance of selecting a random knowledge entry from the top-N list.",
    )
    parser.add_argument(
        "--klg-sample-mode",
        choices=["llm", "rm"],
        default=DEFAULT_KLG_SAMPLE_MODE,
        help="Knowledge sampling mode: llm or rm.",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Optional cap on the number of dataset entries to evaluate.",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        dest="save_log",
        help="Write stdout/stderr to a log file (also enabled when stdout is not a TTY).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for log files when save-log is enabled.",
    )
    return parser.parse_args()


def compute_entry_id(dataset_entry: Dict[str, Any], index: int) -> str:
    explicit_keys = ["id", "ID", "question_id", "QuestionID", "problem_id", "ProblemID"]
    for key in explicit_keys:
        value = dataset_entry.get(key)
        if value:
            return str(value)
    patch_hint = str(dataset_entry.get("patch_file") or dataset_entry.get("patch_path") or "").strip()
    if patch_hint:
        return patch_hint
    csv_path = str(dataset_entry.get("CSVPath") or "").strip()
    if csv_path:
        return csv_path
    topic = str(dataset_entry.get("Topic") or "").strip()
    problem = extract_problem_text(dataset_entry)
    hyper = dataset_entry.get("HyperParamIndex")
    sample = dataset_entry.get("SampleIndex")
    parts = [f"idx_{index:05d}"]
    if topic:
        parts.append(_sanitize_id_component(topic))
    elif problem:
        parts.append(_sanitize_id_component(problem))
    if hyper not in (None, ""):
        parts.append(f"hp_{hyper}")
    if sample not in (None, ""):
        parts.append(f"sample_{sample}")
    return "|".join(parts)


def pick_system_prompt(*, use_knowledge_prompt: bool = False) -> str:
    pool = (
        KLG_SYSTEM_PROMPT_LIST
        if use_knowledge_prompt and KLG_SYSTEM_PROMPT_LIST
        else BASE_SYSTEM_PROMPT_LIST
    )
    return random.choice(pool)


def pick_base_system_prompt() -> str:
    return pick_system_prompt()


def randomize_system_prompt(
    messages: Sequence[Dict[str, Any]], *, use_knowledge_prompt: bool = False
) -> List[Dict[str, Any]]:
    cloned = [dict(msg) for msg in messages]
    prompt = pick_system_prompt(use_knowledge_prompt=use_knowledge_prompt)
    if cloned and str(cloned[0].get("role") or "").lower() == "system":
        cloned[0]["content"] = prompt
    else:
        cloned.insert(0, {"role": "system", "content": prompt})
    return cloned


def _strip_code_fences(text: str) -> str:
    snippet = text.strip()
    if snippet.startswith("```"):
        lines = snippet.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        snippet = "\n".join(lines).strip()
    return snippet


def _parse_json_response(raw: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(raw)
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def _parse_json_array(raw: str) -> List[Any]:
    cleaned = _strip_code_fences(raw)
    candidates = [cleaned]
    if "[" in cleaned and "]" in cleaned:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue
    return []


def _write_knowledge_pool_csv(
    path: Path,
    pool: Sequence[Dict[str, Any]],
    *,
    force: bool = True,
) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "agent", "knowledge", "tags", "step"])
        writer.writeheader()
        for entry in pool:
            text = str(entry.get("text") or "").strip()
            agent = str(entry.get("agent") or "").strip()
            if not text or not agent:
                continue
            tags = entry.get("tags") or []
            writer.writerow(
                {
                    "id": str(entry.get("id") or "").strip(),
                    "agent": agent,
                    "knowledge": text,
                    "tags": json.dumps(tags, ensure_ascii=True),
                    "step": entry.get("step") or "",
                }
            )


def _apply_knowledge_updates(
    pool: List[Dict[str, Any]],
    updates: Sequence[Dict[str, Any]],
) -> List[str]:
    updated_ids: List[str] = []
    id_map = {str(entry.get("id") or "").strip(): entry for entry in pool}
    for update in updates:
        if not isinstance(update, dict):
            continue
        update_id = str(update.get("id") or "").strip()
        new_text = str(update.get("text") or "").strip()
        if not update_id or not new_text:
            continue
        target = id_map.get(update_id)
        if not target:
            continue
        if target.get("text") == new_text:
            continue
        target["text"] = new_text
        updated_ids.append(update_id)
    return updated_ids


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0:
        return 0.0
    if score > 10:
        return 10.0
    return round(score, 2)


def _summarize_scores(scores: Sequence[float]) -> Tuple[float, float, float]:
    if not scores:
        return 0.0, 0.0, 0.0
    mean_score = round(sum(scores) / len(scores), 2)
    max_score = round(max(scores), 2)
    min_score = round(min(scores), 2)
    return mean_score, max_score, min_score


def _is_bugged_score(score: float, feedback: Optional[str]) -> bool:
    if score != 0.0:
        return False
    if feedback is None:
        return True
    if isinstance(feedback, str) and not feedback.strip():
        return True
    return False


def _resolve_run_dir_name(run_id: Optional[str], log_dir: Path) -> Optional[str]:
    run_id_str = str(run_id or "").strip()
    if not run_id_str:
        return None
    cached = RUN_ID_CACHE.get(run_id_str)
    if cached:
        return cached
    if not log_dir.exists():
        return run_id_str
    direct_candidate = log_dir / run_id_str
    if direct_candidate.exists():
        RUN_ID_CACHE[run_id_str] = run_id_str
        return run_id_str
    short = run_id_str[:8]
    if short:
        matches = sorted(log_dir.glob(f"run-*-{short}"))
        if matches:
            resolved = matches[-1].name
            RUN_ID_CACHE[run_id_str] = resolved
            return resolved
    for candidate in log_dir.glob("run-*"):
        meta_path = candidate / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta_run_id = str(meta_data.get("run_id") or "").strip()
        if meta_run_id == run_id_str:
            resolved = candidate.name
            RUN_ID_CACHE[run_id_str] = resolved
            return resolved
    return run_id_str


def normalize_result_run_id(result: Dict[str, Any], log_dir: Path) -> str:
    resolved = _resolve_run_dir_name(result.get("run_id"), Path(log_dir))
    if resolved:
        result["run_id"] = resolved
    return str(result.get("run_id") or "")


def extract_agent_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [msg for msg in messages if msg.get("role") == "agent"]


def get_agent_step_text(messages: Sequence[Dict[str, Any]], step: int) -> str:
    agent_msgs = extract_agent_messages(messages)
    if step < 1 or step > len(agent_msgs):
        return ""
    content = agent_msgs[step - 1].get("content", "")
    return str(content)


def get_agent_name_for_step(messages: Sequence[Dict[str, Any]], step: int) -> Optional[str]:
    agent_msgs = extract_agent_messages(messages)
    if step < 1 or step > len(agent_msgs):
        return None
    return agent_msgs[step - 1].get("name")


def _format_reasoning_history(
    messages: Sequence[Dict[str, Any]],
    knowledge_entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    agent_msgs = extract_agent_messages(messages)
    knowledge_by_step: Dict[int, List[str]] = {}
    for entry in knowledge_entries or []:
        if not isinstance(entry, dict):
            continue
        step_raw = entry.get("step")
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            continue
        text = str(entry.get("text") or "").strip()
        entry_id = str(entry.get("id") or "").strip()
        if not text:
            continue
        labeled = f"{entry_id}: {text}" if entry_id else text
        knowledge_by_step.setdefault(step, []).append(labeled)

    parts = ["<REASONING HISTORY>"]
    for idx, msg in enumerate(agent_msgs, start=1):
        agent_output = str(msg.get("content") or "")
        knowledge_lines = knowledge_by_step.get(idx, [])
        knowledge_text = "\n".join(knowledge_lines)
        parts.append(f"<STEP {idx}>")
        parts.append("<AGENT OUTPUT>")
        parts.append(agent_output)
        parts.append("</AGENT OUTPUT>")
        parts.append("<USED KNOWLEDGE>")
        parts.append(knowledge_text)
        parts.append("</USED KNOWLEDGE>")
        parts.append(f"</STEP {idx}>")
    parts.append("</REASONING HISTORY>")
    return "\n".join(parts)


def _build_scoring_prompt(
    *,
    question: str,
    patch: str,
    reference_answer: str,
    model_answer: str,
    reasoning_history: str,
) -> str:
    instruction1 = (
        "You are an impartial evaluator scoring an assistant's answer against a reference answer. "
        "Score is the sum of:\n"
        "- +2 for modifying the correct file.\n"
        "- +2 for modifying the same part as the correct one.\n"
        "- +3 for writing the same functional code as a deleted part.\n"
        "- +3 for how directly knowledge texts contribute to the reasoning steps "
        "(+1 if knowledge identifies the correct file, +1 if knowledge identifies the correct code snippet, "
        "+1 if knowledge contributes the entire reasoning improvement).\n"
        "Score must be an integer between 0 and 10."
    )
    instruction2 = "Return ONLY a JSON object with keys 'score' (integer) and 'feedback' (concise)."
    prompt = (
        f"{instruction1}\n\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n\n"
        f"<DELETED ORIGINAL CODE>\n{patch}\n</DELETED ORIGINAL CODE>\n\n"
        f"<REFERENCE ANSWER>\n{reference_answer}\n</REFERENCE ANSWER>\n\n"
        f"<MODEL ANSWER>\n{model_answer}\n</MODEL ANSWER>\n\n"
        f"{reasoning_history}\n\n"
        f"{instruction2}"
    )
    return prompt


def _build_knowledge_update_prompt(
    *,
    question: str,
    patch: str,
    reference_answer: str,
    reasoning_history: str,
) -> str:
    instruction1 = (
        "Revise the knowledge snippets used in the reasoning so future runs avoid the same mistakes. "
        "Strengthen the guidance, add missing context, and keep each snippet concise."
    )
    instruction2 = (
        "Return ONLY JSON with an 'updates' array. "
        "Each update must include the knowledge 'id' and the revised 'text'."
    )
    prompt = (
        f"{instruction1}\n\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n\n"
        f"<DELETED ORIGINAL CODE>\n{patch}\n</DELETED ORIGINAL CODE>\n\n"
        f"<REFERENCE ANSWER>\n{reference_answer}\n</REFERENCE ANSWER>\n\n"
        f"{reasoning_history}\n\n"
        f"{instruction2}"
    )
    return prompt


def build_knowledge_config(manual_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"generate_knowledge": False}
    if manual_entries:
        cfg["knowledge"] = manual_entries
    return cfg


def build_manual_knowledge_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for entry in DEFAULT_KNOWLEDGE_POOL:
        text = str(entry.get("text") or "").strip()
        agent = str(entry.get("agent") or "").strip()
        if not text or not agent:
            continue
        payload: Dict[str, Any] = {
            "id": entry.get("id"),
            "agent": agent,
            "text": text,
            "tags": entry.get("tags") or [],
        }
        step = entry.get("step")
        if step not in (None, ""):
            payload["step"] = step
        entries.append(payload)
    return entries


def build_knowledge_search_config(
    *,
    top_n_sample_klg: int,
    distribution_decay_rate: float,
    random_klg_sampling_rate: float,
    klg_sample_mode: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "mode": str(klg_sample_mode or "rm").strip().lower(),
            "step": ">=1",
            "topk": 1,
            "sampling_topk": max(int(top_n_sample_klg), 1),
            "sampling_distribution_decay_rate": float(distribution_decay_rate),
            "random_sampling_rate": float(random_klg_sampling_rate),
        }
    ]


def extract_knowledge_entries_from_messages(
    messages: Sequence[Dict[str, Any]],
    orchestrator: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    agent_messages = extract_agent_messages(messages)
    for step_idx, msg in enumerate(agent_messages, start=1):
        raw = msg.get("knowledge")
        items: List[Tuple[str, Any]] = []
        if orchestrator is not None and hasattr(orchestrator, "_coerce_knowledge_items_with_ids"):
            items = orchestrator._coerce_knowledge_items_with_ids(raw)
            if hasattr(orchestrator, "_dedupe_knowledge_items"):
                items = orchestrator._dedupe_knowledge_items(items)
        else:
            def _coerce_local(value: Any) -> List[Tuple[str, Any]]:
                local_items: List[Tuple[str, Any]] = []
                if value is None:
                    return local_items
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        local_items.append((text, None))
                    return local_items
                if isinstance(value, dict):
                    text_value = (
                        value.get("text")
                        or value.get("knowledge")
                        or value.get("content")
                        or value.get("value")
                    )
                    text = str(text_value or "").strip()
                    kid = value.get("id") or value.get("knowledge_id")
                    if text:
                        local_items.append((text, kid))
                    return local_items
                if isinstance(value, (list, tuple, set)):
                    for entry in value:
                        local_items.extend(_coerce_local(entry))
                    return local_items
                try:
                    text = str(value).strip()
                except Exception:
                    return local_items
                if text:
                    local_items.append((text, None))
                return local_items

            items = _coerce_local(raw)

        if not items:
            continue
        if msg.get("knowledge_id") not in (None, "") and all(kid is None for _, kid in items):
            kid_value = msg.get("knowledge_id")
            if isinstance(kid_value, (list, tuple, set)):
                ids_list = list(kid_value)
            else:
                ids_list = [kid_value]
            if len(ids_list) == len(items):
                items = [(text, ids_list[idx]) for idx, (text, _) in enumerate(items)]
            elif len(ids_list) == 1:
                items = [(text, ids_list[0]) for text, _ in items]

        for text, kid in items:
            entry = {"step": step_idx, "text": text}
            if kid is not None:
                entry["id"] = kid
            entries.append(entry)
    return entries


def split_knowledge_args(knowledge_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not knowledge_config:
        return {}
    load_config: List[Dict[str, Any]] = []
    load_path = knowledge_config.get("load_knowledge_path")
    load_steps = knowledge_config.get("load_knowledge_steps")
    if load_path:
        entry: Dict[str, Any] = {"load_knowledge_path": load_path}
        if load_steps:
            entry["step"] = load_steps
        load_config.append(entry)
    manual_entries = knowledge_config.get("knowledge")
    if manual_entries:
        if isinstance(manual_entries, list):
            for item in manual_entries:
                if isinstance(item, dict):
                    load_config.append(dict(item))
                elif isinstance(item, str):
                    load_config.append({"text": item})
        elif isinstance(manual_entries, dict):
            load_config.append(dict(manual_entries))
        elif isinstance(manual_entries, str):
            load_config.append({"text": manual_entries})
    generate_config = None
    if (
        knowledge_config.get("generate_knowledge")
        or knowledge_config.get("save_knowledge_path")
        or knowledge_config.get("knowledge_prompt_path")
    ):
        generate_config = {}
        save_path = knowledge_config.get("save_knowledge_path")
        if save_path:
            generate_config["save_knowledge_path"] = save_path
        prompt_path = knowledge_config.get("knowledge_prompt_path")
        if prompt_path:
            generate_config["knowledge_generation_prompt_path"] = prompt_path
    payload: Dict[str, Any] = {}
    if load_config:
        payload["knowledge_load_config"] = load_config
    if generate_config:
        payload["knowledge_generate_config"] = generate_config
    return payload


def get_messages_for_run(
    run_result: Dict[str, Any],
    log_dir: Path,
    *,
    max_agent_step: Optional[int] = None,
) -> List[Dict[str, Any]]:
    def _normalize(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        agent_seen = 0
        for msg in entries:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").lower()
            if role == "assistant":
                continue
            filtered.append(dict(msg))
            if role == "agent":
                agent_seen += 1
                if max_agent_step is not None and agent_seen >= max_agent_step:
                    break
        return filtered

    history = run_result.get("msg_history")
    if isinstance(history, list) and history:
        return _normalize(history)
    run_id = run_result.get("run_id")
    if not run_id:
        return []
    try:
        raw_messages = load_run_messages(run_id, runs_dir=log_dir, step=max_agent_step)
    except FileNotFoundError:
        # Fallback: if this run_result came from run_cache and included msg_history, _normalize above handles it.
        return []
    return _normalize(raw_messages)


async def score_answer(
    orchestrator_llm,
    question: str,
    reference_answer: str,
    model_answer: str,
    *,
    patch: str,
    messages: Sequence[Dict[str, Any]],
    knowledge_entries: Optional[Sequence[Dict[str, Any]]] = None,
    llm_request: Optional[LLM_Request] = None,
) -> Dict[str, Any]:
    reasoning_history = _format_reasoning_history(messages, knowledge_entries)
    prompt = _build_scoring_prompt(
        question=question,
        patch=patch,
        reference_answer=reference_answer,
        model_answer=model_answer,
        reasoning_history=reasoning_history,
    )
    try:
        response, usage = await _run_llm_request(
            llm_request,
            orchestrator_llm.chat,
            messages=[{"role": "user", "content": prompt}],
            system=SCORING_SYSTEM_PROMPT,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return {
            "score": 0.0,
            "feedback": f"Scoring failed: {exc}",
            "raw_response": "",
            "judge_usage": {},
        }

    parsed = _parse_json_response(response)
    score_value = _coerce_score(parsed.get("score"))
    feedback = parsed.get("feedback") or parsed.get("rationale") or ""

    return {
        "score": score_value,
        "feedback": feedback.strip(),
        "raw_response": response,
        "judge_usage": usage,
    }


def _parse_knowledge_updates(raw: str) -> List[Dict[str, Any]]:
    parsed = _parse_json_response(raw)
    updates = parsed.get("updates")
    if isinstance(updates, list):
        return [dict(item) for item in updates if isinstance(item, dict)]
    array = _parse_json_array(raw)
    return [dict(item) for item in array if isinstance(item, dict)]


async def update_knowledge_after_scoring(
    orchestrator_llm,
    *,
    question: str,
    reference_answer: str,
    patch: str,
    messages: Sequence[Dict[str, Any]],
    knowledge_entries: Sequence[Dict[str, Any]],
    update_lock: asyncio.Lock,
    llm_request: Optional[LLM_Request] = None,
) -> List[str]:
    reasoning_history = _format_reasoning_history(messages, knowledge_entries)
    prompt = _build_knowledge_update_prompt(
        question=question,
        patch=patch,
        reference_answer=reference_answer,
        reasoning_history=reasoning_history,
    )
    try:
        response, _ = await _run_llm_request(
            llm_request,
            orchestrator_llm.chat,
            messages=[{"role": "user", "content": prompt}],
            system=KNOWLEDGE_UPDATE_SYSTEM_PROMPT,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[knowledge update] failed: {exc}")
        return []

    updates = _parse_knowledge_updates(response)
    if not updates:
        return []

    async with update_lock:
        updated_ids = _apply_knowledge_updates(DEFAULT_KNOWLEDGE_POOL, updates)
        if updated_ids:
            _write_knowledge_pool_csv(DEFAULT_FINAL_KLG_POOL_SAVE_PATH, DEFAULT_KNOWLEDGE_POOL)
    return updated_ids


async def run_problem(
    orchestrator,
    patch_manager: PatchWorkspaceManager,
    dataset_entry: Dict[str, Any],
    index: int,
    entry_id: str,
    *,
    n_no_klg_trials: int,
    n_klg_trials: int,
    top_n_sample_klg: int,
    klg_sample_mode: str,
    distribution_decay_rate: float,
    random_klg_sampling_rate: float,
    checkpoint: Optional[EvalCheckpoint],
    update_lock: asyncio.Lock,
    llm_request: Optional[LLM_Request] = None,
) -> Dict[str, Any]:
    problem_text = extract_problem_text(dataset_entry)
    prompt_text = build_task_prompt(dataset_entry) or problem_text
    if not prompt_text:
        prompt_text = "Investigate the regression and repair the affected code paths."
    question = build_task_prompt(dataset_entry) or prompt_text
    reference_answer = extract_reference_answer(dataset_entry)
    csv_path = str(dataset_entry.get("CSVPath") or "").strip()
    try:
        patch_path = patch_manager.resolve_patch_path(dataset_entry, index)
    except FileNotFoundError as exc:
        raise RuntimeError(f"[sample {index}] Missing patch file: {exc}") from exc
    patch_text = patch_path.read_text(encoding="utf-8")
    n_no_klg_trials = max(int(n_no_klg_trials), 0)
    n_klg_trials = max(int(n_klg_trials), 0)
    top_n_sample_klg = max(int(top_n_sample_klg), 1)

    base_prompt_messages = [{"role": "user", "content": prompt_text}]
    log_dir = Path(orchestrator.log_dir)
    knowledge_search_config = build_knowledge_search_config(
        top_n_sample_klg=top_n_sample_klg,
        distribution_decay_rate=distribution_decay_rate,
        random_klg_sampling_rate=random_klg_sampling_rate,
        klg_sample_mode=klg_sample_mode,
    )

    async def _run_trials(
        *,
        trials: int,
        label: str,
        use_knowledge: bool,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        if trials <= 0:
            return [], []
        trial_records: List[Dict[str, Any]] = []
        valid_scores: List[float] = []
        for trial in range(trials):
            rerun_messages = randomize_system_prompt(
                base_prompt_messages,
                use_knowledge_prompt=use_knowledge,
            )
            #manual_entries = build_manual_knowledge_entries() if use_knowledge else None
            #knowledge_config = build_knowledge_config(
            #    [dict(entry) for entry in manual_entries] if manual_entries else None
            #)
            knowledge_config = {
                "load_knowledge_path": DEFAULT_KLG_POOL_LOAD_PATH,
            }
            run_name = f"train_v6_{index:04d}_{label}_r{trial + 1}"
            result = await run_orchestrator_with_patch(
                orchestrator,
                patch_manager,
                dataset_entry=dataset_entry,
                dataset_index=index,
                messages=rerun_messages,
                run_name=run_name,
                knowledge_config=knowledge_config if use_knowledge else None,
                knowledge_search_config=knowledge_search_config if use_knowledge else None,
                knowledge_search_mode=klg_sample_mode if use_knowledge else None,
                checkpoint=checkpoint,
                entry_id=entry_id,
                llm_request=llm_request,
            )
            run_id = normalize_result_run_id(result, log_dir)
            output = result.get("output", "")
            messages = get_messages_for_run(result, log_dir)
            used_knowledge = (
                extract_knowledge_entries_from_messages(messages, orchestrator)
                if use_knowledge
                else []
            )
            score_info = await score_answer(
                orchestrator.llm,
                question,
                reference_answer,
                output,
                patch=patch_text,
                messages=messages,
                knowledge_entries=used_knowledge,
                llm_request=llm_request,
            )
            score = score_info.get("score", 0.0) or 0.0
            feedback = score_info.get("feedback")
            if not _is_bugged_score(score, feedback):
                valid_scores.append(score)
            if use_knowledge and DEFAULT_ENABLE_UPDATE_KLG_POOL:
                await update_knowledge_after_scoring(
                    orchestrator.llm,
                    question=question,
                    reference_answer=reference_answer,
                    patch=patch_text,
                    messages=messages,
                    knowledge_entries=used_knowledge,
                    update_lock=update_lock,
                    llm_request=llm_request,
                )
            record = {
                "trial": trial + 1,
                "run_name": run_name,
                "run_id": run_id,
                "score": score,
                "score_feedback": feedback,
                "output": output,
            }
            if use_knowledge:
                record["knowledge_used"] = used_knowledge
            trial_records.append(record)
        return trial_records, valid_scores

    no_klg_trials, no_klg_valid_scores = await _run_trials(
        trials=n_no_klg_trials,
        label="no_klg",
        use_knowledge=False,
    )
    klg_trials, klg_valid_scores = await _run_trials(
        trials=n_klg_trials,
        label="klg",
        use_knowledge=True,
    )

    no_klg_mean, no_klg_max, no_klg_min = _summarize_scores(no_klg_valid_scores)
    klg_mean, klg_max, klg_min = _summarize_scores(klg_valid_scores)

    record: Dict[str, Any] = {
        "id": entry_id,
        "problem": problem_text,
        "csv_path": csv_path,
        "patch_path": format_relative_path(patch_path),
        "no_klg_trials": no_klg_trials,
        "klg_trials": klg_trials,
        "no_klg_mean_score": no_klg_mean,
        "klg_mean_score": klg_mean,
        "no_klg_max_score": no_klg_max,
        "klg_max_score": klg_max,
        "no_klg_min_score": no_klg_min,
        "klg_min_score": klg_min,
        "mean_score_improvement": round(klg_mean - no_klg_mean, 2),
        "max_score_improvement": round(klg_max - no_klg_max, 2),
        "min_score_improvement": round(klg_min - no_klg_min, 2),
    }

    print(
        f"[sample {index}] no_klg mean={no_klg_mean} klg mean={klg_mean} "
        f"(Δ {round(klg_mean - no_klg_mean, 2)})"
    )

    return record


def load_existing_eval_entries(
    dataset: List[Dict[str, Any]],
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], Set[str], bool, Dict[str, Dict[str, Any]]]:
    entries: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()
    needs_resave = False
    run_cache: Dict[str, Dict[str, Any]] = {}
    if output_path.exists():
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                entries = raw.get("detail") or []
                rc = raw.get("run_cache") or {}
                if isinstance(rc, dict):
                    # ensure values are dicts
                    run_cache = {str(k): v for k, v in rc.items() if isinstance(v, dict)}
            elif isinstance(raw, list):
                entries = raw
                needs_resave = True
            else:
                print(f"[train_v6] Ignoring malformed cache at {output_path}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[train_v6] Failed to load cached eval entries: {exc}")
        for idx, entry in enumerate(entries):
            entry_id = str(entry.get("id") or "").strip()
            if not entry_id and idx < len(dataset):
                entry_id = compute_entry_id(dataset[idx], idx)
                entry["id"] = entry_id
                needs_resave = True
            if entry_id:
                processed_ids.add(entry_id)
    return entries, processed_ids, needs_resave, run_cache


def build_eval_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    empty_summary = {
        "total_problems": 0,
        "overall_mean_score_improvement": 0.0,
        "max_mean_score_improvement": 0.0,
        "min_mean_score_improvement": 0.0,
        "no_klg_overall_mean": 0.0,
        "klg_overall_mean": 0.0,
        "no_klg_max_mean": 0.0,
        "klg_max_mean": 0.0,
        "no_klg_min_mean": 0.0,
        "klg_min_mean": 0.0,
        "mean_win_loss_tie": {"win": 0, "tie": 0, "loss": 0},
        "max_mean_win_loss_tie": {"win": 0, "tie": 0, "loss": 0},
        "min_mean_win_loss_tie": {"win": 0, "tie": 0, "loss": 0},
        "mean_no_klg_vs_mean_klg": [],
        "max_no_klg_vs_max_klg": [],
        "min_no_klg_vs_min_klg": [],
    }
    if not entries:
        return empty_summary

    mean_pairs: List[List[float]] = []
    max_pairs: List[List[float]] = []
    min_pairs: List[List[float]] = []
    mean_improvements: List[float] = []
    max_improvements: List[float] = []
    min_improvements: List[float] = []

    for entry in entries:
        no_mean = float(entry.get("no_klg_mean_score") or 0.0)
        klg_mean = float(entry.get("klg_mean_score") or 0.0)
        no_max = float(entry.get("no_klg_max_score") or 0.0)
        klg_max = float(entry.get("klg_max_score") or 0.0)
        no_min = float(entry.get("no_klg_min_score") or 0.0)
        klg_min = float(entry.get("klg_min_score") or 0.0)

        mean_pairs.append([round(no_mean, 2), round(klg_mean, 2)])
        max_pairs.append([round(no_max, 2), round(klg_max, 2)])
        min_pairs.append([round(no_min, 2), round(klg_min, 2)])
        mean_improvements.append(klg_mean - no_mean)
        max_improvements.append(klg_max - no_max)
        min_improvements.append(klg_min - no_min)

    total = len(entries)

    def _count_wlt(pairs: Sequence[Sequence[float]]) -> Dict[str, int]:
        wins = sum(1 for no, klg in pairs if klg > no)
        ties = sum(1 for no, klg in pairs if klg == no)
        losses = total - wins - ties
        return {"win": wins, "tie": ties, "loss": losses}

    no_klg_overall_mean = round(sum(pair[0] for pair in mean_pairs) / total, 2)
    klg_overall_mean = round(sum(pair[1] for pair in mean_pairs) / total, 2)
    no_klg_max_mean = round(sum(pair[0] for pair in max_pairs) / total, 2)
    klg_max_mean = round(sum(pair[1] for pair in max_pairs) / total, 2)
    no_klg_min_mean = round(sum(pair[0] for pair in min_pairs) / total, 2)
    klg_min_mean = round(sum(pair[1] for pair in min_pairs) / total, 2)

    return {
        "total_problems": total,
        "overall_mean_score_improvement": round(sum(mean_improvements) / total, 2),
        "max_mean_score_improvement": round(sum(max_improvements) / total, 2),
        "min_mean_score_improvement": round(sum(min_improvements) / total, 2),
        "no_klg_overall_mean": no_klg_overall_mean,
        "klg_overall_mean": klg_overall_mean,
        "no_klg_max_mean": no_klg_max_mean,
        "klg_max_mean": klg_max_mean,
        "no_klg_min_mean": no_klg_min_mean,
        "klg_min_mean": klg_min_mean,
        "mean_win_loss_tie": _count_wlt(mean_pairs),
        "max_mean_win_loss_tie": _count_wlt(max_pairs),
        "min_mean_win_loss_tie": _count_wlt(min_pairs),
        "mean_no_klg_vs_mean_klg": mean_pairs,
        "max_no_klg_vs_max_klg": max_pairs,
        "min_no_klg_vs_min_klg": min_pairs,
    }


def save_eval_entries(
    entries: List[Dict[str, Any]],
    output_path: Path,
    *,
    run_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if os.getenv("SEIMEI_DEBUG_EVAL_SAVE"):
        print(
            "[train_v6] save_eval_entries: "
            f"{len(entries)} entries -> {output_path}"
        )
    payload_obj: Dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "saved_at": _now_iso(),
        "summary": build_eval_summary(entries),
        "detail": entries,
    }
    if run_cache is not None:
        payload_obj["run_cache"] = run_cache
        payload_obj["run_cache_count"] = len(run_cache)

    payload = json.dumps(payload_obj, ensure_ascii=False, indent=2)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(output_path)


async def run_evaluation(args: argparse.Namespace) -> None:
    log_state = _setup_log_output(
        save_log=args.save_log,
        log_dir=args.log_dir,
        prefix="train_v6",
    )
    try:
        dataset = json.loads(args.dataset_path.read_text(encoding="utf-8"))
        print(f"[train_v6] Output path: {args.output_path}")
        patch_files = sorted(PATCH_DIR.glob("*.txt"))
        if not patch_files:
            raise RuntimeError(f"No patch files found under {PATCH_DIR}")
        workspace_count = min(args.batch_size, len(patch_files))
        if workspace_count < 1:
            raise RuntimeError("batch_size must be >= 1 and patch_files must be non-empty.")
        if args.batch_size != workspace_count:
            print(
                "[train_v6] Limiting concurrency to "
                f"{workspace_count} based on patch file count."
            )
        workspace_pool = build_workspace_pool(args, workspace_count)

        eval_entries, processed_ids, needs_resave, run_cache = load_existing_eval_entries(
            dataset, args.output_path
        )
        print(
            f"[train_v6] Loaded {len(eval_entries)} eval entries from {args.output_path}"
        )
        checkpoint = EvalCheckpoint(
            output_path=args.output_path,
            eval_entries=eval_entries,
            run_cache=run_cache,
            lock=asyncio.Lock(),
        )

        if needs_resave:
            save_eval_entries(eval_entries, args.output_path, run_cache=run_cache)
        if processed_ids:
            print(f"[train_v6] Resuming with {len(processed_ids)} cached entries")
        if run_cache:
            print(f"[train_v6] Loaded run_cache with {len(run_cache)} runs")

        pending: List[Tuple[int, Dict[str, Any], str]] = []
        for idx, entry in enumerate(dataset):
            entry_id = compute_entry_id(entry, idx)
            if entry_id in processed_ids:
                continue
            pending.append((idx, entry, entry_id))
            if args.max_problems is not None and len(pending) >= args.max_problems:
                break

        if not pending:
            print(
                f"[train_v6] All {len(processed_ids)} entries already processed. "
                f"Results stored at {args.output_path}"
            )
            _write_knowledge_pool_csv(DEFAULT_FINAL_KLG_POOL_SAVE_PATH, DEFAULT_KNOWLEDGE_POOL)
            return

        total_pending = len(pending)
        print(
            f"[train_v6] Starting {total_pending} problems with "
            f"concurrency={workspace_count} (dataset size={len(dataset)})"
        )
        completed_in_run = 0

        knowledge_update_lock = asyncio.Lock()
        llm_request = LLM_Request(args.batch_size)
        await llm_request.start()
        try:
            for batch_start in range(0, len(pending), workspace_count):
                batch_slice = pending[batch_start : batch_start + workspace_count]
                batch_idx = batch_start // args.batch_size + 1
                print(
                    f"[train_v6] Starting batch {batch_idx} "
                    f"with {len(batch_slice)} problems"
                )
                batch_tasks: List[asyncio.Task] = []
                task_ids: Dict[asyncio.Task, str] = {}
                for handle, (dataset_idx, entry, entry_id) in zip(workspace_pool, batch_slice):
                    task = asyncio.create_task(
                        run_problem(
                            handle.orchestrator,
                            handle.patch_manager,
                            entry,
                            dataset_idx,
                            entry_id,
                            n_no_klg_trials=args.n_no_klg_trials,
                            n_klg_trials=args.n_klg_trials,
                            top_n_sample_klg=args.top_n_sample_klg,
                            klg_sample_mode=args.klg_sample_mode,
                            distribution_decay_rate=args.distribution_decay_rate,
                            random_klg_sampling_rate=args.random_klg_sampling_rate,
                            checkpoint=checkpoint,
                            update_lock=knowledge_update_lock,
                            llm_request=llm_request,
                        )
                    )
                    batch_tasks.append(task)
                    task_ids[task] = entry_id

                for task in asyncio.as_completed(batch_tasks):
                    entry_id = task_ids.get(task, "unknown")
                    record = None
                    try:
                        record = await task
                    except Exception as exc:  # pragma: no cover - runtime guard
                        print(f"[train_v6] Problem {entry_id} failed: {exc}")
                    if record:
                        rid = str(record.get("id") or "").strip()
                        if rid and rid not in processed_ids:
                            eval_entries.append(record)
                            processed_ids.add(rid)
                            print(
                                "[train_v6] Stored record "
                                f"{rid}; total entries={len(eval_entries)}"
                            )
                    async with checkpoint.lock:
                        save_eval_entries(
                            eval_entries,
                            args.output_path,
                            run_cache=checkpoint.run_cache,
                        )
                    completed_in_run += 1
                    percent = (completed_in_run / total_pending) * 100 if total_pending else 100.0
                    print(
                        "[train_v6] Progress: "
                        f"{completed_in_run}/{total_pending} ({percent:.1f}%) problems finished; "
                        f"overall {len(processed_ids)}/{len(dataset)} stored"
                    )

                # Keep batch-level saves as a safety net (run_cache is persisted per run).
                async with checkpoint.lock:
                    save_eval_entries(eval_entries, args.output_path, run_cache=checkpoint.run_cache)
                print(
                    f"[train_v6] Saved batch {batch_idx}: "
                    f"{len(processed_ids)}/{len(dataset)} problems complete"
                )
        finally:
            await llm_request.close()

        _write_knowledge_pool_csv(DEFAULT_FINAL_KLG_POOL_SAVE_PATH, DEFAULT_KNOWLEDGE_POOL)
        print(f"[train_v6] Saved evaluation dataset to {args.output_path}")
    finally:
        _close_log_output(log_state)


if __name__ == "__main__":
    asyncio.run(run_evaluation(parse_args()))
