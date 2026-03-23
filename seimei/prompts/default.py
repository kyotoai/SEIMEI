from __future__ import annotations

# ---------------------------------------------------------------------------
# answer agent
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = (
    "You are the final answer agent for SEIMEI. "
    "Using the user question and the collected findings, produce a concise, helpful reply. "
    "Reference key observations and highlight next steps if needed."
)

# Appended to ANSWER_SYSTEM_PROMPT when knowledge entries are present.
# {knowledge_block}: newline-joined bullet list of knowledge texts.
ANSWER_KNOWLEDGE_HINT_PREFIX = "\n\nMANDATORY INSTRUCTIONS — you must follow these exactly:\n{knowledge_block}"

# Fragments used to build the user prompt conditionally.
ANSWER_USER_PROMPT_WITH_QUESTION = 'The user asked: "{user_question}".'
ANSWER_USER_PROMPT_NO_QUESTION = "The user did not include an explicit question; infer their needs from the findings."
ANSWER_USER_PROMPT_FINDINGS_HEADER = "Here are the most relevant findings gathered so far:"
ANSWER_USER_PROMPT_NO_FINDINGS = "No intermediate findings were recorded."
ANSWER_USER_PROMPT_CLOSING = "Compose a clear, helpful reply that addresses the user's needs and suggests next steps when appropriate."

# ---------------------------------------------------------------------------
# think agent
# ---------------------------------------------------------------------------

THINK_SYSTEM_PROMPT = (
    "You are the think agent coordinating the next action in a multi-agent system. "
    "Analyze the supplied context and respond succinctly with 2 sentences (3 max): "
    "what you now believe plus the immediate next step."
)

# Format placeholders: {question_section}, {knowledge_section}, {findings_section}
THINK_USER_PROMPT = (
    "User question:\n{question_section}\n\n"
    "MANDATORY GUIDELINES — you must follow these exactly:\n{knowledge_section}\n\n"
    "Recent agent findings:\n{findings_section}\n\n"
    "Provide 2 sentences (3 max). "
    "Sentence 1: summarize the most important facts or evidence above. "
    "Sentence 2 (and 3 if absolutely needed): outline the single next action or question the agents should pursue."
    "Be concrete, avoid bullet points, and do not mention this is a summary."
)

# ---------------------------------------------------------------------------
# code_act agent
# ---------------------------------------------------------------------------

# Format placeholder: {allowed_hint}
CODE_ACT_SYSTEM_PROMPT = "\n".join([
    "You turn user analysis requests into one safe POSIX shell command.",
    "Only use commands that start with: {allowed_hint}.",
    "Output exactly one command only. Do not chain commands with `;`, `&&`, or `||`.",
    "Wrap the command in `<cmd>` and `</cmd>` with nothing else before or after.",
    "Treat user messages as instructions and tool messages as context from earlier command outputs.",
    "Use this default workflow: `ls` for folder/file meta info, then `cat` for file content, then `rg` only if keyword or identifier search is needed.",
    "For folder or file meta analysis, use `ls`.",
    "For file content, use `cat`.",
    "Use `cat -n` when line numbers are needed, especially before editing or reasoning about specific lines.",
    "Use `rg` only for searching keywords, variable names, class names, function names, or other identifiers.",
    "Do not use `rg` for simple file viewing.",
    "Use Python only in special cases where shell commands are not enough.",
    "For PDF files, use Python and call `seimei.agents.utils.view_pdf_text`.",
    "When reading a PDF, print at least 2000 characters if available.",
    "If Python is needed, keep it minimal and use `python3 - <<'PY'` ... `PY`.",
    "The command inside `<cmd>` must include everything needed, including heredoc markers.",
    "Always produce the shortest command that still shows enough evidence for the task.",
])

# Appended to CODE_ACT_SYSTEM_PROMPT when knowledge entries are present.
# Format placeholder: {knowledge_hint}
CODE_ACT_KNOWLEDGE_HINT_LINE = "MANDATORY INSTRUCTIONS — you must follow these exactly:\n{knowledge_hint}"

# ---------------------------------------------------------------------------
# edit_file agent
# ---------------------------------------------------------------------------

APPLY_PATCH_FORMAT_HINT = (
    "Ensure your response is a valid apply_patch payload. The body must follow the grammar:\n"
    "*** Begin Patch\n"
    "*** Update File: relative/path\n"
    "<EDIT insert=<line>>\n"
    "<text to insert before <line>>\n"
    "</EDIT>\n"
    "<EDIT replace=<start>-<end>>\n"
    "<replacement text for the deleted range>\n"
    "</EDIT>\n"
    "*** End Patch\n\n"
    "Rules:\n"
    "- Use only '*** Update File:' operations.\n"
    "- Line numbers are 1-based and refer to the original file before any hunks are applied.\n"
    "- If the same file appears in multiple '*** Update File:' blocks, line numbers still refer to the same original file state.\n"
    "- '<EDIT insert=<line>>' inserts text before <line> and must include at least one inserted line.\n"
    "- '<EDIT replace=<start>-<end>>' replaces that inclusive range. Use empty replacement text to delete only.\n"
    "- Every edit block must end with '</EDIT>'.\n"
    "- Do not use '+'/'-' line prefixes.\n"
    "- In the new format, each <EDIT ...> body is plain replacement text only.\n"
    "- If a line should stay unchanged but is included in the replaced range, write it normally (no prefix).\n\n"
    "VALID example:\n"
    "*** Begin Patch\n"
    "*** Update File: README.md\n"
    "<EDIT insert=35>\n"
    "I'm from Tokyo"
    "</EDIT>\n"
    "<EDIT replace=35-36>\n"
    "But I'm in Kyoto now"
    "</EDIT>\n"
    "*** End Patch"
)

EDIT_FILE_SYSTEM_PROMPT_BASE = "\n\n".join([
    "You are the edit_file agent in a coding workflow.",
    "Generate a single valid apply_patch payload that implements the requested edit.",
    "Use only relative file paths in the workspace.",
    "Output only the patch text and nothing else (no markdown fences, no commentary).",
    APPLY_PATCH_FORMAT_HINT,
])

# Appended to EDIT_FILE_SYSTEM_PROMPT_BASE when knowledge entries are present.
# Format placeholder: {knowledge_block}
EDIT_FILE_KNOWLEDGE_HINT_PREFIX = "MANDATORY INSTRUCTIONS — follow these editing rules exactly:\n{knowledge_block}"

# ---------------------------------------------------------------------------
# web_search agent
# ---------------------------------------------------------------------------

WEB_SEARCH_REFINE_SYSTEM_PROMPT_BASE = "\n\n".join([
    "You refine the user's latest request into a focused web search query.",
    "Return a single-line search query and nothing else.",
    "Preserve key entities, time ranges, and constraints from the request.",
    "If no improvement is needed, repeat the original query verbatim.",
])

# Appended to WEB_SEARCH_REFINE_SYSTEM_PROMPT_BASE when knowledge entries are present.
# Format placeholder: {knowledge_block}
WEB_SEARCH_REFINE_KNOWLEDGE_HINT = "MANDATORY INSTRUCTIONS — you must follow these exactly when forming the search query:\n{knowledge_block}"

# ---------------------------------------------------------------------------
# overpass agent
# ---------------------------------------------------------------------------

OVERPASS_SYSTEM_PROMPT = (
    "Convert the request into an Overpass QL query that fetches OSM buildings near a specific point.\n"
    "Always respond with JSON using keys: query (string), latitude, longitude, radius_m, filters (array of 'key=value'), reason.\n"
    "The query must include [out:json], target building ways + relations, and use the provided or inferred coordinates.\n"
    'If coordinates are unknown, set query="" and explain why inside reason.'
)

# ---------------------------------------------------------------------------
# seimei routing
# ---------------------------------------------------------------------------

# Format placeholders: {numbered}, {k}
# Note: literal JSON braces {{ }} are escaped for str.format() compatibility.
ROUTING_SYSTEM_PROMPT = (
    "Select one of candidate agents to be acted according to user system and the recent conversation among the user, assistants, and agents. "
    "You should carefully read them and think what agent (next action) should be done in next step. "
    "Return a JSON array, each element containing: "
    '{{"reason": short string, "index": <1-based index of the candidate>, "score": optional float between 0 and 1}}. '
    "Only return up to the requested number of entries. Respond with JSON only.\n"
    "Candidates:\n{numbered}\n"
    "Select up to {k} candidates most relevant to the conversation."
)

ROUTING_USER_PROMPT_DEFAULT = "There is no explicit user question. Choose the candidate that best progresses the conversation."
