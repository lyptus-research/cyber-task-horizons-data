"""Extract and compare tool use between human experts and AI models.

Human experts interact via terminal sessions recorded as script(1) typescript
files. The `.output` log captures everything printed to the terminal, including
shell prompts that delimit executed commands. We extract commands by matching
prompt patterns and replaying terminal editing characters (backspaces, ANSI
insert sequences) to recover the final command string.

Model agents make structured tool calls (bash, python) recorded in Inspect AI
.eval files. Each tool call is a clean, atomic action with exact command text.

The comparable unit is "shell commands executed" — things that hit enter and ran
in a shell. Human typos, failed commands, and exploratory `ls` all count, just
as a model's failed curl or exploratory ls counts. The only difference is that
human recordings are noisy (editing chars, prompt re-draws) and require replay
to recover the actual command.

Usage:
    from lib.tool_use import (
        extract_human_commands,
        extract_model_tool_calls,
        load_human_eval_sample,
    )

    sample = load_human_eval_sample("/path/to/file.eval.gz")
    cmds = extract_human_commands(sample)
    # cmds = ["ls -la", "cat README.md", "curl http://victim:8080", ...]
"""

from __future__ import annotations

import gzip
import io
import json
import re
import zipfile
from pathlib import Path


# =============================================================================
# Terminal replay
# =============================================================================


def _replay_edits(raw: str) -> str:
    """Replay terminal editing characters to recover the final command string.

    Handles:
        - \\x08 (backspace): delete previous character
        - \\x07 (bell): ignored (tab completion sound)
        - \\x1b[n@ (ANSI CSI insert): insert the next n characters at cursor
        - \\r (carriage return): ignored (prompt re-draws)
        - Control characters < 0x20: stripped (except tab)
    """
    buf: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c == "\x08":  # backspace
            if buf:
                buf.pop()
        elif c == "\x07":  # bell
            pass
        elif c == "\x1b" and i + 1 < n and raw[i + 1] == "[":
            # ANSI CSI sequence: \x1b[ ... <final_byte>
            j = i + 2
            while j < n and raw[j] not in "ABCDEFGHJKSTfmnsu@":
                j += 1
            if j < n and raw[j] == "@":
                # Insert mode: \x1b[1@X means insert char X
                seq = raw[i + 2 : j]
                try:
                    count = int(seq) if seq else 1
                except ValueError:
                    count = 1
                for k in range(count):
                    if j + 1 + k < n:
                        buf.append(raw[j + 1 + k])
                i = j + 1 + count
                continue
            i = j + 1
            continue
        elif c == "\r":
            pass
        elif ord(c) >= 32 or c == "\t":
            buf.append(c)
        i += 1
    return "".join(buf).strip()


# Prompt patterns for different sandbox environments.
# Kali-style: └─# or └─$
# Standard root: root@hostname:/path#
# NYUCTF ctfplayer: ctfplayer@hostname:path$
_PROMPT_RE = re.compile(
    r"(?:"
    r"└─[#$]"  # Kali
    r"|root@[^:]+:[^#$]*[#$]"  # root@host
    r"|ctfplayer@[^:]+:[^$]*\$"  # NYUCTF ctfplayer
    r"|[a-z_]\w*@[a-z0-9_-]+:[^#$]*[#$]"  # generic user@host
    r")\s*(.*)"
)


def extract_human_commands(sample: dict) -> list[str] | None:
    """Extract shell commands from a human eval sample's terminal recording.

    Looks for the `.output` log in HumanAgentState:logs, matches shell prompt
    patterns, and replays editing characters to recover clean command strings.

    Returns None if no `.output` log is available (some older sessions only
    recorded `.input` and `.timing`).
    """
    logs = sample.get("store", {}).get("HumanAgentState:logs", {})
    output_key = next((k for k in logs if ".output" in k), None)
    if output_key is None:
        return None

    raw_output = logs[output_key]

    # Strip ANSI color codes but keep editing sequences (backspace, insert)
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", raw_output)
    # Strip OSC sequences (title bar updates etc)
    cleaned = re.sub(r"\x1b\][^\x07]*\x07", "", cleaned)

    commands = []
    for line in cleaned.split("\n"):
        m = _PROMPT_RE.search(line)
        if not m:
            continue

        raw_cmd = m.group(1)
        # Handle \r-delimited re-draws: take the last non-empty segment
        segments = raw_cmd.split("\r")
        best = ""
        for seg in segments:
            replayed = _replay_edits(seg)
            if replayed:
                best = replayed

        # Skip prompt fragments that leaked through
        if not best:
            continue
        if best.startswith("──(root") or best.startswith("┌──"):
            continue
        # Skip lines that contain an embedded prompt (re-draw artifacts)
        if re.search(r"[a-z_]\w*@[a-z0-9_-]+:[^#$]*[#$]\s", best):
            continue

        commands.append(best)

    return commands


# =============================================================================
# Model tool call extraction
# =============================================================================


def extract_model_tool_calls(sample: dict) -> list[dict]:
    """Extract tool calls from a model eval sample.

    Returns list of dicts with keys: function, cmd.
    """
    tool_calls = []
    for msg in sample.get("messages", []):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            tool_calls.append({
                "function": tc.get("function", ""),
                "cmd": args.get("cmd", args.get("code", "")),
            })
    return tool_calls


# =============================================================================
# Eval file loading
# =============================================================================


def load_human_eval_sample(path: str | Path) -> dict | None:
    """Load the first sample from a human .eval.gz file.

    Returns None if the file has no samples (cancelled sessions).
    """
    path = Path(path)
    with gzip.open(path, "rb") as gz:
        raw = gz.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        sample_files = [n for n in zf.namelist() if n.startswith("samples/")]
        if not sample_files:
            return None
        with zf.open(sample_files[0]) as f:
            return json.load(f)


def load_model_eval_sample(eval_path: str | Path, task_id: str) -> dict | None:
    """Load a specific task's sample from a model .eval zip file.

    Returns None if the task is not found.
    """
    eval_path = Path(eval_path)
    with zipfile.ZipFile(eval_path) as zf:
        for name in zf.namelist():
            if not name.startswith("samples/"):
                continue
            with zf.open(name) as f:
                sample = json.load(f)
            sid = sample.get("id", "")
            # Handle CVEBench suffixes
            clean_id = sid.replace("-one_day", "").replace("-zero_day", "")
            if sid == task_id or clean_id == task_id:
                return sample
    return None


def load_all_model_samples(eval_path: str | Path) -> list[dict]:
    """Load all samples from a model .eval zip file."""
    eval_path = Path(eval_path)
    samples = []
    with zipfile.ZipFile(eval_path) as zf:
        for name in zf.namelist():
            if not name.startswith("samples/"):
                continue
            with zf.open(name) as f:
                samples.append(json.load(f))
    return samples


# =============================================================================
# Paste analysis (bracketed paste detection)
# =============================================================================

# Terminals wrap clipboard pastes in these escape sequences.
_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"

_PASTE_RE = re.compile(
    re.escape(_PASTE_START) + r"(.*?)" + re.escape(_PASTE_END),
    re.DOTALL,
)


def extract_paste_metrics(sample: dict) -> dict | None:
    """Extract paste metrics from a human eval sample's .input recording.

    Detects bracketed paste sequences to distinguish clipboard-pasted input
    from interactively typed input.

    Returns dict with keys:
        paste_ratio     - fraction of input bytes that were pasted (0.0-1.0)
        paste_events    - number of distinct paste operations
        pasted_bytes    - total bytes inside paste brackets
        typed_bytes     - total bytes outside paste brackets (minus markers)
        total_bytes     - pasted_bytes + typed_bytes

    Returns None if no .input log is available.
    """
    logs = sample.get("store", {}).get("HumanAgentState:logs", {})
    input_key = next((k for k in logs if k.endswith(".input")), None)
    if input_key is None:
        return None

    raw_input = logs[input_key]

    # Strip script header/footer lines
    body_start = raw_input.find("\n")
    body = raw_input[body_start + 1:] if body_start != -1 else raw_input
    footer_idx = body.rfind("Script done on")
    if footer_idx != -1:
        body = body[:footer_idx]

    # Find all paste blocks
    paste_blocks = _PASTE_RE.findall(body)
    paste_count = len(paste_blocks)
    pasted_bytes = sum(len(b) for b in paste_blocks)

    # Total content bytes minus marker overhead
    marker_overhead = paste_count * (len(_PASTE_START) + len(_PASTE_END))
    content_bytes = len(body) - marker_overhead
    if content_bytes <= 0:
        return {
            "paste_ratio": 0.0,
            "paste_events": 0,
            "pasted_bytes": 0,
            "typed_bytes": 0,
            "total_bytes": 0,
        }

    typed_bytes = max(content_bytes - pasted_bytes, 0)
    paste_ratio = min(pasted_bytes / content_bytes, 1.0) if content_bytes > 0 else 0.0

    return {
        "paste_ratio": paste_ratio,
        "paste_events": paste_count,
        "pasted_bytes": pasted_bytes,
        "typed_bytes": typed_bytes,
        "total_bytes": content_bytes,
    }


def _clean_for_matching(text: str) -> str:
    """Strip ANSI codes and control chars for text comparison."""
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z~]", "", text)
    text = re.sub(r"\x1b\][^\x07]*\x07", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def _strip_script_wrapper(content: str) -> str:
    """Strip script(1) header and footer lines from recorded content."""
    # Header: "Script started on ..."
    first_nl = content.find("\n")
    if first_nl != -1 and content[:first_nl].startswith("Script started"):
        content = content[first_nl + 1:]
    # Footer: "Script done on ..."
    footer = content.rfind("Script done on")
    if footer != -1:
        content = content[:footer]
    return content


def extract_novel_paste_metrics(sample: dict) -> dict | None:
    """Classify pasted lines as novel (external) vs previously-seen.

    Uses timing correlation: for each paste block, determines how much
    .output existed BEFORE the paste occurred (via the .timing file),
    then checks whether each pasted line appears in that prior output.

    This avoids the echo-counting bias where a pasted command's own echo
    (or echoes of later similar pastes) inflate the match count.

    Only considers lines >= 10 chars to avoid spurious short-string matches.

    Returns dict with keys:
        novel_lines       - paste lines not seen in prior output
        internal_lines    - paste lines that existed in prior output
        total_lines       - total paste lines analyzed (>= 10 chars)
        novel_ratio       - novel_lines / total_lines
        unique_lines      - count of distinct paste lines
        paste_diversity   - unique_lines / total_lines

    Requires .input, .output, AND .timing. Returns None if any are missing.
    """
    logs = sample.get("store", {}).get("HumanAgentState:logs", {})
    input_key = next((k for k in logs if k.endswith(".input")), None)
    output_key = next((k for k in logs if k.endswith(".output")), None)
    timing_key = next((k for k in logs if k.endswith(".timing")), None)

    if not all([input_key, output_key, timing_key]):
        return None

    # Strip script(1) headers/footers so byte positions align with timing
    input_body = _strip_script_wrapper(logs[input_key])
    output_body = _strip_script_wrapper(logs[output_key])
    timing_content = logs[timing_key]

    # Parse timing file: build checkpoints of (cumulative_I_bytes, cumulative_O_bytes)
    # at each I event.  Skip H (header) lines.
    i_cumulative = 0
    o_cumulative = 0
    io_checkpoints: list[tuple[int, int]] = []

    for line in timing_content.split("\n"):
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        event_type = parts[0]
        if event_type == "H":
            continue
        try:
            byte_count = int(parts[2])
        except ValueError:
            continue
        if event_type == "I":
            io_checkpoints.append((i_cumulative, o_cumulative))
            i_cumulative += byte_count
        elif event_type == "O":
            o_cumulative += byte_count

    if not io_checkpoints:
        return None

    # Find paste blocks with their byte offsets in input_body
    paste_matches = list(_PASTE_RE.finditer(input_body))

    novel_lines = 0
    internal_lines = 0
    total_lines = 0
    unique_lines: set[str] = set()

    for match in paste_matches:
        paste_offset = match.start()
        block_content = match.group(1)

        # Binary search: find the last checkpoint where i_pos <= paste_offset
        # to get the output byte count at the time of this paste.
        lo, hi = 0, len(io_checkpoints) - 1
        o_pos = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if io_checkpoints[mid][0] <= paste_offset:
                o_pos = io_checkpoints[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1

        # Clean the output that existed BEFORE this paste
        prior_output = _clean_for_matching(output_body[:o_pos])

        for raw_line in re.split(r"[\n\r]+", block_content):
            line = _clean_for_matching(raw_line).strip()
            if len(line) < 10:
                continue
            total_lines += 1
            unique_lines.add(line)

            if line in prior_output:
                internal_lines += 1
            else:
                novel_lines += 1

    return {
        "novel_lines": novel_lines,
        "internal_lines": internal_lines,
        "total_lines": total_lines,
        "novel_ratio": novel_lines / total_lines if total_lines > 0 else 0.0,
        "unique_lines": len(unique_lines),
        "paste_diversity": len(unique_lines) / total_lines if total_lines > 0 else 0.0,
    }
