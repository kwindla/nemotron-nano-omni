#!/usr/bin/env python3
"""Patch Nemotron Nano Omni chat history rendering for tool-call turns."""

from __future__ import annotations

import os
import sys
from pathlib import Path


OLD = "                    {{- '</function>\\n</tool_call>\\n' -}}\n"
NEW = (
    "                    {{- '</function>\\n</tool_call>' -}}\n"
    "                    {%- if not loop.last %}\n"
    "                        {{- '\\n' -}}\n"
    "                    {%- endif %}\n"
)


def template_path(arg: str | None) -> Path:
    raw = arg or os.environ.get("NEMOTRON_MODEL_PATH")
    if not raw:
        raise SystemExit(
            "Usage: patch_nemotron_chat_template.py <model-dir-or-chat-template>"
        )
    path = Path(raw)
    if path.is_dir():
        path = path / "chat_template.jinja"
    return path


def main() -> int:
    path = template_path(sys.argv[1] if len(sys.argv) > 1 else None)
    if not path.exists():
        raise SystemExit(f"Missing chat template: {path}")

    text = path.read_text(encoding="utf-8")
    if NEW in text:
        print(f"Chat template already patched: {path}")
        return 0
    if OLD not in text:
        raise SystemExit(
            "Expected Nemotron tool-call template block was not found in "
            f"{path}; inspect the model template before patching."
        )

    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched chat template: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
