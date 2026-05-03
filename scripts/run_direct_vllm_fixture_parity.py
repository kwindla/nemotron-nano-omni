#!/usr/bin/env python3
"""Compare cached vs uncached direct vLLM behavior on cache-state scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from nemotron_voice.services.nvidia.nemotron_omni import (
    BASH_TOOL_DEFINITION,
    DEFAULT_VOICE_SYSTEM_INSTRUCTION,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = ROOT / "media" / "cartesia-regression"
DEFAULT_IMAGE_PATH = ROOT / "media" / "cache_image_test.png"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "nemotron_3_nano_omni"
DEFAULT_AUDIO_CONTEXT_TEXT = (
    "User audio follows. Listen to it and respond to the user's latest request."
)


@dataclass(frozen=True)
class Action:
    label: str
    kind: str
    text: str | None = None
    media_path: Path | None = None


def _audio_action(label: str, filename: str) -> Action:
    return Action(
        label=label,
        kind="audio",
        media_path=DEFAULT_AUDIO_DIR / filename,
    )


def _image_action(label: str, text: str) -> Action:
    return Action(
        label=label,
        kind="image",
        text=text,
        media_path=DEFAULT_IMAGE_PATH,
    )


MIXED_20_ACTIONS = [
    _audio_action("audio-unicorn", "audio_unicorn_intro.wav"),
    Action(
        label="text-recall-unicorn",
        kind="text",
        text="What creature did I mention in the previous audio? Answer with one word only.",
    ),
    Action(
        label="text-tool-pwd",
        kind="text",
        text=(
            "You must use the bash tool now. Run exactly this command: pwd. "
            "Do not answer from memory or prior knowledge. Reply with only the tool stdout."
        ),
    ),
    Action(
        label="text-followup-pwd",
        kind="text",
        text="What exact path did the previous command print? Reply with the path only.",
    ),
    _audio_action("audio-tool-echo-one", "audio_tool_echo_one.wav"),
    Action(
        label="text-followup-echo-one",
        kind="text",
        text="What exact words did the previous spoken command print? Reply with those words only.",
    ),
    _audio_action("audio-dragon", "audio_dragon_intro.wav"),
    Action(
        label="text-recall-dragon",
        kind="text",
        text="What creature did I mention in the previous audio? Answer with one word only.",
    ),
    Action(
        label="text-tool-echo-two",
        kind="text",
        text=(
            "You must use the bash tool now. Run exactly this command: "
            "echo spark text two. Do not answer from memory or prior knowledge. "
            "Reply with only the tool stdout."
        ),
    ),
    Action(
        label="text-followup-echo-two",
        kind="text",
        text="What exact words did the previous command print? Reply with those words only.",
    ),
    _audio_action("audio-tool-echo-three", "audio_tool_echo_three.wav"),
    Action(
        label="text-followup-echo-three",
        kind="text",
        text="What exact words did the previous spoken command print? Reply with those words only.",
    ),
    Action(
        label="text-recall-dragon-again",
        kind="text",
        text="What creature was mentioned in the most recent creature-themed audio turn? Answer with one word only.",
    ),
    _audio_action("audio-math", "audio_math_1000_div_25.wav"),
    Action(
        label="text-followup-math",
        kind="text",
        text="Repeat just the number from the previous audio answer.",
    ),
    Action(
        label="text-tool-echo-four",
        kind="text",
        text=(
            "You must use the bash tool now. Run exactly this command: "
            "echo spark text four. Do not answer from memory or prior knowledge. "
            "Reply with only the tool stdout."
        ),
    ),
    Action(
        label="text-followup-echo-four",
        kind="text",
        text="What exact words did the previous command print? Reply with those words only.",
    ),
    _audio_action("audio-goodbye", "audio_goodbye.wav"),
    Action(
        label="text-followup-goodbye",
        kind="text",
        text="What was the last word I asked you to say? Reply with one word only.",
    ),
    _audio_action("audio-tool-echo-five", "audio_tool_echo_five.wav"),
]

SCENARIOS: dict[str, list[Action]] = {
    "mixed20": MIXED_20_ACTIONS,
    "text-text": [
        Action(
            label="text-unicorn",
            kind="text",
            text="Tell me in one sentence about a unicorn.",
        ),
        Action(
            label="text-recall-unicorn",
            kind="text",
            text="What creature did I just ask about? Answer with one word only.",
        ),
    ],
    "text-tool-text": [
        Action(
            label="text-tool-pwd",
            kind="text",
            text=(
                "You must use the bash tool now. Run exactly this command: pwd. "
                "Do not answer from memory or prior knowledge. Reply with only the tool stdout."
            ),
        ),
        Action(
            label="text-followup-pwd",
            kind="text",
            text="What exact path did the previous command print? Reply with the path only.",
        ),
    ],
    "audio-text": [
        _audio_action("audio-unicorn", "audio_unicorn_intro.wav"),
        Action(
            label="text-recall-unicorn",
            kind="text",
            text="What creature did I mention in the previous audio? Answer with one word only.",
        ),
    ],
    "audio-tool-text": [
        _audio_action("audio-tool-echo-one", "audio_tool_echo_one.wav"),
        Action(
            label="text-followup-echo-one",
            kind="text",
            text="What exact words did the previous spoken command print? Reply with those words only.",
        ),
    ],
    "image-text": [
        _image_action(
            "image-describe",
            "Describe this image in a short phrase.",
        ),
        Action(
            label="text-recall-square",
            kind="text",
            text="What color is the square? Answer with one word only.",
        ),
    ],
    "image-tool-text": [
        _image_action(
            "image-tool-echo",
            "Look at this image, then use the bash tool to run exactly this command: "
            "echo red square blue circle. Reply with only the tool stdout.",
        ),
        Action(
            label="text-followup-image-tool",
            kind="text",
            text="What exact words did the previous command print? Reply with those words only.",
        ),
    ],
    "publish-plain-text": [
        Action(
            label="text-dragon",
            kind="text",
            text="Tell me in one sentence about a dragon.",
        ),
        Action(
            label="text-recall-dragon",
            kind="text",
            text="What creature did I just ask about? Answer with one word only.",
        ),
    ],
    "publish-tool": [
        Action(
            label="text-tool-echo-two",
            kind="text",
            text=(
                "You must use the bash tool now. Run exactly this command: "
                "echo spark text two. Do not answer from memory or prior knowledge. "
                "Reply with only the tool stdout."
            ),
        ),
        Action(
            label="text-followup-echo-two",
            kind="text",
            text="What exact words did the previous command print? Reply with those words only.",
        ),
    ],
    "publish-multimodal": [
        _image_action(
            "image-describe",
            "Describe this image in a short phrase.",
        ),
        Action(
            label="text-recall-circle",
            kind="text",
            text="What color is the circle? Answer with one word only.",
        ),
    ],
    "publish-multimodal-tool": [
        _audio_action("audio-tool-echo-three", "audio_tool_echo_three.wav"),
        Action(
            label="text-followup-echo-three",
            kind="text",
            text="What exact words did the previous spoken command print? Reply with those words only.",
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="mixed20")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optionally stop after the first N turns.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help=(
            "Optional vLLM conversation trace directory. When set, compare the "
            "actual per-request rendered prompt traces for cached vs uncached parity."
        ),
    )
    return parser.parse_args()


def normalize_text(text: str | None) -> str:
    return (text or "").replace("\r\n", "\n")


def hash_token_ids(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def audio_user_message(path: Path) -> dict[str, Any]:
    uri = path.resolve().as_uri()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": DEFAULT_AUDIO_CONTEXT_TEXT},
            {"type": "audio_url", "audio_url": {"url": uri}, "uuid": uri},
        ],
    }


def image_user_message(path: Path, prompt: str) -> dict[str, Any]:
    uri = path.resolve().as_uri()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": uri}},
        ],
    }


def text_user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def build_user_message(action: Action) -> dict[str, Any]:
    if action.kind == "audio":
        assert action.media_path is not None
        return audio_user_message(action.media_path)
    if action.kind == "image":
        assert action.media_path is not None
        assert action.text is not None
        return image_user_message(action.media_path, action.text)
    if action.kind == "text":
        return text_user_message(action.text or "")
    raise ValueError(f"Unsupported action kind: {action.kind}")


def execute_bash(code: str) -> str:
    proc = subprocess.run(
        ["bash", "-lc", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout and stderr:
        return f"<stdout>{stdout}</stdout>\n<stderr>{stderr}</stderr>"
    if stdout:
        return stdout
    return stderr


class DirectClient:
    def __init__(self, *, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _payload(
        self,
        messages: list[dict[str, Any]],
        *,
        conversation_id: str | None,
        require_cache: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 1234,
            "tools": [BASH_TOOL_DEFINITION],
            "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if require_cache:
            payload["conversation_require_cache"] = True
        return payload

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        conversation_id: str | None = None,
        require_cache: bool = False,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions/render",
            json=self._payload(
                messages,
                conversation_id=conversation_id,
                require_cache=require_cache,
            ),
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        conversation_id: str | None = None,
        require_cache: bool = False,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=self._payload(
                messages,
                conversation_id=conversation_id,
                require_cache=require_cache,
            ),
            headers=({"X-Request-Id": request_id} if request_id else None),
            timeout=180,
        )
        response.raise_for_status()
        return response.json()


def assistant_message(raw_message: dict[str, Any]) -> dict[str, Any]:
    result = {"role": "assistant", "content": raw_message.get("content")}
    if raw_message.get("tool_calls"):
        result["tool_calls"] = raw_message["tool_calls"]
    return result


def tool_message(tool_call_id: str, content: str, name: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


def load_trace_oracle(trace_dir: Path, request_id: str) -> dict[str, Any] | None:
    candidates = [
        trace_dir / f"{request_id}.vllm-rendered-prompt.json",
        trace_dir / "vllm" / f"{request_id}.vllm-rendered-prompt.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def run_mode(
    client: DirectClient,
    *,
    actions: list[Action],
    scenario_name: str,
    cached: bool,
    max_turns: int | None,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_VOICE_SYSTEM_INSTRUCTION}
    ]
    conversation_id = f"direct-parity-{scenario_name}-{uuid.uuid4().hex}" if cached else None
    turn_records: list[dict[str, Any]] = []
    mode_name = "cached" if cached else "uncached"

    for turn_index, action in enumerate(actions, start=1):
        if max_turns is not None and turn_index > max_turns:
            break
        user_message = build_user_message(action)
        current_messages = [user_message] if cached and turn_index > 1 else [*history, user_message]
        require_cache = cached and turn_index > 1
        step: dict[str, Any] = {
            "turn": turn_index,
            "label": action.label,
            "kind": action.kind,
            "requests": [],
        }

        request_id = f"{scenario_name}-{mode_name}-turn-{turn_index:03d}-pass-01"
        response = client.chat(
            current_messages,
            conversation_id=conversation_id,
            require_cache=require_cache,
            request_id=request_id,
        )
        message = response["choices"][0]["message"]
        request_record: dict[str, Any] = {
            "phase": "user",
            "request_id": request_id,
            "request_messages": current_messages,
            "require_cache": require_cache,
            "response_message": message,
        }
        step["requests"].append(request_record)
        history.append(user_message)

        tool_rounds = 0
        while message.get("tool_calls"):
            tool_rounds += 1
            if tool_rounds > 3:
                raise RuntimeError(f"{scenario_name}:{action.label}: exceeded tool round limit")
            assistant_tool_message = assistant_message(message)
            history.append(assistant_tool_message)
            tool_call = message["tool_calls"][0]
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_output = execute_bash(tool_args["code"])
            tool_result_message = tool_message(tool_call["id"], tool_output, tool_name)
            history.append(tool_result_message)

            followup_messages = [tool_result_message] if cached else [*history]
            request_id = (
                f"{scenario_name}-{mode_name}-turn-{turn_index:03d}-"
                f"pass-{tool_rounds + 1:02d}"
            )
            response = client.chat(
                followup_messages,
                conversation_id=conversation_id,
                require_cache=cached,
                request_id=request_id,
            )
            message = response["choices"][0]["message"]
            request_record = {
                "phase": f"tool_round_{tool_rounds}",
                "request_id": request_id,
                "tool_command": tool_args["code"],
                "tool_output": tool_output,
                "request_messages": followup_messages,
                "require_cache": cached,
                "response_message": message,
            }
            step["requests"].append(request_record)

        history.append(assistant_message(message))
        step["final_response"] = normalize_text(message.get("content"))
        turn_records.append(step)

    return {
        "scenario": scenario_name,
        "cached": cached,
        "conversation_id": conversation_id,
        "turns": turn_records,
    }


def compare_mode_outputs(
    cached: dict[str, Any],
    uncached: dict[str, Any],
    *,
    trace_dir: Path | None,
) -> list[dict[str, Any]]:
    divergences: list[dict[str, Any]] = []
    uncached_by_label = {turn["label"]: turn for turn in uncached["turns"]}

    for cached_turn in cached["turns"]:
        label = cached_turn["label"]
        uncached_turn = uncached_by_label[label]

        if len(cached_turn["requests"]) != len(uncached_turn["requests"]):
            divergences.append(
                {
                    "label": label,
                    "reason": "request_count_mismatch",
                    "cached_request_count": len(cached_turn["requests"]),
                    "uncached_request_count": len(uncached_turn["requests"]),
                }
            )
            break

        for cached_req, uncached_req in zip(
            cached_turn["requests"], uncached_turn["requests"], strict=True
        ):
            if cached_req["phase"] != uncached_req["phase"]:
                divergences.append(
                    {
                        "label": label,
                        "reason": "phase_mismatch",
                        "cached_phase": cached_req["phase"],
                        "uncached_phase": uncached_req["phase"],
                    }
                )
                return divergences
            if trace_dir is not None:
                cached_oracle = load_trace_oracle(trace_dir, cached_req["request_id"])
                uncached_oracle = load_trace_oracle(trace_dir, uncached_req["request_id"])
                if cached_oracle is None or uncached_oracle is None:
                    divergences.append(
                        {
                            "label": label,
                            "reason": "missing_trace_oracle",
                            "phase": cached_req["phase"],
                        }
                    )
                    return divergences
                if cached_oracle["prompt_token_count"] != uncached_oracle["prompt_token_count"]:
                    divergences.append(
                        {
                            "label": label,
                            "reason": "prompt_token_count_mismatch",
                            "phase": cached_req["phase"],
                            "cached_prompt_token_count": cached_oracle["prompt_token_count"],
                            "uncached_prompt_token_count": uncached_oracle["prompt_token_count"],
                        }
                    )
                    return divergences
                if cached_oracle["prompt_text_sha256"] != uncached_oracle["prompt_text_sha256"]:
                    divergences.append(
                        {
                            "label": label,
                            "reason": "prompt_text_hash_mismatch",
                            "phase": cached_req["phase"],
                            "cached_prompt_text_sha256": cached_oracle["prompt_text_sha256"],
                            "uncached_prompt_text_sha256": uncached_oracle["prompt_text_sha256"],
                        }
                    )
                    return divergences

        cached_tools = [
            req.get("tool_command")
            for req in cached_turn["requests"]
            if req["phase"].startswith("tool_round_")
        ]
        uncached_tools = [
            req.get("tool_command")
            for req in uncached_turn["requests"]
            if req["phase"].startswith("tool_round_")
        ]
        if cached_tools != uncached_tools:
            divergences.append(
                {
                    "label": label,
                    "reason": "tool_command_mismatch",
                    "cached_tool_commands": cached_tools,
                    "uncached_tool_commands": uncached_tools,
                }
            )
            break

        if cached_turn["final_response"] != uncached_turn["final_response"]:
            divergences.append(
                {
                    "label": label,
                    "reason": "final_response_mismatch",
                    "cached_response": cached_turn["final_response"],
                    "uncached_response": uncached_turn["final_response"],
                }
            )
            break

    return divergences


def main() -> int:
    args = parse_args()
    actions = SCENARIOS[args.scenario]
    client = DirectClient(base_url=args.base_url, model=args.model)
    cached = run_mode(
        client,
        actions=actions,
        scenario_name=args.scenario,
        cached=True,
        max_turns=args.max_turns,
    )
    uncached = run_mode(
        client,
        actions=actions,
        scenario_name=args.scenario,
        cached=False,
        max_turns=args.max_turns,
    )

    divergences = compare_mode_outputs(
        cached,
        uncached,
        trace_dir=args.trace_dir,
    )
    summary = {
        "scenario": args.scenario,
        "trace_dir": str(args.trace_dir) if args.trace_dir else None,
        "cached": cached,
        "uncached": uncached,
        "divergences": divergences,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.summary_json)
    if divergences:
        print(json.dumps(divergences[0], indent=2))
        return 1
    print("no divergences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
