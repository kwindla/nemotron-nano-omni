#!/usr/bin/env python3
"""Live integration tests for Nemotron Omni conversation prefix caching.

This script intentionally exercises the OpenAI-compatible HTTP endpoint rather
than only unit-testing the cache classes. By default it starts a patched vLLM
server subprocess, runs the tests, and tears the server down. Use
``--reuse-server`` to run against an already-running endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


DEFAULT_BASE_URL = _env_str("NEMOTRON_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = _env_str("NEMOTRON_VLLM_MODEL", "nemotron_3_nano_omni")
DEFAULT_AUDIO = _env_path("NEMOTRON_AUDIO_FIXTURE", ROOT / "media" / "cartesia-unicorn.wav")
DEFAULT_LOG = _env_path("NEMOTRON_VLLM_LOG", ROOT / "logs" / "vllm-prefix-cache.log")
DEFAULT_MODEL_PATH = _env_path(
    "NEMOTRON_MODEL_PATH",
    ROOT / "models" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
)
DEFAULT_VLLM_PYTHON = _env_path(
    "NEMOTRON_VLLM_PYTHON",
    ROOT / ".venv-vllm-0.20.0-cu132" / "bin" / "python3",
)
DEFAULT_VLLM_BIN = _env_path(
    "NEMOTRON_VLLM_BIN",
    ROOT / ".venv-vllm-0.20.0-cu132" / "bin" / "vllm",
)
DEFAULT_VLLM_SOURCE_DIR = _env_path(
    "NEMOTRON_VLLM_SOURCE_DIR",
    ROOT / "vllm-v0.20.0",
)
DEFAULT_GPU_MEMORY_UTILIZATION = _env_float(
    "NEMOTRON_VLLM_GPU_MEMORY_UTILIZATION", 0.75
)
DEFAULT_MAX_MODEL_LEN = _env_int("NEMOTRON_VLLM_MAX_MODEL_LEN", 4096)
DEFAULT_MAX_NUM_SEQS = _env_int("NEMOTRON_VLLM_MAX_NUM_SEQS", 1)
DEFAULT_MAX_NUM_BATCHED_TOKENS = _env_int(
    "NEMOTRON_VLLM_MAX_NUM_BATCHED_TOKENS", DEFAULT_MAX_MODEL_LEN
)
DEFAULT_LIMIT_MM_PER_PROMPT = _env_str(
    "NEMOTRON_VLLM_LIMIT_MM_PER_PROMPT", '{"audio": 8}'
)
DEFAULT_ALLOWED_LOCAL_MEDIA_PATH = _env_str(
    "NEMOTRON_VLLM_ALLOWED_LOCAL_MEDIA_PATH", "/"
)
DEFAULT_MM_ENCODER_ATTN_BACKEND = _env_str(
    "NEMOTRON_VLLM_MM_ENCODER_ATTN_BACKEND", "TORCH_SDPA"
)
DEFAULT_SKIP_MM_PROFILING = _env_bool("NEMOTRON_VLLM_SKIP_MM_PROFILING", True)
DEFAULT_ENFORCE_EAGER = _env_bool("NEMOTRON_VLLM_ENFORCE_EAGER", True)
DEFAULT_REASONING_PARSER = _env_str(
    "NEMOTRON_VLLM_REASONING_PARSER", "nemotron_v3"
)
DEFAULT_ENABLE_AUTO_TOOL_CHOICE = _env_bool(
    "NEMOTRON_VLLM_ENABLE_AUTO_TOOL_CHOICE", True
)
DEFAULT_TOOL_CALL_PARSER = _env_str(
    "NEMOTRON_VLLM_TOOL_CALL_PARSER", "qwen3_coder"
)
DEFAULT_MOE_BACKEND = _env_str("NEMOTRON_VLLM_MOE_BACKEND", "cutlass")
DEFAULT_ENABLE_PREFIX_CACHING = _env_bool(
    "NEMOTRON_VLLM_ENABLE_PREFIX_CACHING", True
)
DEFAULT_MAMBA_CACHE_MODE = _env_str("NEMOTRON_VLLM_MAMBA_CACHE_MODE", "align")
DEFAULT_MAMBA_CACHE_DTYPE = _env_str("NEMOTRON_VLLM_MAMBA_CACHE_DTYPE", "auto")
DEFAULT_MAMBA_SSM_CACHE_DTYPE = _env_str(
    "NEMOTRON_VLLM_MAMBA_SSM_CACHE_DTYPE", "auto"
)
DEFAULT_MAMBA_BACKEND = _env_str("NEMOTRON_VLLM_MAMBA_BACKEND", "triton")
DEFAULT_ATTENTION_BACKEND = os.getenv("NEMOTRON_VLLM_ATTENTION_BACKEND")
DEFAULT_KV_CACHE_MEMORY_BYTES = os.getenv("NEMOTRON_VLLM_KV_CACHE_MEMORY_BYTES")
ATTACH_RE = re.compile(
    r"Attached conversation cache for (?P<cid>\S+) generation \S+ "
    r"tokens=(?P<tokens>\d+) copies=(?P<copies>\d+)"
)
ATTACH_SKIP_RE = re.compile(r"Conversation cache attach skipped for (?P<cid>\S+)\b")
ERROR_RE = re.compile(r"\b(ERROR|Traceback|Exception|No free blocks|Failed to stage)\b")


class TestFailure(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    ttft: float | None = None
    total_time: float | None = None
    tps: float | None = None
    raw: dict[str, Any] | None = None


@dataclass
class TestResult:
    name: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def log_offset(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def read_log_from(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", errors="replace")


def attach_lines(log_text: str, conversation_id: str) -> list[dict[str, int]]:
    lines: list[dict[str, int]] = []
    for match in ATTACH_RE.finditer(log_text):
        if match.group("cid") != conversation_id:
            continue
        lines.append(
            {
                "tokens": int(match.group("tokens")),
                "copies": int(match.group("copies")),
            }
        )
    return lines


def assert_no_server_errors(log_text: str) -> None:
    for line in log_text.splitlines():
        if ERROR_RE.search(line):
            raise TestFailure(f"server log contains error line: {line[:240]}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise TestFailure(f"no JSON object found in output: {text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise TestFailure(f"invalid JSON output: {text!r}") from exc


def gpu_memory_mb() -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    values: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            continue
    return values


def tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def vllm_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.vllm_python),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model_path),
        "--served-model-name",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--limit-mm-per-prompt",
        args.limit_mm_per_prompt,
        "--allowed-local-media-path",
        args.allowed_local_media_path,
        "--mm-encoder-attn-backend",
        args.mm_encoder_attn_backend,
        "--reasoning-parser",
        args.reasoning_parser,
        "--tool-call-parser",
        args.tool_call_parser,
        "--moe-backend",
        args.moe_backend,
    ]
    if args.kv_cache_memory_bytes:
        command.extend(["--kv-cache-memory-bytes", args.kv_cache_memory_bytes])
    if args.attention_backend:
        command.extend(["--attention-backend", args.attention_backend])
    if args.skip_mm_profiling:
        command.append("--skip-mm-profiling")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.enable_auto_tool_choice:
        command.append("--enable-auto-tool-choice")
    if args.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    else:
        command.append("--no-enable-prefix-caching")
    if args.mamba_cache_mode:
        command.extend(["--mamba-cache-mode", args.mamba_cache_mode])
    if args.mamba_cache_dtype and args.mamba_cache_dtype != "auto":
        command.extend(["--mamba-cache-dtype", args.mamba_cache_dtype])
    if args.mamba_ssm_cache_dtype and args.mamba_ssm_cache_dtype != "auto":
        command.extend(["--mamba-ssm-cache-dtype", args.mamba_ssm_cache_dtype])
    if args.mamba_backend:
        command.extend(["--mamba-backend", args.mamba_backend])
    return command


def start_vllm(args: argparse.Namespace) -> tuple[subprocess.Popen[bytes], Any]:
    if not args.vllm_python.exists():
        raise TestFailure(f"missing vLLM Python: {args.vllm_python}")
    if not args.model_path.exists():
        raise TestFailure(f"missing model path: {args.model_path}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("wb")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = (
        f"{args.vllm_source_dir}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(args.vllm_source_dir)
    )
    proc = subprocess.Popen(
        vllm_command(args),
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return proc, log_file


def terminate_vllm(proc: subprocess.Popen[bytes], timeout: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


class VllmClient:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def wait_healthy(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                response = requests.get(
                    f"{self.base_url}/models", timeout=min(5.0, timeout)
                )
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise TestFailure(f"vLLM endpoint is not healthy: {last_error}")

    def wait_healthy_with_process(
        self,
        timeout: float,
        proc: subprocess.Popen[bytes],
        log_path: Path,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error: str | None = None
        while time.monotonic() < deadline:
            returncode = proc.poll()
            if returncode is not None:
                raise TestFailure(
                    f"vLLM exited during startup with code {returncode}.\n"
                    f"Log tail:\n{tail_text(log_path)}"
                )
            try:
                response = requests.get(
                    f"{self.base_url}/models", timeout=min(5.0, timeout)
                )
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise TestFailure(
            f"vLLM endpoint is not healthy after {timeout:.1f}s: {last_error}\n"
            f"Log tail:\n{tail_text(log_path)}"
        )

    def payload(
        self,
        messages: list[dict[str, Any]],
        *,
        conversation_id: str | None = None,
        cache_salt: str | None = None,
        max_tokens: int = 80,
        stream: bool = False,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if cache_salt:
            payload["cache_salt"] = cache_salt
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if extra_payload:
            payload.update(extra_payload)
        return payload

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        conversation_id: str | None = None,
        cache_salt: str | None = None,
        max_tokens: int = 80,
        extra_payload: dict[str, Any] | None = None,
    ) -> ChatResult:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=self.payload(
                messages,
                conversation_id=conversation_id,
                cache_salt=cache_salt,
                max_tokens=max_tokens,
                extra_payload=extra_payload,
            ),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise TestFailure(
                f"chat request failed: HTTP {response.status_code}: {response.text[:1000]}"
            )
        data = response.json()
        content = data["choices"][0]["message"].get("content") or ""
        return ChatResult(content=content, usage=data.get("usage") or {}, raw=data)

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        conversation_id: str | None = None,
        cache_salt: str | None = None,
        max_tokens: int = 96,
    ) -> ChatResult:
        payload = self.payload(
            messages,
            conversation_id=conversation_id,
            cache_salt=cache_salt,
            max_tokens=max_tokens,
            stream=True,
        )
        for attempt in range(6):
            start = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                break
            body = response.text[:1000]
            response.close()
            if (
                response.status_code == 409
                and conversation_id
                and "already generating" in body.lower()
                and attempt < 5
            ):
                time.sleep(0.25 * (attempt + 1))
                continue
            raise TestFailure(f"stream request failed: HTTP {response.status_code}: {body}")
        else:
            raise TestFailure("stream request retry loop exhausted unexpectedly")

        first_token_time: float | None = None
        usage: dict[str, Any] = {}
        chunks: list[str] = []
        with response:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks.append(text)

        end = time.perf_counter()
        ttft = None if first_token_time is None else first_token_time - start
        total = end - start
        completion_tokens = usage.get("completion_tokens")
        tps = None
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            decode_time = total - (ttft or 0.0)
            if decode_time > 0:
                tps = completion_tokens / decode_time
        return ChatResult(
            content="".join(chunks),
            usage=usage,
            ttft=ttft,
            total_time=total,
            tps=tps,
        )


def user(text: str | list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def audio_content(audio_path: Path, text: str) -> list[dict[str, Any]]:
    audio_url = Path(os.path.abspath(audio_path)).as_uri()
    return [
        {"type": "audio_url", "audio_url": {"url": audio_url}, "uuid": audio_url},
        {"type": "text", "text": text},
    ]


def assert_attached(log_path: Path, offset: int, conversation_id: str) -> list[dict[str, int]]:
    log_text = read_log_from(log_path, offset)
    assert_no_server_errors(log_text)
    lines = attach_lines(log_text, conversation_id)
    if not lines:
        raise TestFailure(f"no conversation-cache attach log for {conversation_id}")
    return lines


def assert_no_attach_skips(log_path: Path, offset: int, conversation_id: str) -> None:
    log_text = read_log_from(log_path, offset)
    for match in ATTACH_SKIP_RE.finditer(log_text):
        if match.group("cid") == conversation_id:
            raise TestFailure(
                f"conversation cache attach skip detected for {conversation_id}"
            )


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and report the output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def extract_single_tool_call(result: ChatResult) -> dict[str, Any]:
    if not result.raw:
        raise TestFailure("tool-call response is missing the raw payload")
    message = result.raw["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) != 1:
        raise TestFailure(f"expected exactly one tool call, got: {message!r}")
    return tool_calls[0]


def tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def run_equivalence_test(client: VllmClient, log_path: Path) -> TestResult:
    name = "cached_vs_uncached_equivalence"
    offset = log_offset(log_path)
    cid = f"it-equivalence-{uuid.uuid4().hex[:8]}"
    salt = cid
    facts = (
        "Store this dossier for the next turn. Reply with exactly STORED.\n\n"
        "Dossier:\n"
        "- codename: LANTERN-42\n"
        "- city: Quito\n"
        "- alloy: vanadium\n"
        "- phrase: silver comet\n"
        "- sequence: 9, 4, 7, 1\n"
    )
    first_messages = [user(facts)]
    first = client.chat(
        first_messages,
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=16,
    )
    second_user = (
        "Using only the dossier above, return one compact JSON object with "
        "keys codename, city, alloy, phrase, sequence_sum. No markdown and no "
        "extra text."
    )
    history = [user(facts), assistant(first.content), user(second_user)]
    cached = client.chat(
        history,
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=96,
    )
    uncached = client.chat(
        history,
        cache_salt=f"uncached-{uuid.uuid4().hex}",
        max_tokens=96,
    )
    attach = assert_attached(log_path, offset, cid)

    cached_json = extract_json_object(cached.content)
    uncached_json = extract_json_object(uncached.content)
    expected = {
        "codename": "LANTERN-42",
        "city": "Quito",
        "alloy": "vanadium",
        "phrase": "silver comet",
        "sequence_sum": 21,
    }
    for key, value in expected.items():
        if str(cached_json.get(key)) != str(value):
            raise TestFailure(f"cached output wrong for {key}: {cached.content!r}")
        if str(uncached_json.get(key)) != str(value):
            raise TestFailure(f"uncached output wrong for {key}: {uncached.content!r}")
    if cached_json != uncached_json:
        raise TestFailure(
            "cached and uncached parsed JSON differ: "
            f"cached={cached_json!r} uncached={uncached_json!r}"
        )
    return TestResult(
        name=name,
        ok=True,
        details={
            "first_output": normalize_text(first.content),
            "cached_output": normalize_text(cached.content),
            "uncached_output": normalize_text(uncached.content),
            "attach": attach,
        },
    )


def run_text_suffix_only_equivalence_test(
    client: VllmClient, log_path: Path
) -> TestResult:
    name = "text_suffix_only_equivalence"
    offset = log_offset(log_path)
    cid = f"it-suffix-text-{uuid.uuid4().hex[:8]}"
    salt = cid
    facts = (
        "Remember this compact record. Reply exactly RECORDED.\n"
        "Record: codename=EMBER-91; city=Oslo; mineral=quartz; "
        "phrase=violet harbor; numbers=5,8,13."
    )
    first = client.chat(
        [user(facts)],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=16,
    )
    second_user = (
        "Using the remembered record, return exactly one JSON object with keys "
        "codename, city, mineral, phrase, number_sum. No markdown."
    )
    suffix_only = client.chat(
        [assistant(first.content), user(second_user)],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=80,
        extra_payload={"conversation_require_cache": True},
    )
    full_history = [user(facts), assistant(first.content), user(second_user)]
    uncached = client.chat(
        full_history,
        cache_salt=f"uncached-{uuid.uuid4().hex}",
        max_tokens=80,
    )
    attach = assert_attached(log_path, offset, cid)

    suffix_json = extract_json_object(suffix_only.content)
    uncached_json = extract_json_object(uncached.content)
    expected = {
        "codename": "EMBER-91",
        "city": "Oslo",
        "mineral": "quartz",
        "phrase": "violet harbor",
        "number_sum": 26,
    }
    for key, value in expected.items():
        if str(suffix_json.get(key)) != str(value):
            raise TestFailure(
                f"suffix-only cached output wrong for {key}: {suffix_only.content!r}"
            )
        if str(uncached_json.get(key)) != str(value):
            raise TestFailure(f"uncached output wrong for {key}: {uncached.content!r}")
    if suffix_json != uncached_json:
        raise TestFailure(
            "suffix-only cached and uncached parsed JSON differ: "
            f"cached={suffix_json!r} uncached={uncached_json!r}"
        )
    return TestResult(
        name=name,
        ok=True,
        details={
            "first_output": normalize_text(first.content),
            "suffix_output": normalize_text(suffix_only.content),
            "uncached_output": normalize_text(uncached.content),
            "attach": attach,
        },
    )


def run_hard_mamba_context_test(client: VllmClient, log_path: Path) -> TestResult:
    name = "hard_mamba_context_dependency"
    offset = log_offset(log_path)
    cid = f"it-mamba-{uuid.uuid4().hex[:8]}"
    salt = cid
    setup = (
        "Remember this vault state for the next turn. Reply only MEMORY SET.\n"
        "The operator is Aria. The blue prism is in drawer 17. The brass key "
        "is under the fern. The passphrase is river under moon. The decoy "
        "passphrase is glass over sun and it must be ignored."
    )
    first = client.chat(
        [user(setup)],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=16,
    )
    question = (
        "Use the remembered vault state. Answer exactly in this format and "
        "nothing else: <operator>|<passphrase>|<drawer-number>"
    )
    history = [user(setup), assistant(first.content), user(question)]
    cached = client.chat(
        history,
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=48,
    )
    uncached = client.chat(
        history,
        cache_salt=f"uncached-{uuid.uuid4().hex}",
        max_tokens=48,
    )
    attach = assert_attached(log_path, offset, cid)
    cached_norm = normalize_text(cached.content)
    uncached_norm = normalize_text(uncached.content)
    required = ["Aria", "river under moon", "17"]
    for value in required:
        if value.lower() not in cached_norm.lower():
            raise TestFailure(f"cached output is missing {value!r}: {cached.content!r}")
        if value.lower() not in uncached_norm.lower():
            raise TestFailure(
                f"uncached output is missing {value!r}: {uncached.content!r}"
            )
    if cached_norm != uncached_norm:
        raise TestFailure(
            f"cached and uncached outputs differ: {cached_norm!r} != {uncached_norm!r}"
        )
    return TestResult(
        name=name,
        ok=True,
        details={
            "first_output": normalize_text(first.content),
            "output": cached_norm,
            "attach": attach,
        },
    )


def run_tool_multiturn_cache_test(client: VllmClient, log_path: Path) -> TestResult:
    name = "tool_multiturn_suffix_only_cache"
    offset = log_offset(log_path)
    cid = f"it-tool-{uuid.uuid4().hex[:8]}"
    salt = cid
    tool_payload = {"tools": [BASH_TOOL], "tool_choice": "auto"}
    repo_root = str(ROOT)

    turn1_first = client.chat(
        [user("Use the bash tool to run pwd and report the output only.")],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=64,
        extra_payload=tool_payload,
    )
    turn1_tool_call = extract_single_tool_call(turn1_first)
    turn1_message = turn1_first.raw["choices"][0]["message"] if turn1_first.raw else {}
    turn1_final = client.chat(
        [turn1_message, tool_message(turn1_tool_call["id"], f"{repo_root}\n")],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=32,
        extra_payload={
            **tool_payload,
            "conversation_require_cache": True,
        },
    )

    turn2_first = client.chat(
        [
            turn1_message,
            tool_message(turn1_tool_call["id"], f"{repo_root}\n"),
            assistant(turn1_final.content),
            user(
                "Use the bash tool to run basename "
                f"{repo_root} and report the output only."
            )
        ],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=64,
        extra_payload={
            **tool_payload,
            "conversation_require_cache": True,
        },
    )
    turn2_message = turn2_first.raw["choices"][0]["message"] if turn2_first.raw else {}
    turn2_tool_calls = turn2_message.get("tool_calls") or []
    if len(turn2_tool_calls) != 1:
        raise TestFailure(f"expected exactly one tool call, got: {turn2_message!r}")
    turn2_tool_call = turn2_tool_calls[0]
    turn2_final = client.chat(
        [turn2_message, tool_message(turn2_tool_call["id"], "nemotron-nano-omni\n")],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=24,
        extra_payload={
            **tool_payload,
            "conversation_require_cache": True,
        },
    )

    turn3_final = client.chat(
        [
            turn2_message,
            tool_message(turn2_tool_call["id"], "nemotron-nano-omni\n"),
            assistant(turn2_final.content),
            user("What exact token did the previous command print? Reply with that token only."),
        ],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=16,
        extra_payload={
            **tool_payload,
            "conversation_require_cache": True,
        },
    )

    attach = assert_attached(log_path, offset, cid)
    assert_no_attach_skips(log_path, offset, cid)
    if len(attach) < 4:
        raise TestFailure(
            f"expected at least 4 attach events for tool turns, saw {len(attach)}"
        )

    if repo_root not in turn1_final.content:
        raise TestFailure(f"turn 1 tool follow-up output is wrong: {turn1_final.content!r}")
    if "nemotron-nano-omni" not in turn2_final.content:
        raise TestFailure(f"turn 2 tool follow-up output is wrong: {turn2_final.content!r}")
    if "nemotron-nano-omni" not in turn3_final.content:
        raise TestFailure(f"turn 3 suffix-only output is wrong: {turn3_final.content!r}")

    return TestResult(
        name=name,
        ok=True,
        details={
            "turn1_output": normalize_text(turn1_final.content),
            "turn2_output": normalize_text(turn2_final.content),
            "turn3_output": normalize_text(turn3_final.content),
            "attach_count": len(attach),
            "attach": attach,
        },
    )


def run_audio_prefix_test(
    client: VllmClient, log_path: Path, audio_path: Path
) -> TestResult:
    name = "audio_input_turn_prefix_cache"
    offset = log_offset(log_path)
    cid = f"it-audio-{uuid.uuid4().hex[:8]}"
    salt = cid
    first_content = audio_content(
        audio_path,
        "Listen to the audio. In one short sentence, state what story subject "
        "the speaker requested, then remember it for the next turn.",
    )
    first_messages = [user(first_content)]
    first = client.chat(
        first_messages,
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=64,
    )
    second = (
        "From the previous audio turn, what creature was the speaker asking "
        "about? Answer with one lowercase word."
    )
    history = [user(first_content), assistant(first.content), user(second)]
    cached = client.chat(
        history,
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=32,
    )
    attach = assert_attached(log_path, offset, cid)
    if "unicorn" not in cached.content.lower():
        raise TestFailure(
            f"audio cached follow-up did not identify unicorn: {cached.content!r}"
        )
    return TestResult(
        name=name,
        ok=True,
        details={
            "first_output": normalize_text(first.content),
            "followup_output": normalize_text(cached.content),
            "attach": attach,
        },
    )


def run_audio_suffix_only_test(
    client: VllmClient, log_path: Path, audio_path: Path
) -> TestResult:
    name = "audio_suffix_only_no_old_media_resend"
    offset = log_offset(log_path)
    cid = f"it-suffix-audio-{uuid.uuid4().hex[:8]}"
    salt = cid
    first_content = audio_content(
        audio_path,
        "Listen to the audio. State the creature the speaker requested, then "
        "remember it for the next turn.",
    )
    first = client.chat(
        [user(first_content)],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=64,
    )
    followup = user(
        "From the previous audio turn, what creature was mentioned? Answer with "
        "one lowercase word."
    )
    cached = client.chat(
        [assistant(first.content), followup],
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=32,
        extra_payload={"conversation_require_cache": True},
    )
    attach = assert_attached(log_path, offset, cid)
    if "unicorn" not in cached.content.lower():
        raise TestFailure(
            "audio suffix-only follow-up did not identify unicorn: "
            f"{cached.content!r}"
        )
    return TestResult(
        name=name,
        ok=True,
        details={
            "first_output": normalize_text(first.content),
            "followup_output": normalize_text(cached.content),
            "second_request_audio_parts": 0,
            "attach": attach,
        },
    )


def long_prefix_text() -> str:
    facts = [
        "This is a deterministic ledger for latency testing.",
        "The project codename is HARBOR-SIGNAL.",
        "The routing city is Lisbon.",
        "The checksum color is teal.",
        "The operator phrase is quiet lantern.",
    ]
    for idx in range(1, 80):
        facts.append(
            f"Ledger row {idx:02d}: sample={idx * 17}, marker=alpha-{idx % 9}, "
            f"note=retain row order and ignore decoy-{(idx * 3) % 11}."
        )
    return (
        "Remember the following ledger for subsequent questions. Reply exactly READY.\n"
        + "\n".join(facts)
    )


def median(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def run_perf_test(client: VllmClient, log_path: Path, repeats: int) -> TestResult:
    name = "ttft_tps_cached_vs_uncached"
    offset = log_offset(log_path)
    setup = long_prefix_text()
    question = (
        "Using the remembered ledger, answer in one concise sentence with the "
        "project codename, routing city, checksum color, and operator phrase."
    )
    cached_runs: list[ChatResult] = []
    uncached_runs: list[ChatResult] = []
    cached_conversation_ids: list[str] = []
    for _ in range(repeats):
        cid = f"it-perf-{uuid.uuid4().hex[:8]}"
        salt = cid
        first = client.chat(
            [user(setup)],
            conversation_id=cid,
            cache_salt=salt,
            max_tokens=16,
        )
        history = [user(setup), assistant(first.content), user(question)]
        cached_runs.append(
            client.stream_chat(
                history,
                conversation_id=cid,
                cache_salt=salt,
                max_tokens=80,
            )
        )
        cached_conversation_ids.append(cid)
    for _ in range(repeats):
        first = client.chat(
            [user(setup)],
            cache_salt=f"uncached-setup-{uuid.uuid4().hex}",
            max_tokens=16,
        )
        history = [user(setup), assistant(first.content), user(question)]
        uncached_runs.append(
            client.stream_chat(
                history,
                cache_salt=f"uncached-perf-{uuid.uuid4().hex}",
                max_tokens=80,
            )
        )

    log_text = read_log_from(log_path, offset)
    assert_no_server_errors(log_text)
    for cid in cached_conversation_ids:
        assert_no_attach_skips(log_path, offset, cid)
    attach_by_conversation = {
        cid: attach_lines(log_text, cid) for cid in cached_conversation_ids
    }
    missing_attach = [
        cid for cid, attach in attach_by_conversation.items() if not attach
    ]
    if missing_attach:
        raise TestFailure(
            "no conversation-cache attach log for perf conversation(s): "
            + ", ".join(missing_attach)
        )

    cached_ttft = median([r.ttft for r in cached_runs])
    uncached_ttft = median([r.ttft for r in uncached_runs])
    cached_tps = median([r.tps for r in cached_runs])
    uncached_tps = median([r.tps for r in uncached_runs])

    required = ["HARBOR-SIGNAL", "Lisbon", "teal", "quiet lantern"]
    for run in cached_runs + uncached_runs:
        norm = run.content.lower()
        for value in required:
            if value.lower() not in norm:
                raise TestFailure(
                    f"perf run output missing {value!r}: {run.content!r}"
                )
    if cached_ttft is None or uncached_ttft is None:
        raise TestFailure("could not measure TTFT from streamed responses")
    return TestResult(
        name=name,
        ok=True,
        details={
            "cached_ttft_s_median": cached_ttft,
            "uncached_ttft_s_median": uncached_ttft,
            "ttft_delta_s": uncached_ttft - cached_ttft,
            "cached_tps_median": cached_tps,
            "uncached_tps_median": uncached_tps,
            "cached_outputs": [normalize_text(r.content) for r in cached_runs],
            "uncached_outputs": [normalize_text(r.content) for r in uncached_runs],
            "attach_by_conversation": attach_by_conversation,
        },
    )


def run_long_conversation_test(
    client: VllmClient, log_path: Path, turns: int, memory_warn_mb: int
) -> TestResult:
    name = "long_multiturn_memory_refcount_smoke"
    offset = log_offset(log_path)
    cid = f"it-long-{uuid.uuid4().hex[:8]}"
    salt = cid
    before_mem = gpu_memory_mb()
    history = [
        user(
            "This is a long-turn stability test. Remember base word ORCHID. "
            "Reply only ACK."
        )
    ]
    first = client.chat(
        history,
        conversation_id=cid,
        cache_salt=salt,
        max_tokens=16,
    )
    history.append(assistant(first.content))
    outputs: list[str] = []
    for turn in range(1, turns + 1):
        history.append(
            user(
                f"Turn {turn}: use the remembered base word and answer exactly "
                f"ORCHID-{turn}."
            )
        )
        result = client.chat(
            history,
            conversation_id=cid,
            cache_salt=salt,
            max_tokens=32,
        )
        outputs.append(normalize_text(result.content))
        if f"ORCHID-{turn}".lower() not in result.content.lower():
            raise TestFailure(
                f"turn {turn} did not preserve prior context: {result.content!r}"
            )
        history.append(assistant(result.content))

    after_mem = gpu_memory_mb()
    log_text = read_log_from(log_path, offset)
    assert_no_server_errors(log_text)
    attach = attach_lines(log_text, cid)
    if len(attach) < turns:
        raise TestFailure(
            f"expected at least {turns} attach events, saw {len(attach)}"
        )

    memory_delta_mb: list[int] = []
    if before_mem and after_mem and len(before_mem) == len(after_mem):
        memory_delta_mb = [after - before for before, after in zip(before_mem, after_mem)]
        max_delta = max(memory_delta_mb)
        if max_delta > memory_warn_mb:
            raise TestFailure(
                f"GPU memory grew by {max_delta} MB, above {memory_warn_mb} MB"
            )

    return TestResult(
        name=name,
        ok=True,
        details={
            "turns": turns,
            "first_output": normalize_text(first.content),
            "last_outputs": outputs[-3:],
            "attach_count": len(attach),
            "last_attach": attach[-1] if attach else None,
            "gpu_memory_before_mb": before_mem,
            "gpu_memory_after_mb": after_mem,
            "gpu_memory_delta_mb": memory_delta_mb,
        },
    )


def run_test(fn, *args) -> TestResult:
    try:
        return fn(*args)
    except Exception as exc:
        return TestResult(name=getattr(fn, "__name__", "test"), ok=False, error=str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vllm-python", type=Path, default=DEFAULT_VLLM_PYTHON)
    parser.add_argument("--vllm-bin", type=Path, default=DEFAULT_VLLM_BIN)
    parser.add_argument("--vllm-source-dir", type=Path, default=DEFAULT_VLLM_SOURCE_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        default=DEFAULT_LIMIT_MM_PER_PROMPT,
    )
    parser.add_argument(
        "--allowed-local-media-path",
        default=DEFAULT_ALLOWED_LOCAL_MEDIA_PATH,
    )
    parser.add_argument(
        "--mm-encoder-attn-backend",
        default=DEFAULT_MM_ENCODER_ATTN_BACKEND,
    )
    parser.add_argument(
        "--skip-mm-profiling",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SKIP_MM_PROFILING,
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENFORCE_EAGER,
    )
    parser.add_argument(
        "--reasoning-parser",
        default=DEFAULT_REASONING_PARSER,
    )
    parser.add_argument(
        "--enable-auto-tool-choice",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_AUTO_TOOL_CHOICE,
    )
    parser.add_argument(
        "--tool-call-parser",
        default=DEFAULT_TOOL_CALL_PARSER,
    )
    parser.add_argument("--moe-backend", default=DEFAULT_MOE_BACKEND)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_PREFIX_CACHING,
    )
    parser.add_argument(
        "--mamba-cache-mode",
        default=DEFAULT_MAMBA_CACHE_MODE,
    )
    parser.add_argument(
        "--mamba-cache-dtype",
        default=DEFAULT_MAMBA_CACHE_DTYPE,
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        default=DEFAULT_MAMBA_SSM_CACHE_DTYPE,
    )
    parser.add_argument(
        "--mamba-backend",
        default=DEFAULT_MAMBA_BACKEND,
    )
    parser.add_argument(
        "--attention-backend",
        default=DEFAULT_ATTENTION_BACKEND,
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        default=DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--health-timeout", type=float, default=360.0)
    parser.add_argument("--perf-repeats", type=int, default=2)
    parser.add_argument("--long-turns", type=int, default=8)
    parser.add_argument("--memory-warn-mb", type=int, default=1024)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=ROOT / "conversation-prefix-cache-results.json",
    )
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument(
        "--reuse-server",
        action="store_true",
        help="Do not launch vLLM; run against the existing --base-url endpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.audio.exists() and not args.skip_audio:
        print(f"missing audio fixture: {args.audio}", file=sys.stderr)
        return 2

    client = VllmClient(args.base_url, args.model, args.timeout)
    vllm_proc: subprocess.Popen[bytes] | None = None
    vllm_log_file = None

    try:
        if args.reuse_server:
            client.wait_healthy(args.health_timeout)
        else:
            vllm_proc, vllm_log_file = start_vllm(args)
            print(
                "Started managed vLLM subprocess "
                f"pid={vllm_proc.pid}; log={args.log}"
            )
            client.wait_healthy_with_process(args.health_timeout, vllm_proc, args.log)

        tests: list[tuple[Any, tuple[Any, ...]]] = [
            (run_equivalence_test, (client, args.log)),
            (run_text_suffix_only_equivalence_test, (client, args.log)),
            (run_hard_mamba_context_test, (client, args.log)),
            (run_tool_multiturn_cache_test, (client, args.log)),
        ]
        if not args.skip_audio:
            tests.append((run_audio_prefix_test, (client, args.log, args.audio)))
            tests.append((run_audio_suffix_only_test, (client, args.log, args.audio)))
        tests.extend(
            [
                (run_perf_test, (client, args.log, args.perf_repeats)),
                (
                    run_long_conversation_test,
                    (client, args.log, args.long_turns, args.memory_warn_mb),
                ),
            ]
        )

        results = [run_test(fn, *fn_args) for fn, fn_args in tests]
        payload = {
            "base_url": args.base_url,
            "model": args.model,
            "managed_vllm": not args.reuse_server,
            "vllm_command": None if args.reuse_server else vllm_command(args),
            "vllm_log": str(args.log),
            "results": [result.__dict__ for result in results],
        }
        args.results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        for result in results:
            status = "PASS" if result.ok else "FAIL"
            print(f"{status} {result.name}")
            if result.error:
                print(f"  {result.error}")
            elif result.details:
                compact = json.dumps(result.details, sort_keys=True)
                if len(compact) > 1200:
                    compact = compact[:1200] + "...<truncated>"
                print(f"  {compact}")

        failed = [result for result in results if not result.ok]
        print(f"\nResults written to {args.results_json}")
        return 1 if failed else 0
    finally:
        if vllm_proc is not None:
            terminate_vllm(vllm_proc)
            print(f"Stopped managed vLLM subprocess pid={vllm_proc.pid}")
        if vllm_log_file is not None:
            vllm_log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
