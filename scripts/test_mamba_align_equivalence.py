#!/usr/bin/env python3
"""Mamba block-alignment x conversation-cache coherence sweep.

For each target turn-2 num_tokens position relative to mamba block_size,
constructs a 2-turn conversation, runs it with and without the
conversation cache, and asserts byte-identical outputs.

Critical properties (an earlier version of this test missed these and
silently passed):

1. Turn 2 cache-on REQUIRES ``conversation_require_cache=True`` so the
   serving layer reconstructs the full prompt from the lease's
   committed_messages + the client's suffix. Without it, the conversation
   lease is acquired but never attached (prefix mismatch in the request
   payload), and the test reduces to "two unrelated 2-message prompts
   produced the same output" which is uninformative.

2. The bug zone for ``_mamba_block_aligned_split`` is ``num_tokens %
   block_size < gen_prompt_length``, i.e. turn 2's RESULTING full prompt
   length lands within ``gen_prompt_length`` of a block boundary. We
   compute turn 2's incremental token cost once and adjust turn 1's
   padding so that the resulting turn-2 num_tokens lands at the desired
   mod-block-size position.

3. Turn 2's user prompt is discriminating: it asks the model to recall
   a unique codeword embedded in turn 1's user message. If the cache
   path is broken in a way that hides turn-1 from the model, turn 2's
   answer will differ.

4. Per-case, after turn 2 runs, the script verifies that an
   "Attached conversation cache for {conv_id}" log line appears in the
   vLLM log between turn 1's response and turn 2's response. Without
   that line, the cache attach didn't happen and the case is reported
   as a TEST-METHODOLOGY FAILURE rather than a comparison pass.
"""

from __future__ import annotations

import argparse
import json
import os
import string
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from test_vllm_conversation_prefix_cache import (  # noqa: E402
    VllmClient,
    user,
    assistant,
    log_offset,
    read_log_from,
    attach_lines,
)


DEFAULT_BLOCK_SIZE = 4240
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "models" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
)


def _padding_lines(count: int) -> str:
    if count <= 0:
        return ""
    return "\n".join(
        f"Note {i:04d}: filler line for token-count padding only."
        for i in range(1, count + 1)
    )


def _render_len(tokenizer, messages, add_generation_prompt: bool) -> int:
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


@dataclass
class TurnTemplate:
    base_system: str
    user_t1_template: str   # contains {codeword} placeholder
    fake_assistant_t1: str  # used only for delta measurement
    user_t2: str


def measure_turn2_delta(
    tokenizer,
    template: TurnTemplate,
    *,
    sample_codeword: str,
    block_size: int,
) -> tuple[int, int]:
    """Return (delta_tokens, baseline_t2_num_tokens) at zero padding.

    delta_tokens = (turn 2's full prompt length with gen prompt) MINUS
    (turn 1's commit-render length without gen prompt) — i.e., how many
    tokens turn 2 ADDS to the committed prefix. This is invariant under
    padding the system prompt (both lengths grow by the same amount).
    """
    user_t1 = template.user_t1_template.format(codeword=sample_codeword)
    msgs_t1_committed = [
        {"role": "system", "content": template.base_system},
        {"role": "user", "content": user_t1},
    ]
    len_t1_committed = _render_len(
        tokenizer, msgs_t1_committed, add_generation_prompt=False
    )
    msgs_t2_full = msgs_t1_committed + [
        {"role": "assistant", "content": template.fake_assistant_t1},
        {"role": "user", "content": template.user_t2},
    ]
    len_t2_full = _render_len(tokenizer, msgs_t2_full, add_generation_prompt=True)
    return len_t2_full - len_t1_committed, len_t2_full


def build_padded_system(
    tokenizer,
    template: TurnTemplate,
    *,
    target_t2_num_tokens: int,
    sample_codeword: str,
) -> tuple[str, int, int]:
    """Iterate padding lines until turn 2's full prompt length == target.

    Returns (system_content, turn1_committed_token_count, turn2_num_tokens).
    """

    def t2_len(pad_lines: int) -> int:
        system_content = template.base_system + (
            "\n" + _padding_lines(pad_lines) if pad_lines else ""
        )
        msgs_t2 = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": template.user_t1_template.format(codeword=sample_codeword)},
            {"role": "assistant", "content": template.fake_assistant_t1},
            {"role": "user", "content": template.user_t2},
        ]
        return _render_len(tokenizer, msgs_t2, add_generation_prompt=True)

    # Binary search on n_pad_lines.
    low, high = 0, 64
    if t2_len(low) > target_t2_num_tokens:
        raise RuntimeError(
            f"target_t2_num_tokens={target_t2_num_tokens} is below the minimum "
            f"({t2_len(0)} with no padding)."
        )
    while t2_len(high) < target_t2_num_tokens:
        high *= 2
        if high > 1_000_000:
            raise RuntimeError("padding search overflowed")
    while low < high:
        mid = (low + high) // 2
        if t2_len(mid) >= target_t2_num_tokens:
            high = mid
        else:
            low = mid + 1
    n_pad = low
    # We may have overshot. Try trimming the LAST padding line character-wise
    # to land exactly on target_t2_num_tokens.
    longest_line = "Note 0000: filler line for token-count padding only."
    overshoot = t2_len(n_pad) - target_t2_num_tokens
    if overshoot == 0:
        system_content = template.base_system + (
            "\n" + _padding_lines(n_pad) if n_pad else ""
        )
    else:
        # Walk down through trimmed last-line lengths.
        best_system: str | None = None
        for trim in range(len(longest_line) + 1):
            if n_pad == 0:
                trimmed_lines = ""
            else:
                trimmed_last = longest_line[: len(longest_line) - trim] if trim else longest_line
                head = _padding_lines(n_pad - 1)
                trimmed_lines = head + ("\n" if head else "") + trimmed_last
            candidate = template.base_system + ("\n" + trimmed_lines if trimmed_lines else "")
            msgs_t2 = [
                {"role": "system", "content": candidate},
                {"role": "user", "content": template.user_t1_template.format(codeword=sample_codeword)},
                {"role": "assistant", "content": template.fake_assistant_t1},
                {"role": "user", "content": template.user_t2},
            ]
            if _render_len(tokenizer, msgs_t2, add_generation_prompt=True) == target_t2_num_tokens:
                best_system = candidate
                break
        system_content = best_system if best_system is not None else (
            template.base_system + ("\n" + _padding_lines(n_pad) if n_pad else "")
        )

    # Measure turn-1 committed length with the chosen system.
    msgs_t1_committed = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": template.user_t1_template.format(codeword=sample_codeword)},
    ]
    len_t1_committed = _render_len(
        tokenizer, msgs_t1_committed, add_generation_prompt=False
    )
    msgs_t2 = msgs_t1_committed + [
        {"role": "assistant", "content": template.fake_assistant_t1},
        {"role": "user", "content": template.user_t2},
    ]
    len_t2_full = _render_len(tokenizer, msgs_t2, add_generation_prompt=True)
    return system_content, len_t1_committed, len_t2_full


def run_case(
    client: VllmClient,
    tokenizer,
    *,
    template: TurnTemplate,
    target_t2_num_tokens: int,
    block_size: int,
    max_tokens: int,
    log_path: Path,
) -> dict[str, Any]:
    codeword = "EMBER-" + "".join(
        uuid.uuid4().hex[:6].upper()[i] for i in range(6)
    )  # unique per case
    # Ensure the codeword tokenization length is invariant for our search.
    system_content, t1_committed_len, t2_num_tokens = build_padded_system(
        tokenizer,
        template,
        target_t2_num_tokens=target_t2_num_tokens,
        sample_codeword=codeword,
    )
    user_t1 = template.user_t1_template.format(codeword=codeword)
    msgs_t1 = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_t1},
    ]
    msgs_t2_suffix = [
        assistant(template.fake_assistant_t1),  # placeholder; overwritten below
        user(template.user_t2),
    ]

    # === Cache-on path ===
    conv_id = f"align-{uuid.uuid4().hex[:8]}"
    t1_cached = client.chat(
        msgs_t1, conversation_id=conv_id, max_tokens=max_tokens,
    )
    # Use the actual assistant_t1 from cache-on turn 1 as the suffix's
    # assistant message — turn 2's render then reflects the real conversation.
    msgs_t2_suffix = [assistant(t1_cached.content), user(template.user_t2)]
    pre_t2_offset = log_offset(log_path)
    t2_cached = client.chat(
        msgs_t2_suffix,
        conversation_id=conv_id,
        max_tokens=max_tokens,
        extra_payload={"conversation_require_cache": True},
    )
    post_t2_log = read_log_from(log_path, pre_t2_offset)
    attaches = attach_lines(post_t2_log, conv_id)

    # === Cache-off path ===
    t1_uncached = client.chat(msgs_t1, conversation_id=None, max_tokens=max_tokens)
    full_history = msgs_t1 + [assistant(t1_uncached.content), user(template.user_t2)]
    t2_uncached = client.chat(full_history, conversation_id=None, max_tokens=max_tokens)

    return {
        "target_t2_num_tokens": target_t2_num_tokens,
        "actual_t2_num_tokens": t2_num_tokens,
        "mod_block_size": t2_num_tokens % block_size,
        "t1_committed_len": t1_committed_len,
        "delta_tokens": t2_num_tokens - t1_committed_len,
        "codeword": codeword,
        "t1_match": t1_cached.content == t1_uncached.content,
        "t2_match": t2_cached.content == t2_uncached.content,
        "t2_attach_count": len(attaches),
        "t2_attach_tokens": attaches[0]["tokens"] if attaches else None,
        "t1_cached": t1_cached.content,
        "t1_uncached": t1_uncached.content,
        "t2_cached": t2_cached.content,
        "t2_uncached": t2_uncached.content,
        "conv_id": conv_id,
    }


def default_targets(block_size: int, gen_prompt_len_estimate: int = 4) -> list[int]:
    """Pick turn-2 num_tokens positions that exercise every branch.

    For each of 3 block boundaries, generates positions in:
      * the bug zone: mod_block_size in [1..gen_prompt_len_estimate-1]
      * at boundary: mod_block_size == 0
      * just past gen_prompt_len: mod = gen_prompt_len + small
      * mid-block: ~ block_size / 2
      * near next boundary: mod ~ block_size - 200, -50, -5
    """
    targets: list[int] = []
    for multiplier in (1, 2, 3):
        boundary = multiplier * block_size
        # Bug zone (the historically-stuck shape):
        for offset in range(1, gen_prompt_len_estimate):
            targets.append(boundary + offset)
        # Just past gen_prompt_len (force-last-chunk fires)
        targets.append(boundary + gen_prompt_len_estimate + 1)
        targets.append(boundary + 50)
        # At boundary
        targets.append(boundary)
        # Near next boundary (force-last-chunk fires)
        targets.extend([boundary + block_size - 200, boundary + block_size - 50, boundary + block_size - 5])
        # Mid block
        targets.append(boundary + block_size // 2)
    return sorted(set(targets))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "NEMOTRON_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NEMOTRON_VLLM_MODEL", "nemotron_3_nano_omni"),
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("NEMOTRON_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
    )
    parser.add_argument(
        "--log-path",
        default=os.environ.get(
            "NEMOTRON_VLLM_LOG",
            str(REPO_ROOT / "logs" / "rtx5090-vllm.log"),
        ),
        help="vLLM log file path; the test reads it to verify attach events.",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--targets", type=int, nargs="*", default=None,
        help="Override the default sweep with explicit turn-2 num_tokens targets.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--results-json", default=None,
        help="Optional path to dump full results as JSON.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    client = VllmClient(args.base_url, args.model, args.timeout)
    client.wait_healthy(15.0)
    log_path = Path(args.log_path)

    template = TurnTemplate(
        base_system=(
            "You are a deterministic test assistant. Below this line are 'Note' "
            "lines that exist purely to pad the prompt to a specific length. "
            "IGNORE them entirely; do not reference, summarize, or quote them. "
            "Follow each user message's instruction exactly."
        ),
        user_t1_template=(
            "Remember this codeword: {codeword}. Reply with the single token "
            "ACK and nothing else."
        ),
        fake_assistant_t1="ACK",
        user_t2=(
            "What was the codeword? Reply with exactly the codeword and "
            "nothing else (uppercase, including the dash)."
        ),
    )

    # Sanity check that codeword tokenization length is invariant
    # across uuid-generated codewords (chars are uppercase + dash; we
    # pad in the same character set to keep tokenization stable).
    delta_a, _ = measure_turn2_delta(
        tokenizer, template, sample_codeword="EMBER-AAAAAA", block_size=args.block_size,
    )
    delta_b, _ = measure_turn2_delta(
        tokenizer, template, sample_codeword="EMBER-ZZZZZZ", block_size=args.block_size,
    )
    if delta_a != delta_b:
        print(
            f"WARNING: turn-2 delta varies with codeword content "
            f"(EMBER-AAAAAA: {delta_a}, EMBER-ZZZZZZ: {delta_b}). The test "
            f"will still run but expected num_tokens may be slightly off."
        )

    targets = args.targets if args.targets else default_targets(args.block_size)

    print(
        f"Running {len(targets)} cases against {args.base_url} "
        f"(model={args.model}, block_size={args.block_size})"
    )
    print(
        f"{'target':>7} {'actual':>7} {'mod':>5}  t1   t2   attach"
    )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for target in targets:
        try:
            r = run_case(
                client, tokenizer,
                template=template,
                target_t2_num_tokens=target,
                block_size=args.block_size,
                max_tokens=args.max_tokens,
                log_path=log_path,
            )
            results.append(r)
            ok = r["t1_match"] and r["t2_match"] and r["t2_attach_count"] >= 1
            tag = "PASS" if ok else "FAIL"
            print(
                f"{tag} target={target:6d} actual={r['actual_t2_num_tokens']:6d} "
                f"mod={r['mod_block_size']:5d} t1={'Y' if r['t1_match'] else 'N'} "
                f"t2={'Y' if r['t2_match'] else 'N'} "
                f"attach_count={r['t2_attach_count']} "
                f"attach_tokens={r['t2_attach_tokens']}"
            )
            if not ok:
                failures.append(r)
        except Exception as exc:
            err = {"target_t2_num_tokens": target, "error": str(exc)}
            results.append(err)
            failures.append(err)
            print(f"ERROR target={target}: {exc}")

    n_total = len(results)
    n_pass = n_total - len(failures)
    print(f"\n{n_pass}/{n_total} cases pass.")

    if args.results_json:
        Path(args.results_json).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.results_json}")

    if failures:
        print("\nFailure details:")
        for f in failures:
            print(json.dumps(f, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
