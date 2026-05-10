#!/usr/bin/env python3
"""Run an audio-path fragility screening study with retention outputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import shutil
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
import requests
import soundfile

from generate_cartesia_audio_fixtures import (  # noqa: E402
    DEFAULT_ENV_FILES,
    _load_api_key,
    _synthesize_pcm,
    _write_wav,
)


ROOT = Path(__file__).resolve().parents[1]
REQUEST_TEMPLATE_PATH = (
    ROOT / "artifacts" / "nvidia-audio-repro-20260504" / "request_template.json"
)
SEED_AUDIO_DIR = ROOT / "artifacts" / "audio-resampling-pilot-20260504" / "samples"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "audio-path-fragility-study-20260504"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "nemotron_3_nano_omni"
DEFAULT_PROBE_DIR = Path("/tmp/vllm-audio-probe")
TARGET_AUDIO_SR = 16000
SOURCE_AUDIO_SR = 48000


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    topic: str
    is_creature: bool

    @property
    def article(self) -> str:
        return "an" if self.topic[0].lower() in "aeiou" else "a"

    @property
    def audio_prompt(self) -> str:
        return f"Tell me in one sentence about {self.article} {self.topic}."

    @property
    def assistant_response(self) -> str:
        article_cap = self.article.capitalize()
        return (
            f"{article_cap} {self.topic} is a simple example subject used for this "
            "regression study."
        )

    @property
    def recall_answer(self) -> str:
        return self.topic

    @property
    def topic_length(self) -> int:
        return len(self.topic)

    @property
    def vowel_count(self) -> int:
        return sum(1 for char in self.topic.lower() if char in "aeiou")

    @property
    def has_repeated_letters(self) -> bool:
        lowered = self.topic.lower()
        return len(set(lowered)) != len(lowered)


@dataclass(frozen=True)
class Behavior:
    kind: str
    value: str

    def encoded(self) -> str:
        return f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class TestFamily:
    family_id: str
    group: str
    description: str
    prompt_builder: Callable[[PilotCase], str]
    expected_builder: Callable[[PilotCase], Behavior]
    normalize_text: Callable[[str], str]


BASELINE_CASE_TOPICS: list[tuple[str, bool]] = [
    ("unicorn", True),
    ("dragon", True),
    ("phoenix", True),
    ("tiger", True),
    ("robot", False),
    ("castle", False),
    ("comet", False),
    ("dolphin", True),
    ("lantern", False),
    ("cactus", False),
    ("rocket", False),
    ("sparrow", True),
    ("volcano", False),
    ("elephant", True),
    ("igloo", False),
    ("octopus", True),
    ("astronaut", False),
    ("engine", False),
    ("orchid", False),
    ("umpire", False),
]

CREATURE_TOPICS: list[str] = [
    "antelope",
    "aardvark",
    "badger",
    "beaver",
    "bison",
    "camel",
    "cheetah",
    "cougar",
    "donkey",
    "falcon",
    "ferret",
    "gazelle",
    "gecko",
    "giraffe",
    "hamster",
    "heron",
    "hyena",
    "iguana",
    "jaguar",
    "koala",
    "lemur",
    "leopard",
    "lobster",
    "manatee",
    "meerkat",
    "narwhal",
    "newt",
    "orca",
    "otter",
    "panther",
    "pelican",
    "penguin",
    "quail",
    "rabbit",
    "raccoon",
    "reindeer",
    "salmon",
    "squid",
    "toucan",
    "turtle",
    "viper",
    "walrus",
    "wombat",
    "yak",
    "zebra",
]

NON_CREATURE_TOPICS: list[str] = [
    "anchor",
    "apron",
    "airport",
    "backpack",
    "balloon",
    "bridge",
    "bunker",
    "canyon",
    "carousel",
    "chimney",
    "compass",
    "diamond",
    "doorknob",
    "elevator",
    "fountain",
    "fortress",
    "garden",
    "glacier",
    "hammer",
    "harbor",
    "iceberg",
    "island",
    "jacket",
    "joystick",
    "keyboard",
    "kettle",
    "library",
    "lighthouse",
    "magnet",
    "mirror",
    "mountain",
    "nebula",
    "necklace",
    "notebook",
    "observatory",
    "passport",
    "planet",
    "pyramid",
    "quartz",
    "quarry",
    "radiator",
    "refinery",
    "river",
    "satellite",
    "saucepan",
    "staircase",
    "subway",
    "teapot",
    "telescope",
    "thunder",
    "tractor",
    "turbine",
    "umbrella",
    "uniform",
    "universe",
    "valley",
    "vineyard",
    "violin",
    "waterfall",
    "windmill",
    "window",
    "workshop",
    "xylophone",
    "yoyo",
    "zeppelin",
]


def _build_case_pool() -> list[PilotCase]:
    cases: list[PilotCase] = []
    seen: set[str] = set()
    for topic, is_creature in BASELINE_CASE_TOPICS:
        if topic in seen:
            continue
        seen.add(topic)
        cases.append(
            PilotCase(
                case_id=f"{len(cases) + 1:03d}-{topic}",
                topic=topic,
                is_creature=is_creature,
            )
        )
    for topic in CREATURE_TOPICS:
        if topic in seen:
            continue
        seen.add(topic)
        cases.append(
            PilotCase(
                case_id=f"{len(cases) + 1:03d}-{topic}",
                topic=topic,
                is_creature=True,
            )
        )
    for topic in NON_CREATURE_TOPICS:
        if topic in seen:
            continue
        seen.add(topic)
        cases.append(
            PilotCase(
                case_id=f"{len(cases) + 1:03d}-{topic}",
                topic=topic,
                is_creature=False,
            )
        )
    return cases


PILOT_CASES: list[PilotCase] = _build_case_pool()


def _tool_behavior(command: str) -> Behavior:
    return Behavior("tool", json.dumps({"code": command}, ensure_ascii=True))


def _normalize_plain_text(text: str) -> str:
    return text.strip()


def _normalize_json_only(text: str) -> str:
    raw = text.strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw
    return json.dumps(parsed, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _letters_math(case: PilotCase) -> str:
    return str(case.topic_length * 3 + 1)


def _first_last(case: PilotCase) -> str:
    return case.topic[0] + case.topic[-1]


def _vowel_count_plus_constant(case: PilotCase) -> str:
    return str(case.vowel_count + 4)


def _sorted_letters(case: PilotCase) -> str:
    return "".join(sorted(case.topic))


ALL_TEST_FAMILIES: list[TestFamily] = [
    TestFamily(
        family_id="tool_pwd",
        group="tool",
        description="Explicit required tool use for pwd.",
        prompt_builder=lambda case: (
            "You must use the bash tool now. Run exactly this command: pwd. "
            "Do not answer from memory or prior knowledge. Reply with only the tool stdout."
        ),
        expected_builder=lambda case: _tool_behavior("pwd"),
        normalize_text=_normalize_plain_text,
    ),
    TestFamily(
        family_id="tool_git_branch",
        group="tool",
        description="Explicit required tool use for git branch lookup.",
        prompt_builder=lambda case: (
            "You must use the bash tool now. Run exactly this command: git branch --show-current. "
            "Do not answer from memory or prior knowledge. Reply with only the tool stdout."
        ),
        expected_builder=lambda case: _tool_behavior("git branch --show-current"),
        normalize_text=_normalize_plain_text,
    ),
    TestFamily(
        family_id="tool_top_level_file_count",
        group="tool",
        description="Explicit required tool use for a local file count command.",
        prompt_builder=lambda case: (
            "You must use the bash tool now. Run exactly this command: "
            "find . -maxdepth 1 -type f | wc -l. "
            "Do not answer from memory or prior knowledge. Reply with only the tool stdout."
        ),
        expected_builder=lambda case: _tool_behavior("find . -maxdepth 1 -type f | wc -l"),
        normalize_text=_normalize_plain_text,
    ),
    TestFamily(
        family_id="text_uppercase_topic",
        group="control",
        description="Strict uppercase-only recall.",
        prompt_builder=lambda case: (
            "What topic did I mention in the previous audio? Reply with the word in "
            "UPPERCASE only. No punctuation."
        ),
        expected_builder=lambda case: Behavior("text", case.topic.upper()),
        normalize_text=_normalize_plain_text,
    ),
    TestFamily(
        family_id="text_json_topic",
        group="control",
        description="JSON-only structured recall.",
        prompt_builder=lambda case: (
            "Reply with JSON only and no surrounding prose. Use exactly one object with one key "
            'named "topic" whose value is the topic from the previous audio.'
        ),
        expected_builder=lambda case: Behavior(
            "text",
            json.dumps(
                {"topic": case.topic},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
        normalize_text=_normalize_json_only,
    ),
    TestFamily(
        family_id="text_letters_times_three_plus_one",
        group="strict",
        description="Digits-only arithmetic derived from the recalled topic length.",
        prompt_builder=lambda case: (
            "Take the number of letters in the topic from the previous audio, multiply it by 3, "
            "add 1, and reply with digits only."
        ),
        expected_builder=lambda case: Behavior("text", _letters_math(case)),
        normalize_text=_normalize_plain_text,
    ),
    TestFamily(
        family_id="text_first_and_last_letter_only",
        group="strict",
        description="Reply with just the first and last letters of the topic.",
        prompt_builder=lambda case: (
            "What topic did I mention in the previous audio? Reply with exactly two lowercase letters: "
            "the first letter of the topic followed immediately by the last letter. No punctuation."
        ),
        expected_builder=lambda case: Behavior("text", _first_last(case)),
        normalize_text=_normalize_plain_text,
    ),
    TestFamily(
        family_id="text_vowel_count_plus_constant",
        group="strict",
        description="Digits-only vowel count plus a constant.",
        prompt_builder=lambda case: (
            "Count the vowels in the topic from the previous audio, add 4, and reply with digits only."
        ),
        expected_builder=lambda case: Behavior("text", _vowel_count_plus_constant(case)),
        normalize_text=_normalize_plain_text,
    ),
    TestFamily(
        family_id="text_alphabetical_sort_topic",
        group="strict",
        description="Alphabetically sorted letters of the topic.",
        prompt_builder=lambda case: (
            "Reply with the letters of the topic from the previous audio sorted in alphabetical order, "
            "all lowercase, with no spaces or punctuation."
        ),
        expected_builder=lambda case: Behavior("text", _sorted_letters(case)),
        normalize_text=_normalize_plain_text,
    ),
    TestFamily(
        family_id="text_yes_no_category",
        group="strict",
        description="Exact yes/no creature classification.",
        prompt_builder=lambda case: (
            "Is the topic from the previous audio an animal or mythical creature? "
            "Reply with exactly yes or no."
        ),
        expected_builder=lambda case: Behavior(
            "text", "yes" if case.is_creature else "no"
        ),
        normalize_text=_normalize_plain_text,
    ),
]

DEFAULT_ACTIVE_FAMILY_IDS = [
    "tool_pwd",
    "tool_git_branch",
    "tool_top_level_file_count",
    "text_uppercase_topic",
    "text_json_topic",
    "text_letters_times_three_plus_one",
    "text_first_and_last_letter_only",
    "text_vowel_count_plus_constant",
    "text_alphabetical_sort_topic",
    "text_yes_no_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--voice-id", default="71a7ad14-091c-4e8e-a314-022ece01c121")
    parser.add_argument("--tts-model", default="sonic-3")
    parser.add_argument("--cartesia-version", default="2024-11-13")
    parser.add_argument("--api-key-env", default="CARTESIA_API_KEY")
    parser.add_argument("--env-file", action="append", type=Path)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-artifacts", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--unit-manifest",
        type=Path,
        help="Optional JSON manifest of exact case-family units to run.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        choices=[family.family_id for family in ALL_TEST_FAMILIES],
        help="Optional subset of family ids to run.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=[case.case_id for case in PILOT_CASES],
        help="Optional subset of case ids to run.",
    )
    return parser.parse_args()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_f32(audio: np.ndarray) -> str:
    audio_f32 = np.asarray(audio, dtype=np.float32).reshape(-1)
    return hashlib.sha256(audio_f32.tobytes()).hexdigest()


def _resample_audio_pyav(audio: np.ndarray, *, orig_sr: float, target_sr: float) -> np.ndarray:
    orig_sr_int = int(round(orig_sr))
    target_sr_int = int(round(target_sr))
    if orig_sr_int == target_sr_int:
        return np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        return np.stack(
            [
                _resample_audio_pyav(channel, orig_sr=orig_sr, target_sr=target_sr)
                for channel in audio
            ],
            axis=0,
        )

    expected_len = int(math.ceil(audio.shape[-1] * target_sr_int / orig_sr_int))
    min_samples = 1024
    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) < min_samples:
        audio_f32 = np.pad(audio_f32, (0, min_samples - len(audio_f32)))
    audio_f32 = audio_f32.reshape(1, -1)

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr_int)
    frame = av.AudioFrame.from_ndarray(audio_f32, format="fltp", layout="mono")
    frame.sample_rate = orig_sr_int
    out_frames = resampler.resample(frame)
    out_frames.extend(resampler.resample(None))
    result = np.concatenate([frame.to_ndarray() for frame in out_frames], axis=1).squeeze(0)
    return result[:expected_len]


def _write_pcm16_wav(path: Path, audio: np.ndarray, *, sample_rate: int, mode: str) -> None:
    audio_1d = np.asarray(audio, dtype=np.float32).reshape(-1)
    scaled = np.clip(audio_1d, -1.0, 1.0) * 32767.0
    if mode == "trunc":
        pcm16 = np.trunc(scaled).astype(np.int16)
    elif mode == "round":
        pcm16 = np.round(scaled).astype(np.int16)
    else:
        raise ValueError(f"Unsupported PCM16 mode: {mode}")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _read_wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
    return {
        "path": str(path),
        "sha256_bytes": _sha256_bytes(path.read_bytes()),
        "sample_rate": sr,
        "channels": channels,
        "sample_width_bytes": width,
        "frames": frames,
        "duration_s": frames / sr,
    }


def _audio_file_to_data_url(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _load_request_template() -> dict[str, Any]:
    return json.loads(REQUEST_TEMPLATE_PATH.read_text(encoding="utf-8"))


def _build_payload(case: PilotCase, family: TestFamily, sample_path: Path, model: str) -> dict[str, Any]:
    payload = _load_request_template()
    payload["model"] = model
    payload["stream"] = False
    payload.pop("stream_options", None)
    payload["messages"][1]["content"][1]["audio_url"]["url"] = _audio_file_to_data_url(
        sample_path
    )
    payload["messages"][2]["content"] = case.assistant_response
    payload["messages"][3]["content"] = (
        "What topic did I mention in the previous audio? Answer with one word only."
    )
    payload["messages"][4]["content"] = case.recall_answer
    payload["messages"][5]["content"] = family.prompt_builder(case)
    return payload


def _run_payload(
    *,
    base_url: str,
    payload: dict[str, Any],
    probe_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, Path | None]:
    before = {path.name for path in probe_dir.glob("*.json")} if probe_dir else set()
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    message = body["choices"][0]["message"]
    summary = {
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "tool_calls": message.get("tool_calls"),
        "content": message.get("content"),
        "finish_reason": body["choices"][0].get("finish_reason"),
    }

    if probe_dir is None:
        return summary, None, None

    after = sorted(probe_dir.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    new_meta_files = [path for path in after if path.name not in before]
    if not new_meta_files:
        raise RuntimeError("Expected a new vLLM audio probe file, but none appeared")
    meta_path = new_meta_files[-1]
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return summary, metadata, meta_path


def _normalize_behavior(family: TestFamily, result: dict[str, Any]) -> Behavior:
    tool_calls = result.get("tool_calls") or []
    if tool_calls:
        return Behavior("tool", tool_calls[0]["function"]["arguments"])
    return Behavior("text", family.normalize_text(result.get("content") or ""))


def _summarize_variant(family: TestFamily, case: PilotCase, results: list[dict[str, Any]]) -> dict[str, Any]:
    behaviors = [_normalize_behavior(family, result) for result in results]
    encoded = [behavior.encoded() for behavior in behaviors]
    prompt_tokens = [result["prompt_tokens"] for result in results]
    counter = Counter(encoded)
    majority_behavior, majority_count = counter.most_common(1)[0]
    expected = family.expected_builder(case).encoded()
    return {
        "trials": len(results),
        "prompt_tokens": prompt_tokens,
        "behaviors": encoded,
        "behavior_counts": dict(sorted(counter.items())),
        "majority_behavior": majority_behavior,
        "majority_count": majority_count,
        "matches_expected_trials": sum(1 for value in encoded if value == expected),
        "expected_behavior": expected,
        "expected_match_rate": sum(1 for value in encoded if value == expected) / len(results),
    }


def _copy_probe_artifacts(meta_path: Path, case_dir: Path) -> tuple[Path, Path]:
    copied_meta = case_dir / "probe_source48.json"
    copied_wav = case_dir / "probe_source48.parsed.wav"
    shutil.copy2(meta_path, copied_meta)
    probe_meta = json.loads(copied_meta.read_text(encoding="utf-8"))
    shutil.copy2(Path(probe_meta["parsed_wav_path"]), copied_wav)
    probe_meta["parsed_wav_path"] = str(copied_wav)
    copied_meta.write_text(
        json.dumps(probe_meta, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return copied_meta, copied_wav


def _select_cases(case_ids: list[str] | None, max_cases: int | None) -> list[PilotCase]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in PILOT_CASES if case.case_id in wanted]
    else:
        selected = PILOT_CASES
    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def _select_families(family_ids: list[str] | None) -> list[TestFamily]:
    if family_ids:
        wanted = set(family_ids)
    else:
        wanted = set(DEFAULT_ACTIVE_FAMILY_IDS)
    return [family for family in ALL_TEST_FAMILIES if family.family_id in wanted]


def _select_case_family_units(
    *,
    unit_manifest: Path | None,
    case_ids: list[str] | None,
    max_cases: int | None,
    family_ids: list[str] | None,
) -> tuple[list[PilotCase], list[TestFamily], dict[str, list[TestFamily]], list[dict[str, Any]]]:
    case_by_id = {case.case_id: case for case in PILOT_CASES}
    family_by_id = {family.family_id: family for family in ALL_TEST_FAMILIES}

    if unit_manifest is None:
        cases = _select_cases(case_ids, max_cases)
        families = _select_families(family_ids)
        return (
            cases,
            families,
            {case.case_id: families for case in cases},
            [],
        )

    raw_units = json.loads(unit_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw_units, list):
        raise RuntimeError(f"Unit manifest {unit_manifest} must contain a JSON list")

    units_meta: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_family_ids: set[str] = set()
    ordered_case_ids: list[str] = []
    ordered_family_ids: list[str] = []
    case_family_map: dict[str, list[TestFamily]] = {}

    for idx, raw_unit in enumerate(raw_units):
        if not isinstance(raw_unit, dict):
            raise RuntimeError(f"Unit manifest entry {idx} in {unit_manifest} must be an object")
        case_id = raw_unit.get("case_id")
        family_id = raw_unit.get("family_id")
        if case_id not in case_by_id:
            raise RuntimeError(f"Unknown case_id {case_id!r} in {unit_manifest}")
        if family_id not in family_by_id:
            raise RuntimeError(f"Unknown family_id {family_id!r} in {unit_manifest}")

        if case_id not in seen_case_ids:
            ordered_case_ids.append(case_id)
            seen_case_ids.add(case_id)
        if family_id not in seen_family_ids:
            ordered_family_ids.append(family_id)
            seen_family_ids.add(family_id)

        case_family_map.setdefault(case_id, []).append(family_by_id[family_id])
        units_meta.append(raw_unit)

    cases = [case_by_id[case_id] for case_id in ordered_case_ids]
    families = [family_by_id[family_id] for family_id in ordered_family_ids]
    return cases, families, case_family_map, units_meta


def _ensure_source_audio(
    *,
    case: PilotCase,
    case_dir: Path,
    overwrite_audio: bool,
    api_key: str,
    voice_id: str,
    tts_model: str,
    cartesia_version: str,
) -> Path:
    source48_path = case_dir / "source48.wav"
    if source48_path.exists() and not overwrite_audio:
        return source48_path

    seed_path = SEED_AUDIO_DIR / case.case_id / "source48.wav"
    if seed_path.is_file() and not overwrite_audio:
        shutil.copy2(seed_path, source48_path)
        return source48_path

    pcm = _synthesize_pcm(
        api_key=api_key,
        transcript=case.audio_prompt,
        voice_id=voice_id,
        model=tts_model,
        sample_rate=SOURCE_AUDIO_SR,
        cartesia_version=cartesia_version,
        speed=None,
    )
    _write_wav(source48_path, pcm, sample_rate=SOURCE_AUDIO_SR)
    return source48_path


def _classify_case_family(family: TestFamily, per_variant: dict[str, dict[str, Any]]) -> dict[str, Any]:
    majority_behaviors = {
        variant_name: summary["majority_behavior"]
        for variant_name, summary in per_variant.items()
    }
    cross_variant_disagreement = len(set(majority_behaviors.values())) > 1
    within_variant_stochastic = any(
        len(summary["behavior_counts"]) > 1 for summary in per_variant.values()
    )
    prompt_token_drift = len(
        {
            tuple(summary["prompt_tokens"])
            for summary in per_variant.values()
        }
    ) > 1
    expected_rates = [summary["expected_match_rate"] for summary in per_variant.values()]
    expected_rate_spread = max(expected_rates) - min(expected_rates)
    all_expected = all(
        summary["matches_expected_trials"] == summary["trials"]
        for summary in per_variant.values()
    )

    retained = False
    label = "drop"
    reason = "uniformly_unhelpful"
    if family.group == "control" and all_expected and not cross_variant_disagreement and not within_variant_stochastic:
        retained = True
        label = "stable_control"
        reason = "perfect_control_behavior"
    elif cross_variant_disagreement:
        retained = True
        label = "fragile_cross_variant"
        reason = "different_majority_behavior_by_audio_variant"
    elif within_variant_stochastic:
        retained = True
        label = "fragile_stochastic"
        reason = "stochastic_behavior_within_variant"
    elif expected_rate_spread >= 0.25:
        retained = True
        label = "fragile_expected_rate_spread"
        reason = "expected_match_rate_spread_by_variant"
    elif family.group == "control":
        label = "drop_control_not_stable"
        reason = "control_family_failed_to_stay_stable"
    else:
        label = "drop_uniform"
        reason = "all_variants_behaved_the_same"

    return {
        "retained": retained,
        "retention_label": label,
        "retention_reason": reason,
        "cross_variant_disagreement": cross_variant_disagreement,
        "within_variant_stochastic": within_variant_stochastic,
        "prompt_token_drift": prompt_token_drift,
        "expected_rate_spread": expected_rate_spread,
        "majority_behaviors": majority_behaviors,
    }


def _compact_case_family(case: PilotCase, family: TestFamily, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "topic": case.topic,
        "topic_length": case.topic_length,
        "vowel_count": case.vowel_count,
        "has_repeated_letters": case.has_repeated_letters,
        "is_creature": case.is_creature,
        "family_id": family.family_id,
        "family_group": family.group,
        "retained": metrics["retained"],
        "retention_label": metrics["retention_label"],
        "retention_reason": metrics["retention_reason"],
        "cross_variant_disagreement": metrics["cross_variant_disagreement"],
        "within_variant_stochastic": metrics["within_variant_stochastic"],
        "prompt_token_drift": metrics["prompt_token_drift"],
        "expected_rate_spread": metrics["expected_rate_spread"],
        "majority_behaviors": metrics["majority_behaviors"],
    }


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Audio Path Fragility Study",
        "",
        "This study screens audio prompt / follow-up-family units for model fragility under",
        "semantically equivalent audio inputs. The goal is to retain units that are useful either",
        "as stable controls or as examples of cross-variant or stochastic instability.",
        "",
        "## Design",
        "",
        f"- Cases screened: `{report['summary']['cases_total']}`",
        f"- Families screened: `{report['summary']['families_total']}`",
        f"- Trials per variant: `{report['config']['trials']}`",
        "- Audio variants per case-family unit:",
        "  - `source48`",
        "  - `pyav_trunc16`",
        "  - `pyav_round16`",
        "",
        "## Screening Summary",
        "",
        f"- Retained case-family units: `{report['summary']['retained_case_families']}`",
        f"- Dropped case-family units: `{report['summary']['dropped_case_families']}`",
        f"- Offline PyAV float32 hash matched `vLLM` parsed float32 hash: "
        f"`{report['summary']['probe_matches_offline_pyav_float32']}/"
        f"{report['summary']['cases_total']}` cases",
        "",
        "## Family Summary",
        "",
    ]

    for family_id, family_summary in report["summary"]["families"].items():
        lines.extend(
            [
                f"### `{family_id}`",
                "",
                f"- Group: `{family_summary['group']}`",
                f"- Description: {family_summary['description']}",
                f"- Case-family units: `{family_summary['cases_total']}`",
                f"- Retained units: `{family_summary['retained_case_families']}`",
                f"- Cross-variant disagreement cases: `{family_summary['cross_variant_disagreement_cases']}`",
                f"- Within-variant stochastic cases: `{family_summary['within_variant_stochastic_cases']}`",
                "",
                "| Variant | Expected-match trials |",
                "| --- | ---: |",
            ]
        )
        for variant_name, variant_summary in family_summary["variant_totals"].items():
            lines.append(
                f"| `{variant_name}` | "
                f"`{variant_summary['matches_expected_trials']}/{variant_summary['trials']}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Retention Labels",
            "",
            "- `stable_control`: stable expected behavior across all variants",
            "- `fragile_cross_variant`: different majority behaviors across audio variants",
            "- `fragile_stochastic`: stochastic behavior inside at least one audio variant",
            "- `fragile_expected_rate_spread`: same majority behavior but materially different expected-match rates",
            "",
            "## Files",
            "",
            "- `manifest.json`: case and family metadata",
            "- `results.json`: full per-case-family results and summaries",
            "- `retained_case_families.json`: curated retained units",
            "- `dropped_case_families.json`: screened-but-dropped units",
            "- `samples/<case-id>/`: source and derived WAVs plus probe artifacts",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.probe_dir.is_dir():
        raise RuntimeError(
            f"Probe directory {args.probe_dir} does not exist. Start vLLM with "
            "VLLM_AUDIO_PROBE_DIR enabled before running this study."
        )

    cases, families, case_family_map, units_meta = _select_case_family_units(
        unit_manifest=args.unit_manifest,
        case_ids=args.cases,
        max_cases=args.max_cases,
        family_ids=args.families,
    )

    if args.out_dir.exists() and args.overwrite_artifacts:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_root = args.out_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    env_files = args.env_file or DEFAULT_ENV_FILES
    api_key = _load_api_key(args.api_key_env, env_files)

    manifest: dict[str, Any] = {
        "cases": [
            asdict(case)
            | {
                "article": case.article,
                "audio_prompt": case.audio_prompt,
                "topic_length": case.topic_length,
                "vowel_count": case.vowel_count,
                "has_repeated_letters": case.has_repeated_letters,
            }
            for case in cases
        ],
        "families": [
            {
                "family_id": family.family_id,
                "group": family.group,
                "description": family.description,
            }
            for family in families
        ],
        "selected_units": units_meta,
    }

    case_reports: list[dict[str, Any]] = []
    retained_units: list[dict[str, Any]] = []
    dropped_units: list[dict[str, Any]] = []
    probe_matches = 0

    family_variant_totals: dict[str, dict[str, Counter[str]]] = {
        family.family_id: {
            "source48": Counter(),
            "pyav_trunc16": Counter(),
            "pyav_round16": Counter(),
        }
        for family in families
    }
    family_cross_variant_disagreement: Counter[str] = Counter()
    family_within_variant_stochastic: Counter[str] = Counter()
    family_retained_counts: Counter[str] = Counter()
    family_case_totals: Counter[str] = Counter()
    family_retention_labels: dict[str, Counter[str]] = {
        family.family_id: Counter() for family in families
    }

    for case in cases:
        case_dir = samples_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        source48_path = _ensure_source_audio(
            case=case,
            case_dir=case_dir,
            overwrite_audio=args.overwrite_audio,
            api_key=api_key,
            voice_id=args.voice_id,
            tts_model=args.tts_model,
            cartesia_version=args.cartesia_version,
        )
        source_audio, source_sr = soundfile.read(
            source48_path, dtype="float32", always_2d=False
        )
        if isinstance(source_sr, np.generic):
            source_sr = int(source_sr)
        if np.asarray(source_audio).ndim > 1:
            source_audio = np.mean(np.asarray(source_audio), axis=1)

        offline_pyav_f32 = _resample_audio_pyav(
            np.asarray(source_audio, dtype=np.float32),
            orig_sr=source_sr,
            target_sr=TARGET_AUDIO_SR,
        )
        offline_pyav_f32_hash = _sha256_f32(offline_pyav_f32)

        trunc16_path = case_dir / "pyav_trunc16.wav"
        round16_path = case_dir / "pyav_round16.wav"
        _write_pcm16_wav(trunc16_path, offline_pyav_f32, sample_rate=TARGET_AUDIO_SR, mode="trunc")
        _write_pcm16_wav(round16_path, offline_pyav_f32, sample_rate=TARGET_AUDIO_SR, mode="round")

        probe_metadata = None
        probe_meta_path = None
        case_family_reports: list[dict[str, Any]] = []

        families_for_case = case_family_map[case.case_id]
        for family in families_for_case:
            family_case_totals[family.family_id] += 1

        for family_idx, family in enumerate(families_for_case):
            source48_results: list[dict[str, Any]] = []
            for trial_idx in range(args.trials):
                payload = _build_payload(case, family, source48_path, args.model)
                need_probe = family_idx == 0 and trial_idx == 0
                result, metadata, meta_path = _run_payload(
                    base_url=args.base_url,
                    payload=payload,
                    probe_dir=args.probe_dir if need_probe else None,
                )
                source48_results.append(result)
                if need_probe:
                    probe_metadata = metadata
                    probe_meta_path = meta_path

            trunc_results = [
                _run_payload(
                    base_url=args.base_url,
                    payload=_build_payload(case, family, trunc16_path, args.model),
                    probe_dir=None,
                )[0]
                for _ in range(args.trials)
            ]
            round_results = [
                _run_payload(
                    base_url=args.base_url,
                    payload=_build_payload(case, family, round16_path, args.model),
                    probe_dir=None,
                )[0]
                for _ in range(args.trials)
            ]

            per_variant = {
                "source48": _summarize_variant(family, case, source48_results),
                "pyav_trunc16": _summarize_variant(family, case, trunc_results),
                "pyav_round16": _summarize_variant(family, case, round_results),
            }
            metrics = _classify_case_family(family, per_variant)

            if metrics["cross_variant_disagreement"]:
                family_cross_variant_disagreement[family.family_id] += 1
            if metrics["within_variant_stochastic"]:
                family_within_variant_stochastic[family.family_id] += 1
            if metrics["retained"]:
                family_retained_counts[family.family_id] += 1

            family_retention_labels[family.family_id][metrics["retention_label"]] += 1
            compact = _compact_case_family(case, family, metrics)
            if metrics["retained"]:
                retained_units.append(compact)
            else:
                dropped_units.append(compact)

            for variant_name, summary in per_variant.items():
                family_variant_totals[family.family_id][variant_name]["trials"] += summary["trials"]
                family_variant_totals[family.family_id][variant_name]["matches_expected_trials"] += (
                    summary["matches_expected_trials"]
                )

            case_family_reports.append(
                {
                    "family_id": family.family_id,
                    "group": family.group,
                    "description": family.description,
                    "expected_behavior": family.expected_builder(case).encoded(),
                    **per_variant,
                    **metrics,
                }
            )

        if probe_metadata is None or probe_meta_path is None:
            raise RuntimeError(f"Missing source48 probe metadata for {case.case_id}")
        copied_probe_meta, copied_probe_wav = _copy_probe_artifacts(probe_meta_path, case_dir)
        probe_matches_offline_pyav = (
            probe_metadata["parsed_sha256_f32"] == offline_pyav_f32_hash
        )
        if probe_matches_offline_pyav:
            probe_matches += 1

        case_reports.append(
            {
                "case": asdict(case)
                | {
                    "article": case.article,
                    "topic_length": case.topic_length,
                    "vowel_count": case.vowel_count,
                    "has_repeated_letters": case.has_repeated_letters,
                },
                "audio_prompt": case.audio_prompt,
                "assistant_response": case.assistant_response,
                "files": {
                    "source48": _read_wav_info(source48_path),
                    "pyav_trunc16": _read_wav_info(trunc16_path),
                    "pyav_round16": _read_wav_info(round16_path),
                    "probe_source48_meta": str(copied_probe_meta),
                    "probe_source48_parsed_wav": str(copied_probe_wav),
                },
                "offline_pyav_float32": {
                    "sample_rate": TARGET_AUDIO_SR,
                    "sha256_f32": offline_pyav_f32_hash,
                    "num_samples": int(np.asarray(offline_pyav_f32).reshape(-1).shape[-1]),
                },
                "probe_source48": {
                    **probe_metadata,
                    "parsed_wav_path": str(copied_probe_wav),
                    "meta_path": str(copied_probe_meta),
                    "matches_offline_pyav_float32": probe_matches_offline_pyav,
                },
                "families": case_family_reports,
            }
        )

    family_summary = {}
    for family in families:
        family_summary[family.family_id] = {
            "group": family.group,
            "description": family.description,
            "cases_total": family_case_totals[family.family_id],
            "retained_case_families": family_retained_counts[family.family_id],
            "cross_variant_disagreement_cases": family_cross_variant_disagreement[
                family.family_id
            ],
            "within_variant_stochastic_cases": family_within_variant_stochastic[
                family.family_id
            ],
            "retention_labels": dict(sorted(family_retention_labels[family.family_id].items())),
            "variant_totals": {
                variant_name: {
                    "trials": counter["trials"],
                    "matches_expected_trials": counter["matches_expected_trials"],
                }
                for variant_name, counter in family_variant_totals[family.family_id].items()
            },
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "trials": args.trials,
            "probe_dir": str(args.probe_dir),
            "out_dir": str(args.out_dir),
            "unit_manifest": str(args.unit_manifest) if args.unit_manifest else None,
            "tts_model": args.tts_model,
            "voice_id": args.voice_id,
            "cartesia_version": args.cartesia_version,
            "families": [family.family_id for family in families],
            "cases": [case.case_id for case in cases],
            "selected_units_total": len(units_meta) if units_meta else len(cases) * len(families),
        },
        "summary": {
            "cases_total": len(cases),
            "families_total": len(families),
            "probe_matches_offline_pyav_float32": probe_matches,
            "retained_case_families": len(retained_units),
            "dropped_case_families": len(dropped_units),
            "families": family_summary,
        },
        "cases": case_reports,
    }

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "retained_case_families.json").write_text(
        json.dumps(retained_units, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "dropped_case_families.json").write_text(
        json.dumps(dropped_units, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(args.out_dir / "README.md", report)

    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
