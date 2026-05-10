#!/usr/bin/env python3
"""Generate Pipecat SmallWebRTC and vLLM path WAVs from a 48 kHz source WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_frontend_path_tools import (
    PIPECAT_FRAME_MS,
    TARGET_SR,
    generate_path_wavs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_wav", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--basename", default=None)
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    parser.add_argument("--frame-ms", type=int, default=PIPECAT_FRAME_MS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated = generate_path_wavs(
        args.source_wav,
        args.out_dir,
        basename=args.basename,
        target_sr=args.target_sr,
        frame_ms=args.frame_ms,
    )
    print(
        json.dumps(
            {
                "source_wav": str(generated.source_wav),
                "pipecat_wav": str(generated.pipecat_wav),
                "vllm_wav": str(generated.vllm_wav),
                "metadata_json": str(generated.metadata_json),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
