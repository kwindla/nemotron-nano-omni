#!/usr/bin/env python3
"""Regenerate and optionally validate the shared vLLM conversation-cache patch."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "checkouts.lock.json"


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def load_checkout_info() -> tuple[str, Path, tuple[Path, ...]]:
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    checkout = data["checkouts"]["vllm-v0.20.0"]
    patch_relpath = checkout["common_patches"][0]
    platform_patches: list[Path] = []
    seen: set[Path] = set()
    for patch_list in checkout.get("platform_patches", {}).values():
        for patch_relpath_item in patch_list:
            patch_path = REPO_ROOT / patch_relpath_item
            if patch_path not in seen:
                seen.add(patch_path)
                platform_patches.append(patch_path)
    return checkout["commit"], REPO_ROOT / patch_relpath, tuple(platform_patches)


def materialize_common_source(
    vllm_dir: Path,
    platform_patches: tuple[Path, ...],
    dest: Path,
) -> None:
    run(["git", "clone", str(vllm_dir), str(dest)])
    run(["rsync", "-a", "--delete", "--exclude", ".git", f"{vllm_dir}/", f"{dest}/"])
    for patch_path in platform_patches:
        reverse_check = subprocess.run(
            ["git", "-C", str(dest), "apply", "--reverse", "--check", str(patch_path)],
            check=False,
            text=True,
            capture_output=True,
        )
        if reverse_check.returncode == 0:
            run(["git", "-C", str(dest), "apply", "--reverse", str(patch_path)])


def generate_patch(
    vllm_dir: Path,
    pinned_ref: str,
    patch_path: Path,
    platform_patches: tuple[Path, ...],
) -> None:
    if not (vllm_dir / ".git").exists():
        raise SystemExit(f"Missing vLLM checkout: {vllm_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        common_source = Path(tmpdir) / "vllm"
        materialize_common_source(vllm_dir, platform_patches, common_source)
        temp_index = Path(tmpdir) / "index"
        env = {**os.environ, "GIT_INDEX_FILE": str(temp_index)}
        run(["git", "-C", str(common_source), "read-tree", pinned_ref], env=env)
        run(["git", "-C", str(common_source), "add", "-A"], env=env)
        patch = run(
            ["git", "-C", str(common_source), "diff", "--binary", "--cached", pinned_ref],
            env=env,
            capture_output=True,
        ).stdout
    patch_path.write_text(patch, encoding="utf-8")


def materialize_pristine_tree(vllm_dir: Path, pinned_ref: str, dest: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(vllm_dir), "archive", pinned_ref],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(dest)


def import_check(tree_dir: Path, python_bin: Path) -> None:
    modules = [
        "vllm.entrypoints.openai.conversation_cache",
        "vllm.v1.core.conversation_cache",
        "vllm.entrypoints.openai.chat_completion.serving",
        "vllm.v1.core.sched.scheduler",
    ]
    code = "\n".join(
        [
            "import importlib",
            *(f"importlib.import_module('{module}')" for module in modules),
            "print('import-check-ok')",
        ]
    )
    env = {**os.environ, "PYTHONPATH": str(tree_dir)}
    run([str(python_bin), "-c", code], env=env)


def validate_patch(
    vllm_dir: Path,
    pinned_ref: str,
    patch_path: Path,
    platform_patches: tuple[Path, ...],
    python_bin: Path | None,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tree_dir = Path(tmpdir) / "vllm"
        tree_dir.mkdir()
        materialize_pristine_tree(vllm_dir, pinned_ref, tree_dir)
        run(["git", "-C", str(tree_dir), "apply", "--check", str(patch_path)])
        run(["git", "-C", str(tree_dir), "apply", str(patch_path)])
        required_files = [
            tree_dir / "vllm/entrypoints/openai/conversation_cache.py",
            tree_dir / "vllm/v1/core/conversation_cache.py",
            tree_dir / "tests/entrypoints/openai/test_conversation_cache.py",
        ]
        missing = [str(path.relative_to(tree_dir)) for path in required_files if not path.exists()]
        if missing:
            raise SystemExit(
                "Patch applied but required files are still missing: "
                + ", ".join(missing)
            )
        for platform_patch in platform_patches:
            reverse_check = subprocess.run(
                ["git", "-C", str(tree_dir), "apply", "--reverse", "--check", str(platform_patch)],
                check=False,
                text=True,
                capture_output=True,
            )
            if reverse_check.returncode == 0:
                raise SystemExit(
                    "Common patch already includes platform patch changes from "
                    f"{platform_patch}"
                )
            run(["git", "-C", str(tree_dir), "apply", "--check", str(platform_patch)])
        if python_bin is not None:
            import_check(tree_dir, python_bin)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-dir",
        type=Path,
        default=REPO_ROOT / "vllm-v0.20.0",
        help="Local vLLM checkout to diff against the pinned upstream commit.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python interpreter to use for the post-apply import smoke test.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Only regenerate the patch; skip the pristine-tree apply/import check.",
    )
    args = parser.parse_args()

    pinned_ref, patch_path, platform_patches = load_checkout_info()
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    generate_patch(args.vllm_dir, pinned_ref, patch_path, platform_patches)
    if not args.no_validate:
        validate_patch(
            args.vllm_dir,
            pinned_ref,
            patch_path,
            platform_patches,
            args.python,
        )
    print(f"Wrote {patch_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
