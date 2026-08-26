"""Resolve the encoder revisions to pin, and print them as shell exports.

    python scripts/pin_encoder_revisions.py

Prints two export lines. Put them in the environment before caching features,
and record them in AMENDMENTS.md, because they are part of what the frozen
protocol means by "the encoder".

What a revision pins, and what it does not.

VGGT is a Hugging Face repository, and the revision is the commit of that
repository. The weights are files in it, so pinning the commit pins the bytes.
The local cache records which commit main resolved to at download time, in
models--facebook--VGGT-1B/refs/main, so the exact revision that produced an
existing cache is recoverable after the fact.

DINOv2 is a Torch Hub repository, and the ref pins the code, which is what
chooses the checkpoint URL on dl.fbaipublicfiles.com. It does not pin the bytes
at that URL. Torch Hub also does not record which commit it downloaded: it
extracts the zipball into a directory named after the ref, so a cache fetched
from "main" cannot tell you which main. This is why the fingerprint exists
beside the pin, and why the two are not redundant. The pin makes a checkpoint
retrievable; the fingerprint notices if the bytes behind it moved.

The consequence for ordering: pin first, then cache. A pin resolved today and a
cache built today agree by construction, and CACHE_VERSION 2 requires the caches
to be rebuilt anyway, so there is nothing to reconcile.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DINOV2_REPO = "https://github.com/facebookresearch/dinov2"
VGGT_REPO = "facebook/VGGT-1B"


def torch_home() -> Path:
    return Path(os.environ.get("TORCH_HOME") or Path.cwd() / "cache" / "torch")


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or Path.cwd() / "cache" / "huggingface")


def dinov2_ref() -> tuple[str, str]:
    """The commit main points at now, from GitHub. Returns (sha, how)."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", DINOV2_REPO, "HEAD"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return "", f"git ls-remote failed: {error}"
    if result.returncode != 0:
        return "", f"git ls-remote failed: {result.stderr.strip()[:200]}"
    line = result.stdout.split("\n", 1)[0].strip()
    if not line:
        return "", "git ls-remote returned nothing"
    return line.split()[0], "current HEAD of the default branch"


def vggt_revision() -> tuple[str, str]:
    """The cached revision if there is one, else the current one. Returns (sha, how)."""
    owner, name = VGGT_REPO.split("/")
    ref_file = hf_home() / "hub" / f"models--{owner}--{name}" / "refs" / "main"
    if ref_file.is_file():
        sha = ref_file.read_text(encoding="utf-8").strip()
        if sha:
            return sha, f"the revision already downloaded, from {ref_file}"
    try:
        from huggingface_hub import HfApi
    except ImportError:
        pass
    else:
        try:
            return HfApi().model_info(VGGT_REPO).sha, "current revision of main, from the hub"
        except Exception as error:  # noqa: BLE001 - any hub failure is the same answer here
            return "", f"hub lookup failed: {type(error).__name__}: {error}"
    # Same question without the library, so this runs outside the encode
    # environment. A public read of one model's metadata, nothing else.
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{VGGT_REPO}", timeout=60
        ) as response:
            sha = json.loads(response.read().decode("utf-8")).get("sha", "")
    except (urllib.error.URLError, OSError, ValueError) as error:
        return "", f"hub lookup failed: {type(error).__name__}: {error}"
    return sha, "current revision of main, from the hub api"


def main() -> int:
    print(f"TORCH_HOME  {torch_home()}")
    print(f"HF_HOME     {hf_home()}")
    print()

    dinov2, dinov2_how = dinov2_ref()
    vggt, vggt_how = vggt_revision()

    for label, sha, how in (
        ("DINOv2  (torch.hub facebookresearch/dinov2)", dinov2, dinov2_how),
        (f"VGGT    (hugging face {VGGT_REPO})", vggt, vggt_how),
    ):
        print(f"{label}\n  {sha or 'UNRESOLVED'}\n  {how}\n")

    if not (dinov2 and vggt):
        print("Could not resolve both revisions; nothing to pin.", file=sys.stderr)
        return 1

    print("Export these before caching features, and record them in AMENDMENTS.md:")
    print()
    print(f'export LOT_DINOV2_REVISION="{dinov2}"')
    print(f'export LOT_VGGT_REVISION="{vggt}"')
    print()
    print(
        "Then rebuild the caches. The fingerprint each cache records is only\n"
        "meaningful beside a pin: it says the weights changed, and the pin is\n"
        "what gets the old ones back."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
