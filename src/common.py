"""Paths, config loading, JSONL helpers and run metadata shared by every script.

Where this fits: the pipeline is quantize -> generate -> judge -> analyze, and every step imports from here.
Think of it as the shared toolbox: one place that knows where the folders are, how to read the config, and how
to save results safely. Nothing in this file runs a model.

JSONL means "JSON Lines": a text file with one small JSON record per line. It is easy to append to and easy to
recover if a run dies halfway.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import uuid

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
OUTPUTS_DIR = ROOT / "outputs"
JUDGMENTS_DIR = ROOT / "judgments"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
# GPTQ/AWQ checkpoints are big. On RunPod set QS_QUANT_DIR to a folder on the network volume.
QUANT_DIR = pathlib.Path(os.environ.get("QS_QUANT_DIR", ROOT / "quantized"))

# Library versions worth recording in each run's receipt (see env_info). Missing ones are simply skipped.
PACKAGES = [
    "torch", "transformers", "accelerate", "bitsandbytes", "gptqmodel", "auto-gptq", "autoawq",
    "optimum", "datasets", "huggingface-hub", "nvidia-ml-py", "numpy", "pandas", "scipy",
]


def load_config(path: pathlib.Path | str = CONFIG_PATH) -> dict:
    """Read the experiment settings (models, precisions, datasets, seeds...) from the YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """A short fingerprint of the config: if any setting changes, the fingerprint changes."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def now_iso() -> str:
    """The current local time as text, like 2025-01-31T14:05:09+0000, for timestamps in the records."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    """Append one record and force it to disk, so an interrupted pod loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        # flush pushes Python's buffer to the operating system, and fsync pushes the OS buffer to the disk.
        # Slower, but it is what makes the "safe to restart" promise real.
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: pathlib.Path) -> list[dict]:
    """Read every record from a JSONL file. A missing file counts as empty, and a half-written last line is skipped."""
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # the last line can be cut off if the process was killed mid-write
    return out


def cell_name(model: str, precision: str, dataset: str) -> str:
    """File name for one (model, precision, dataset) cell. Double underscores keep the three parts easy to split."""
    return f"{model}__{precision}__{dataset}.jsonl"


def _run(cmd: list[str]) -> str | None:
    """Run a small shell command and return its output, or None if it fails (e.g. git is not installed)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def env_info() -> dict:
    """A snapshot of the machine and software (Python, libraries, GPU, git commit) to store with each run."""
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_run(["git", "-C", str(ROOT), "status", "--porcelain"])),
    }
    # torch is imported here, not at the top, so the analysis scripts work on a laptop with no GPU libraries.
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def new_run_id() -> str:
    """A unique name for one run: the date and time, plus a few random characters in case two start together."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def expand(arg: str, all_values: list[str]) -> list[str]:
    """'all' -> every value. 'a,b' -> ['a', 'b']. Unknown names raise a clear error."""
    vals = all_values if arg == "all" else [v.strip() for v in arg.split(",") if v.strip()]
    bad = [v for v in vals if v not in all_values]
    if bad:
        raise SystemExit(f"Unknown value(s) {bad}. Valid: {all_values}")
    return vals
