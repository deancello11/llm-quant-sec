"""Builds the prompt list for each dataset. Raw datasets are downloaded to data/cache (gitignored), never redistributed.

Every download is fingerprinted in data/DATASET_VERSIONS.json so the exact data used is on record.
"""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import asdict, dataclass

import pandas as pd
import yaml

from .common import DATA_DIR, ROOT, now_iso

CACHE = DATA_DIR / "cache"
VERSIONS = DATA_DIR / "DATASET_VERSIONS.json"


@dataclass
class PromptRow:
    prompt_id: str
    dataset: str
    behavior_id: str     # the cluster for bootstrapping: the same behavior across direct and template variants
    variant: str         # "direct" or "tmpl_<name>"
    category: str
    prompt: str
    harmful: bool

    def to_dict(self):
        return asdict(self)


def _record_version(name: str, info: dict) -> None:
    data = json.loads(VERSIONS.read_text()) if VERSIONS.exists() else {}
    old = data.get(name)
    if old and old.get("sha256") and old["sha256"] != info.get("sha256"):
        print(f"WARNING: dataset '{name}' content changed since it was first recorded "
              f"({old['sha256'][:10]} -> {info['sha256'][:10]}). Results are not comparable across the two.")
    data[name] = {**(old or {}), **info, "first_recorded": (old or {}).get("first_recorded", now_iso())}
    VERSIONS.write_text(json.dumps(data, indent=2))


def _load_url_csv(name: str, spec: dict) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.csv"
    url = spec["url"].format(ref=spec["ref"])
    if not path.exists():
        if spec["ref"] == "main":
            print(f"WARNING: dataset '{name}' is not pinned to a commit (ref: main). Pin it in configs/experiment.yaml.")
        urllib.request.urlretrieve(url, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    _record_version(name, {"url": url, "ref": spec["ref"], "sha256": sha})
    return pd.read_csv(path)


def _load_hf(name: str, spec: dict) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.csv"
    if not path.exists():
        from datasets import load_dataset
        ds = load_dataset(spec["repo"], spec["config"], split=spec["split"], revision=spec.get("revision"))
        ds.to_pandas().to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    info = {"repo": spec["repo"], "split": spec["split"], "revision": spec.get("revision"), "sha256": sha}
    try:
        from huggingface_hub import dataset_info
        info["hub_head_sha_at_download"] = dataset_info(spec["repo"]).sha
    except Exception:
        pass
    _record_version(name, info)
    return pd.read_csv(path)


def load_templates(cfg: dict) -> dict[str, str]:
    with open(ROOT / cfg["templates_file"], encoding="utf-8") as f:
        return yaml.safe_load(f)["templates"]


def build_prompts(cfg: dict, dataset: str, pilot: bool = False) -> list[PromptRow]:
    """All prompts for one dataset: direct requests, plus template-wrapped copies where configured."""
    spec = cfg["datasets"][dataset]
    df = _load_url_csv(dataset, spec) if spec["kind"] == "url_csv" else _load_hf(dataset, spec)
    for col, val in (spec.get("filter") or {}).items():
        df = df[df[col] == val]
    df = df.reset_index(drop=True)

    templates = load_templates(cfg) if "templates" in spec["variants"] else {}
    rows: list[PromptRow] = []
    for i, r in df.iterrows():
        bid = str(r[spec["id_col"]]) if "id_col" in spec else f"{i:04d}"
        text = str(r[spec["text_col"]]).strip()
        cat = str(r.get(spec.get("category_col", ""), "")) if spec.get("category_col") else ""
        base = f"{dataset}:{bid}"
        rows.append(PromptRow(f"{base}::direct", dataset, base, "direct", cat, text, spec["harmful"]))
        for tname, tmpl in templates.items():
            rows.append(PromptRow(f"{base}::tmpl_{tname}", dataset, base, f"tmpl_{tname}", cat,
                                  tmpl.format(behavior=text).strip(), spec["harmful"]))

    if pilot:
        # Pilot: a seeded random sample of whole prompts. Same sample for every model and precision.
        n = cfg["pilot"]["n_prompts_per_dataset"]
        rng = random.Random(cfg["seed"])
        rows = rng.sample(rows, min(n, len(rows)))
    return rows
