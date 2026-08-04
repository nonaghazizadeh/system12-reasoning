#!/usr/bin/env python3
"""Download and materialize the exact benchmark splits used in the paper.

Third-party data is intentionally not vendored.  Sources are pinned to immutable
revisions and a local manifest records file hashes after preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


ROLE_PLAY_REVISION = "43e73ed38135768eeb7c772ac979760a7ca572b5"
AGIEVAL_REVISION = "84ab72d94318290aad2e4ec820d535a95a1f7552"
COM2SENSE_REVISION = "15864a7c0637b950b5f28dc3556f71be01133d47"
PIQA_REVISION = "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0"
SOCIAL_IQA_REVISION = "8835ceb9141d7896d9d968634a9b21ae440e3ec5"

EXPECTED_COUNTS = {
    "multiarith": 600,
    "gsm8k": 1319,
    "addsub": 395,
    "aqua": 254,
    "singleeq": 508,
    "svamp": 1000,
    "agieval": 1000,
    "coin_flip": 500,
    "last_letters": 500,
    "commonsensqa": 1221,
    "strategyqa": 2290,
    "piqa": 1838,
    "siqa": 1954,
    "com2sense": 782,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/benchmark")
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing benchmark directory"
    )
    return parser.parse_args()


def clone_at_revision(url: str, revision: str, destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", url, str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--quiet", revision],
        check=True,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_json(url: str) -> object:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_huggingface_splits(output_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before preparing benchmarks") from exc

    piqa = load_dataset(
        "ybisk/piqa",
        revision=PIQA_REVISION,
        split="validation",
        trust_remote_code=True,
    )
    write_jsonl(output_dir / "PIQA/validation.jsonl", [dict(row) for row in piqa])

    social_iqa = load_dataset(
        "allenai/social_i_qa",
        revision=SOCIAL_IQA_REVISION,
        split="validation",
        trust_remote_code=True,
    )
    write_jsonl(
        output_dir / "SocialIQA/validation.jsonl",
        [dict(row) for row in social_iqa],
    )


def prepare_agieval(output_dir: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/ruixiangcui/AGIEval/"
        f"{AGIEVAL_REVISION}/data/v1/math.jsonl"
    )
    with urllib.request.urlopen(url) as response:
        source = [json.loads(line) for line in response if line.strip()]
    converted = [
        {
            "query": f"Q: {item['question']}\nA: The answer is",
            "answer": item["answer"],
        }
        for item in source
    ]
    path = output_dir / "agieval/output.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n")


def prepare_com2sense(output_dir: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/PlusLabNLP/Com2Sense/"
        f"{COM2SENSE_REVISION}/data/dev.json"
    )
    path = output_dir / "Com2Sense/dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(download_json(url), indent=2) + "\n")


def build_manifest(output_dir: Path) -> dict:
    # Import after the output exists so this script can provide a useful error.
    from system12.benchmarks import BENCHMARKS, load_benchmark

    benchmarks = {}
    for name, expected in EXPECTED_COUNTS.items():
        examples = load_benchmark(name, data_root=output_dir)
        if len(examples) != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {len(examples)}")
        path = output_dir / BENCHMARKS[name].relative_path
        benchmarks[name] = {
            "examples": len(examples),
            "path": str(path.relative_to(output_dir)),
            "sha256": sha256(path),
        }
    return {
        "source_revisions": {
            "NKU-HLT/Role-Play-Prompting": ROLE_PLAY_REVISION,
            "ruixiangcui/AGIEval": AGIEVAL_REVISION,
            "PlusLabNLP/Com2Sense": COM2SENSE_REVISION,
            "ybisk/piqa": PIQA_REVISION,
            "allenai/social_i_qa": SOCIAL_IQA_REVISION,
        },
        "benchmarks": benchmarks,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise SystemExit(
                f"{output_dir} is not empty; pass --force to replace generated data"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="system12-benchmarks-") as temporary:
        checkout = Path(temporary) / "role-play-prompting"
        clone_at_revision(
            "https://github.com/NKU-HLT/Role-Play-Prompting.git",
            ROLE_PLAY_REVISION,
            checkout,
        )
        shutil.copytree(checkout / "dataset", output_dir, dirs_exist_ok=True)

    prepare_huggingface_splits(output_dir)
    prepare_agieval(output_dir)
    prepare_com2sense(output_dir)
    manifest = build_manifest(output_dir)
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Prepared and verified 14 benchmarks in {output_dir}")


if __name__ == "__main__":
    main()
