"""Dataset registry and exact-match scoring for the 14 paper benchmarks."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    category: str
    relative_path: str
    expected_examples: int
    direct_answer_instruction: str
    answer_type: str


@dataclass(frozen=True)
class BenchmarkExample:
    prompt: str
    answer: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "multiarith": BenchmarkSpec(
        "multiarith", "arithmetic", "MultiArith/MultiArith.json", 600,
        "Therefore, the answer (arabic numerals) is", "number",
    ),
    "gsm8k": BenchmarkSpec(
        "gsm8k", "arithmetic", "grade-school-math/test.jsonl", 1319,
        "Therefore, the answer (arabic numerals) is", "number",
    ),
    "addsub": BenchmarkSpec(
        "addsub", "arithmetic", "AddSub/AddSub.json", 395,
        "Therefore, the answer (arabic numerals) is", "number",
    ),
    "aqua": BenchmarkSpec(
        "aqua", "arithmetic", "AQuA/test.json", 254,
        "Therefore, among A through E, the answer is", "choice_ae",
    ),
    "singleeq": BenchmarkSpec(
        "singleeq", "arithmetic", "SingleEq/questions.json", 508,
        "Therefore, the answer (arabic numerals) is", "number",
    ),
    "svamp": BenchmarkSpec(
        "svamp", "arithmetic", "SVAMP/SVAMP.json", 1000,
        "Therefore, the answer (arabic numerals) is", "number",
    ),
    "agieval": BenchmarkSpec(
        "agieval", "arithmetic", "agieval/output.json", 1000,
        "Therefore, the final answer is", "math",
    ),
    "coin_flip": BenchmarkSpec(
        "coin_flip", "symbolic", "coin_flip/coin_flip.json", 500,
        "Therefore, the answer (Yes or No) is", "yes_no",
    ),
    "last_letters": BenchmarkSpec(
        "last_letters", "symbolic", "last_letters/last_letters.json", 500,
        "Therefore, the final answer is", "word",
    ),
    "commonsensqa": BenchmarkSpec(
        "commonsensqa", "commonsense", "CommonsenseQA/dev_rand_split.jsonl", 1221,
        "Therefore, among A through E, the answer is", "choice_ae",
    ),
    "strategyqa": BenchmarkSpec(
        "strategyqa", "commonsense", "StrategyQA/task.json", 2290,
        "Therefore, the answer (Yes or No) is", "yes_no",
    ),
    "piqa": BenchmarkSpec(
        "piqa", "commonsense", "PIQA/validation.jsonl", 1838,
        "Therefore, among A and B, the answer is", "choice_ab",
    ),
    "siqa": BenchmarkSpec(
        "siqa", "commonsense", "SocialIQA/validation.jsonl", 1954,
        "Therefore, among A through C, the answer is", "choice_ac",
    ),
    "com2sense": BenchmarkSpec(
        "com2sense", "commonsense", "Com2Sense/dev.json", 782,
        "Therefore, the answer (TRUE or FALSE) is", "true_false",
    ),
}

ALIASES = {
    "csqa": "commonsensqa",
    "commonsenseqa": "commonsensqa",
    "socialiqa": "siqa",
    "social_i_qa": "siqa",
    "com2": "com2sense",
    "letter": "last_letters",
    "coin": "coin_flip",
}


def canonical_benchmark_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in BENCHMARKS:
        raise ValueError(
            f"unknown benchmark {name!r}; choose from {', '.join(BENCHMARKS)}"
        )
    return normalized


def benchmark_spec(name: str) -> BenchmarkSpec:
    return BENCHMARKS[canonical_benchmark_name(name)]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _format_choices(labels: list[str], choices: list[str]) -> str:
    return "Answer Choices:" + "".join(
        f" ({label}) {choice}" for label, choice in zip(labels, choices)
    )


def _strip_integer_decimal(answer: object) -> str:
    text = str(answer)
    return text[:-2] if text.endswith(".0") else text


def _load_aqua(path: Path, _: random.Random) -> list[BenchmarkExample]:
    examples = []
    for item in _read_jsonl(path):
        options = [re.sub(r"^\([A-E]\)\s*", "", option) for option in item["options"]]
        prompt = f"{item['question'].strip()} {_format_choices(list('ABCDE'), options)}"
        examples.append(BenchmarkExample(prompt, item["correct"]))
    return examples


def _load_gsm8k(path: Path, _: random.Random) -> list[BenchmarkExample]:
    return [
        BenchmarkExample(item["question"].strip(), item["answer"].split("#### ")[-1])
        for item in _read_jsonl(path)
    ]


def _load_equation_json(path: Path, _: random.Random) -> list[BenchmarkExample]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        BenchmarkExample(item["sQuestion"].strip(), _strip_integer_decimal(item["lSolutions"][0]))
        for item in data
    ]


def _load_svamp(path: Path, _: random.Random) -> list[BenchmarkExample]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        BenchmarkExample(
            f"{item['Body'].strip()} {item['Question'].strip()}",
            _strip_integer_decimal(item["Answer"]),
        )
        for item in data
    ]


def _load_csqa(path: Path, _: random.Random) -> list[BenchmarkExample]:
    examples = []
    for item in _read_jsonl(path):
        choices = item["question"]["choices"]
        prompt = (
            item["question"]["stem"].strip()
            + " "
            + _format_choices(
                [choice["label"] for choice in choices],
                [choice["text"] for choice in choices],
            )
        )
        examples.append(BenchmarkExample(prompt, item["answerKey"]))
    return examples


def _load_strategyqa(path: Path, _: random.Random) -> list[BenchmarkExample]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)["examples"]
    return [
        BenchmarkExample(
            item["input"].strip(),
            "yes" if int(item["target_scores"]["Yes"]) == 1 else "no",
        )
        for item in data
    ]


def _load_symbolic(path: Path, _: random.Random) -> list[BenchmarkExample]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)["examples"]
    return [BenchmarkExample(item["question"].strip(), str(item["answer"])) for item in data]


def _load_agieval(path: Path, _: random.Random) -> list[BenchmarkExample]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return [BenchmarkExample(item["query"].strip(), str(item["answer"])) for item in data]


def _load_piqa(path: Path, _: random.Random) -> list[BenchmarkExample]:
    examples = []
    for item in _read_jsonl(path):
        prompt = f"{item['goal'].strip()} {_format_choices(['A', 'B'], [item['sol1'], item['sol2']])}"
        examples.append(BenchmarkExample(prompt, "A" if int(item["label"]) == 0 else "B"))
    return examples


def _load_siqa(path: Path, _: random.Random) -> list[BenchmarkExample]:
    examples = []
    for item in _read_jsonl(path):
        question = f"{item['context'].strip()} {item['question'].strip()}"
        prompt = f"{question} {_format_choices(['A', 'B', 'C'], [item['answerA'], item['answerB'], item['answerC']])}"
        label = {"1": "A", "2": "B", "3": "C"}[str(item["label"])]
        examples.append(BenchmarkExample(prompt, label))
    return examples


def _load_com2sense(path: Path, _: random.Random) -> list[BenchmarkExample]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        BenchmarkExample(
            f"Is the following statement true or false? {item['sent'].strip()}",
            str(item["label"]).lower(),
        )
        for item in data
    ]


LOADERS: dict[str, Callable[[Path, random.Random], list[BenchmarkExample]]] = {
    "aqua": _load_aqua,
    "gsm8k": _load_gsm8k,
    "addsub": _load_equation_json,
    "multiarith": _load_equation_json,
    "singleeq": _load_equation_json,
    "svamp": _load_svamp,
    "commonsensqa": _load_csqa,
    "strategyqa": _load_strategyqa,
    "coin_flip": _load_symbolic,
    "last_letters": _load_symbolic,
    "agieval": _load_agieval,
    "piqa": _load_piqa,
    "siqa": _load_siqa,
    "com2sense": _load_com2sense,
}


def load_benchmark(
    name: str,
    *,
    data_root: str | Path = "data/benchmark",
    seed: int = 1,
    validate_size: bool = True,
) -> list[BenchmarkExample]:
    canonical = canonical_benchmark_name(name)
    spec = BENCHMARKS[canonical]
    path = Path(data_root) / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run `python scripts/prepare_benchmarks.py` first"
        )
    examples = LOADERS[canonical](path, random.Random(seed))
    if validate_size and len(examples) != spec.expected_examples:
        raise ValueError(
            f"{canonical} has {len(examples)} examples; expected {spec.expected_examples}"
        )
    return examples


def second_stage_prompt(name: str, prompt: str, reasoning: str) -> str:
    instruction = benchmark_spec(name).direct_answer_instruction
    return f"{prompt}\n{reasoning}\n{instruction}"


def _last_match(pattern: str, text: str, flags: int = 0) -> str:
    matches = re.findall(pattern, text, flags)
    if not matches:
        return ""
    match = matches[-1]
    return match if isinstance(match, str) else match[-1]


def normalize_answer(name: str, text: object) -> str:
    """Normalize a generated or reference answer for exact-match scoring."""

    spec = benchmark_spec(name)
    value = str(text).strip()
    if spec.answer_type == "number":
        value = value.replace(",", "")
        number = _last_match(r"-?\d+(?:\.\d+)?", value)
        return number.rstrip("0").rstrip(".") if "." in number else number
    if spec.answer_type.startswith("choice_"):
        allowed = {
            "choice_ab": "AB",
            "choice_ac": "ABC",
            "choice_ae": "ABCDE",
        }[spec.answer_type]
        return _last_match(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", value.upper())
    if spec.answer_type == "yes_no":
        return _last_match(r"\b(yes|no)\b", value.lower())
    if spec.answer_type == "true_false":
        return _last_match(r"\b(true|false)\b", value.lower())
    if spec.answer_type == "word":
        quoted = re.findall(r"[\"']([^\"']+)[\"']", value)
        if quoted:
            value = quoted[-1]
        elif ":" in value:
            value = value.rsplit(":", 1)[-1]
        return re.sub(r"[^a-z0-9]", "", value.lower())
    if spec.answer_type == "math":
        boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", value)
        if boxed:
            value = boxed[-1]
        elif "answer is" in value.lower():
            value = re.split(r"answer is:?", value, flags=re.IGNORECASE)[-1]
        return re.sub(r"[\s$]", "", value).rstrip(".")
    raise AssertionError(f"unsupported answer type: {spec.answer_type}")


def exact_match(name: str, prediction: object, reference: object) -> bool:
    return normalize_answer(name, prediction) == normalize_answer(name, reference)
