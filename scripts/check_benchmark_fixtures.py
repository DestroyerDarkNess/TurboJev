#!/usr/bin/env python3
"""Validate committed benchmark fixtures without executing a model runtime."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks" / "fixtures"
CONTRACT = "turbojev.benchmark.v1"
QUESTION_TYPES = {"choice", "noul", "boolean", "score"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"BENCHMARK FIXTURE CHECK FAILED: {message}")


def validate_question(case_id: str, question_id: str, question: object) -> None:
    require(isinstance(question, dict), f"{case_id}/{question_id}: question must be an object")
    kind = question.get("type")
    require(kind in QUESTION_TYPES, f"{case_id}/{question_id}: unsupported type {kind!r}")
    instructions = question.get("instructions")
    require(isinstance(instructions, str) and instructions.strip(), f"{case_id}/{question_id}: empty instructions")
    if kind == "choice":
        criteria = question.get("criteria")
        require(isinstance(criteria, dict) and len(criteria) >= 2, f"{case_id}/{question_id}: choice requires >=2 criteria")
        require(all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in criteria.items()), f"{case_id}/{question_id}: invalid choice criteria")
    elif kind == "score":
        criteria = question.get("criteria")
        require(isinstance(criteria, list) and len(criteria) >= 2, f"{case_id}/{question_id}: score requires >=2 levels")
        require(all(isinstance(v, str) and v.strip() for v in criteria), f"{case_id}/{question_id}: invalid score levels")


def main() -> None:
    suites = sorted(FIXTURES.glob("*.json"))
    require(suites, "no JSON fixtures")
    suite_count = 0
    case_count = 0
    languages: set[str] = set()
    for path in suites:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "contract" not in data:
            continue
        suite_count += 1
        require(data.get("contract") == CONTRACT, f"{path.name}: wrong contract")
        require(isinstance(data.get("id"), str) and data["id"].strip(), f"{path.name}: empty suite id")
        cases = data.get("cases")
        require(isinstance(cases, list) and cases, f"{path.name}: empty cases")
        seen: set[str] = set()
        for case in cases:
            require(isinstance(case, dict), f"{path.name}: case must be object")
            case_id = case.get("id")
            require(isinstance(case_id, str) and case_id.strip(), f"{path.name}: empty case id")
            require(case_id not in seen, f"{path.name}: duplicate case {case_id}")
            seen.add(case_id)
            tags = case.get("tags", [])
            require(isinstance(tags, list) and all(isinstance(x, str) for x in tags), f"{case_id}: invalid tags")
            languages.update(tag for tag in tags if tag in {"english", "spanish", "french", "japanese"})
            request = case.get("request")
            require(isinstance(request, dict), f"{case_id}: missing request")
            state = request.get("state")
            require(isinstance(state, (str, dict, list)) and bool(state), f"{case_id}: invalid state")
            questions = request.get("questions")
            require(isinstance(questions, dict) and questions, f"{case_id}: no questions")
            for qid, question in questions.items():
                require(isinstance(qid, str) and qid.strip(), f"{case_id}: empty question id")
                validate_question(case_id, qid, question)
            case_count += 1
    require(suite_count >= 1, "no benchmark suite found")
    require(len(languages) >= 4, "reference fixture must exercise at least four human languages")
    print(f"OK: {suite_count} benchmark suite(s), {case_count} cases, languages={','.join(sorted(languages))}")


if __name__ == "__main__":
    main()
