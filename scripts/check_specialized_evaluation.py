#!/usr/bin/env python3
"""Offline semantic checks for v0.20 specialization evaluation/interchange."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "specialized"
SCHEMAS = ROOT / "schemas"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(schema_name: str, document: dict) -> None:
    schema = load(SCHEMAS / schema_name)
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda e: list(e.path))
    if errors:
        details = "\n".join(f"- {list(error.path)}: {error.message}" for error in errors)
        raise AssertionError(f"{schema_name} validation failed:\n{details}")


def softmax(logits: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def baseline_quality(run: dict) -> tuple[float, float, float]:
    correct = nll = brier = total = 0.0
    for obs in run["observations"]:
        if obs["variant"] != "baseline":
            continue
        weight = obs.get("weight", 1.0)
        target = obs["candidate_keys"].index(obs["target_key"])
        probabilities = softmax(obs["logits"], run["model"]["calibration"]["temperature"])
        prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        correct += weight * (prediction == target)
        nll += weight * -math.log(max(probabilities[target], float.fromhex("0x1.0p-1022")))
        brier += weight * sum(
            (probability - (1.0 if index == target else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        )
        total += weight
    return correct / total, nll / total, brier / total


def prediction(obs: dict) -> str:
    index = max(range(len(obs["logits"])), key=obs["logits"].__getitem__)
    return obs["candidate_keys"][index]


def robustness(run: dict, kind: str) -> tuple[float, float]:
    baselines = {obs["family_id"]: obs for obs in run["observations"] if obs["variant"] == "baseline"}
    total = invariant = accurate = 0.0
    for obs in run["observations"]:
        if obs["variant"] != kind:
            continue
        weight = obs.get("weight", 1.0)
        total += weight
        invariant += weight * (prediction(obs) == prediction(baselines[obs["family_id"]]))
        accurate += weight * (prediction(obs) == obs["target_key"])
    assert total > 0
    return invariant / total, accurate / total


def check_run(run: dict) -> None:
    assert run["contract"] == "turbojev.specialization-run.v1"
    assert run["observations"]
    assert run["environment"]["backend"]
    assert run["environment"]["host"]
    calibration = run["model"]["calibration"]
    assert calibration["method"] == "temperature-softmax"
    assert calibration["temperature"] > 0
    if calibration["fitted"]:
        assert calibration["profile_id"]
    ids = {obs["id"] for obs in run["observations"]}
    assert len(ids) == len(run["observations"])
    families: dict[str, list[dict]] = {}
    for obs in run["observations"]:
        assert len(obs["candidate_keys"]) == len(obs["logits"]) >= 2
        assert len(set(obs["candidate_keys"])) == len(obs["candidate_keys"])
        assert obs["target_key"] in obs["candidate_keys"]
        assert obs["latency_seconds"] > 0
        assert obs["peak_memory_bytes"] > 0
        families.setdefault(obs["family_id"], []).append(obs)
    assert any(obs["variant"] == "candidate-order" for obs in run["observations"])
    assert any(obs["variant"] == "paraphrase" for obs in run["observations"])
    for family in families.values():
        baseline = [obs for obs in family if obs["variant"] == "baseline"]
        assert len(baseline) == 1
        base = baseline[0]
        for obs in family:
            assert obs["case_id"] == base["case_id"]
            assert obs["question_id"] == base["question_id"]
            assert obs["target_key"] == base["target_key"]
            assert set(obs["candidate_keys"]) == set(base["candidate_keys"])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = load(FIXTURES / "evaluation-plan-smoke.json")
    generic = load(FIXTURES / "run-generic-0.6b.json")
    specialized = load(FIXTURES / "run-specialized-0.6b.json")
    report = load(FIXTURES / "evaluation-report-smoke.json")
    bundle = load(FIXTURES / "experiment-bundle-smoke.json")

    validate("turbojev.specialization-evaluation.schema.json", plan)
    validate("turbojev.specialization-run.schema.json", generic)
    validate("turbojev.specialization-run.schema.json", specialized)
    validate("turbojev.specialization-report.schema.json", report)
    validate("turbojev.specialization-bundle.schema.json", bundle)

    check_run(generic)
    check_run(specialized)
    pair = plan["pairs"][0]
    assert generic["model"]["id"] == pair["generic_model"]
    assert specialized["model"]["id"] == pair["specialized_model"]
    assert generic["plan_id"] == specialized["plan_id"] == plan["id"]
    assert generic["benchmark_suite"] == specialized["benchmark_suite"] == plan["benchmark_suite"]
    for field in ("prompt_ir", "renderer_profile", "tokenizer_revision"):
        assert generic["model"][field] == specialized["model"][field]
    for field in ("backend", "host", "quantization", "execution_path"):
        assert generic["environment"].get(field) == specialized["environment"].get(field)

    generic_obs = {obs["id"]: obs for obs in generic["observations"]}
    specialized_obs = {obs["id"]: obs for obs in specialized["observations"]}
    assert generic_obs.keys() == specialized_obs.keys()
    for obs_id in generic_obs:
        left, right = generic_obs[obs_id], specialized_obs[obs_id]
        for field in ("family_id", "case_id", "question_id", "variant", "candidate_keys", "target_key", "weight"):
            assert left[field] == right[field]

    ga, gnll, gbrier = baseline_quality(generic)
    sa, snll, sbrier = baseline_quality(specialized)
    go, goa = robustness(generic, "candidate-order")
    so, soa = robustness(specialized, "candidate-order")
    gp, gpa = robustness(generic, "paraphrase")
    sp, spa = robustness(specialized, "paraphrase")

    comparison = report["comparisons"][0]
    assert comparison["pair"] == pair
    assert comparison["generic_run"] == generic["id"]
    assert comparison["specialized_run"] == specialized["id"]
    assert math.isclose(comparison["generic"]["baseline_quality"]["accuracy"], ga, abs_tol=1e-12)
    assert math.isclose(comparison["generic"]["baseline_quality"]["nll"], gnll, abs_tol=1e-12)
    assert math.isclose(comparison["generic"]["baseline_quality"]["brier"], gbrier, abs_tol=1e-12)
    assert math.isclose(comparison["specialized"]["baseline_quality"]["accuracy"], sa, abs_tol=1e-12)
    assert math.isclose(comparison["specialized"]["baseline_quality"]["nll"], snll, abs_tol=1e-12)
    assert math.isclose(comparison["specialized"]["baseline_quality"]["brier"], sbrier, abs_tol=1e-12)
    assert math.isclose(comparison["generic"]["candidate_order"]["invariance"], go, abs_tol=1e-12)
    assert math.isclose(comparison["specialized"]["candidate_order"]["invariance"], so, abs_tol=1e-12)
    assert math.isclose(comparison["generic"]["paraphrase"]["invariance"], gp, abs_tol=1e-12)
    assert math.isclose(comparison["specialized"]["paraphrase"]["invariance"], sp, abs_tol=1e-12)

    delta = comparison["delta"]
    expected = {
        "accuracy": sa - ga,
        "nll": snll - gnll,
        "brier": sbrier - gbrier,
        "candidate_order_invariance": so - go,
        "candidate_order_accuracy": soa - goa,
        "paraphrase_invariance": sp - gp,
        "paraphrase_accuracy": spa - gpa,
    }
    for key, value in expected.items():
        assert math.isclose(delta[key], value, abs_tol=1e-12), (key, delta[key], value)

    artifact_paths = {
        (plan["contract"], plan["id"]): FIXTURES / "evaluation-plan-smoke.json",
        (generic["contract"], generic["id"]): FIXTURES / "run-generic-0.6b.json",
        (specialized["contract"], specialized["id"]): FIXTURES / "run-specialized-0.6b.json",
        (report["contract"], report["id"]): FIXTURES / "evaluation-report-smoke.json",
        ("turbojev.specialized-training-recipe.v1", "turbojev-0.6b-lora-candidate-ce-v1"): FIXTURES / "recipe-0.6b-lora.json",
        ("turbojev.specialized-model.v1", "turbojev-0.6b-specialized-fixture"): FIXTURES / "model-manifest-0.6b.json",
        ("turbojev.calibration-profile.v1", "turbojev-0.6b-calibration-v1"): FIXTURES / "calibration-specialized-profile.json",
        ("turbojev.benchmark.v1", "reference-semantic-v1"): ROOT / "benchmarks" / "fixtures" / "reference-semantic-v1.json",
    }
    artifacts = [bundle["plan"], bundle["benchmark"], *bundle["runs"], bundle["report"], *bundle.get("recipes", []), *bundle.get("model_manifests", []), *bundle.get("calibration_profiles", [])]
    for artifact in artifacts:
        path = artifact_paths[(artifact["contract"], artifact["id"])]
        assert artifact["sha256"] == sha256(path), (artifact["id"], path)

    print("OK: v0.20 external runs + paired evaluation + robustness + reproducible bundle PASS")


if __name__ == "__main__":
    main()
