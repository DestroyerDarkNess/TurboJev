#!/usr/bin/env python3
"""Offline semantic checks for Phase 10 specialized-model fixtures."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "specialized"
REQUIRED_METRICS = {
    "accuracy",
    "nll",
    "brier",
    "ece",
    "latency",
    "peak-memory",
    "decisions-per-second",
    "candidate-order-robustness",
    "paraphrase-robustness",
}


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def check_recipe(recipe: dict) -> None:
    assert recipe["contract"] == "turbojev.specialized-training-recipe.v1"
    assert recipe["objective"] == "candidate-logit-cross-entropy"
    assert recipe["method"] in {"lora", "qlora"}
    assert recipe["prompt_ir"] == "turbojev.prompt-ir.v2"
    assert recipe["max_context_tokens"] > 0
    assert recipe["adapter"]["rank"] > 0
    assert recipe["adapter"]["alpha"] > 0
    assert 0 <= recipe["adapter"]["dropout"] < 1
    assert recipe["debiasing"]["candidate_order_permutations"] > 0
    assert recipe["debiasing"]["candidate_code_permutations"] > 0
    assert len(set(recipe["data"]["languages"])) == len(recipe["data"]["languages"])
    assert recipe["data"]["languages"]
    splits = recipe["splits"]
    split_ids = [splits[k] for k in ("train", "validation", "calibration", "benchmark")]
    assert all(split_ids)
    assert len(set(split_ids)) == 4, "training/validation/calibration/benchmark leakage"


def check_semantic_dataset(dataset: dict, recipe: dict) -> None:
    assert dataset["contract"] == "turbojev.semantic-training-dataset.v1"
    assert dataset["id"] == recipe["splits"]["train"]
    examples = dataset["examples"]
    assert examples
    ids = [example["id"] for example in examples]
    assert len(ids) == len(set(ids))
    languages = {example["language"] for example in examples}
    assert languages == set(recipe["data"]["languages"])

    contrastive = Counter()
    has_negative = has_adversarial = has_long = False
    for example in examples:
        request = example["request"]
        assert request["questions"]
        targets = example["targets"]
        assert set(targets) == set(request["questions"])
        props = example.get("properties", {})
        if props.get("contrastive_group"):
            contrastive[props["contrastive_group"]] += 1
        has_negative |= bool(props.get("negative_no_match"))
        has_adversarial |= bool(props.get("adversarial_criteria"))
        has_long |= bool(props.get("long_shared_state"))
        assert example.get("weight", 1.0) > 0

        for question_id, question in request["questions"].items():
            target = targets[question_id]
            if question["type"] == "choice":
                assert target["type"] == "choice"
                assert target["value"] in question["criteria"]
            elif question["type"] in {"noul", "boolean"}:
                assert target["type"] == "noul"
                assert isinstance(target["value"], bool)
            elif question["type"] == "score":
                assert target["type"] == "score"
                assert 0 <= target["value"] < len(question["criteria"])
            else:
                raise AssertionError(f"unknown question type {question['type']}")

    assert any(count >= 2 for count in contrastive.values())
    assert has_negative and has_adversarial and has_long


def check_materialized_sample(materialized: dict, recipe: dict) -> None:
    assert materialized["contract"] == "turbojev.materialized-training-dataset.v1"
    assert materialized["source_dataset"] == recipe["splits"]["train"]
    assert materialized["recipe_id"] == recipe["id"]
    assert materialized["prompt_ir"] == recipe["prompt_ir"]
    assert materialized["renderer_profile"] == recipe["renderer_profile"]
    assert materialized["seed"] == recipe["debiasing"]["seed"]
    assert materialized["records"]
    assert materialized["stats"]["records"] == len(materialized["records"])
    total_weight = 0.0
    ids = set()
    for record in materialized["records"]:
        assert record["id"] not in ids
        ids.add(record["id"])
        document = record["document"]
        assert document["contract"] == "turbojev.prompt-ir.v2"
        options = document["options"]
        assert len({option["key"] for option in options}) == len(options)
        assert len({option["code"] for option in options}) == len(options)
        target = options[record["target_index"]]
        assert target["key"] == record["target_key"]
        assert target["code"] == record["target_code"]
        assert record["weight"] > 0
        total_weight += record["weight"]
    assert abs(total_weight - materialized["stats"]["total_record_weight"]) < 1e-12



def check_model_bound_sample(bound: dict, materialized: dict, recipe: dict) -> None:
    assert bound["contract"] == "turbojev.model-bound-training-dataset.v1"
    assert bound["source_materialized_dataset"] == materialized["id"]
    assert bound["source_semantic_dataset"] == recipe["splits"]["train"]
    assert bound["recipe_id"] == recipe["id"]
    assert bound["objective"] == "candidate-logit-cross-entropy"
    assert bound["model"] == recipe["base_model"]
    assert bound["prompt_ir"] == recipe["prompt_ir"]
    assert bound["renderer_profile"] == recipe["renderer_profile"]
    assert bound["max_context_tokens"] == recipe["max_context_tokens"]
    assert bound["records"]
    ids = set()
    total_weight = 0.0
    max_prompt = 0
    max_context = 0
    for record in bound["records"]:
        assert record["id"] not in ids
        ids.add(record["id"])
        assert record["messages"]["profile"] == bound["renderer_profile"]
        assert record["prompt_token_ids"]
        assert len(record["candidates"]) >= 2
        assert len({c["key"] for c in record["candidates"]}) == len(record["candidates"])
        assert len({c["code"] for c in record["candidates"]}) == len(record["candidates"])
        assert all(len(c["token_ids"]) == 1 for c in record["candidates"])
        assert len({c["token_ids"][0] for c in record["candidates"]}) == len(record["candidates"])
        target = record["candidates"][record["target_index"]]
        assert target["key"] == record["target_key"]
        assert target["code"] == record["target_code"]
        assert target["token_ids"][0] == record["target_token_id"]
        used = len(record["prompt_token_ids"]) + 1
        assert used <= bound["max_context_tokens"]
        assert record["weight"] > 0
        total_weight += record["weight"]
        max_prompt = max(max_prompt, len(record["prompt_token_ids"]))
        max_context = max(max_context, used)
    stats = bound["stats"]
    assert stats["source_records"] >= 1
    assert stats["records"] == len(bound["records"])
    assert abs(stats["total_record_weight"] - total_weight) < 1e-12
    assert stats["max_prompt_tokens"] == max_prompt
    assert stats["max_context_tokens_used"] == max_context


def check_specialized_calibration(profile: dict, recipe: dict) -> None:
    assert profile["contract"] == "turbojev.calibration-profile.v1"
    assert profile["method"] == "temperature-softmax"
    assert profile["temperature"] > 0
    assert profile["holdout_id"] == recipe["splits"]["calibration"]
    assert profile["tokenizer_revision"] == recipe["base_model"]["tokenizer_revision"]
    assert profile["prompt_profile"] == recipe["prompt_ir"]
    assert profile["renderer_profile"] == recipe["renderer_profile"]
    assert profile["model_id"]
    assert profile["sample_count"] > 0
    assert profile["ece_bins"] > 0
    for stage in ("before", "after"):
        metrics = profile[stage]
        assert metrics["count"] > 0
        assert metrics["total_weight"] > 0
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["nll"] >= 0
        assert metrics["brier"] >= 0
        assert metrics["ece"] >= 0

def main() -> None:
    names = [
        "recipe-0.6b-lora.json",
        "recipe-0.8b-lora.json",
        "recipe-2b-qlora.json",
        "recipe-4b-qlora.json",
    ]
    recipes = [load(name) for name in names]
    for recipe in recipes:
        check_recipe(recipe)

    assert [r["base_model"]["id"].split("-")[-1] for r in recipes] == [
        "0.6b", "0.8b", "2b", "4b"
    ]

    manifest = load("model-manifest-0.6b.json")
    recipe = recipes[0]
    assert manifest["contract"] == "turbojev.specialized-model.v1"
    assert manifest["recipe_id"] == recipe["id"]
    assert manifest["base_model"] == recipe["base_model"]
    decision = manifest["decision_contract"]
    assert decision["contract"] == "turbojev.decision-model.v1"
    assert decision["prompt_ir"] == recipe["prompt_ir"]
    assert decision["renderer_profile"] == recipe["renderer_profile"]
    assert decision["candidate_encoding"] == recipe["candidate_encoding"]
    assert decision["tokenizer_revision"] == recipe["base_model"]["tokenizer_revision"]
    assert decision["max_context_tokens"] == recipe["max_context_tokens"]
    assert manifest["training_dataset"] == recipe["splits"]["train"]
    assert manifest["calibration_profile"] == decision["calibration_profile"]

    evaluation = load("evaluation-plan.json")
    assert evaluation["contract"] == "turbojev.specialization-evaluation.v1"
    assert evaluation["benchmark_suite"] == recipe["splits"]["benchmark"]
    assert len(evaluation["pairs"]) == 4
    assert set(evaluation["metrics"]) == REQUIRED_METRICS
    for pair in evaluation["pairs"]:
        assert pair["generic_model"] != pair["specialized_model"]

    check_semantic_dataset(load("semantic-training-dataset.json"), recipe)
    materialized = load("materialized-record-sample.json")
    check_materialized_sample(materialized, recipe)
    check_model_bound_sample(load("model-bound-training-sample.json"), materialized, recipe)
    calibration = load("calibration-specialized-profile.json")
    check_specialized_calibration(calibration, recipe)
    assert calibration["id"] == manifest["calibration_profile"]
    assert calibration["model_id"] == manifest["id"]
    assert calibration["tokenizer_revision"] == manifest["base_model"]["tokenizer_revision"]
    assert calibration["prompt_profile"] == manifest["decision_contract"]["prompt_ir"]
    assert calibration["renderer_profile"] == manifest["decision_contract"]["renderer_profile"]

    print("OK: Phase 10 recipes + semantic/materialized/model-bound artifact fixtures PASS")


if __name__ == "__main__":
    main()
