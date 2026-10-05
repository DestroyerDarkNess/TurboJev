"""Compare published TurboJev to direct llama.cpp on matched prompts and GGUF.

Run with the registry-installed Python package. The direct executable is a
diagnostic consumer of the very same runtime DLLs, not a production dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import turbojev
from turbojev import TurboJev


def checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_answer(generation, codes, keys):
    if generation["stop_reason"] == "token_limit":
        return None, "unfinished generation"
    try:
        answer = json.loads(generation["text"])["answer"]
        return keys[codes.index(answer)], None
    except (ValueError, KeyError, TypeError):
        return None, "invalid requested JSON/code"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--demo", type=Path, default=Path("other/TurboJev.Demos/snake-python"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.demo.resolve()))
    import snake
    import benchmark
    package_version = importlib.metadata.version("turbojev")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output.with_suffix(".inputs.jsonl")
    outputs_path = args.output.with_suffix(".direct.jsonl")
    records, inputs = [], []
    with TurboJev.load(str(args.model.resolve()), device="cpu", n_gpu_layers=0,
                       n_seq_max=4, thinking=False,
                       execution={"orderRobustness": "none"}) as engine:
        runtime_info = engine.info()
        cases = []
        for fmt in ("coordinates", "sensors"):
            for name, body, direction, food, expected in benchmark.CASES[:4]:
                cases.append((fmt + ":" + name, snake.make_move_request(body, direction, food, fmt), expected))
        cases.append(("control:color", {"state":"The traffic signal is red.", "questions":{"move":{
            "type":"choice", "instructions":"What color is the traffic signal?",
            "criteria":{"blue":"Blue","green":"Green","red":"Red","yellow":"Yellow"}}}}, {"red"}))
        for name, original, expected in cases:
            entries = list(original["questions"]["move"]["criteria"].items())
            for rotation in range(4):
                request = json.loads(json.dumps(original))
                order = entries[rotation:] + entries[:rotation]
                question = request["questions"]["move"]
                question["criteria"] = dict(order)
                keys = [key for key, _ in order]
                codes = [chr(ord("A") + i) for i in range(len(order))]
                # Mirror ReferencePromptRenderer exactly. Candidate probabilities
                # are subsequently compared, so a renderer/codebook mismatch is
                # reported rather than silently assumed equivalent.
                options = "\n".join(f"{code}. {key}: {description}" if key != description else f"{code}. {description}"
                                    for code, (key, description) in zip(codes, order))
                user = f'State:\n{request["state"]}\n\nQuestion:\n{question["instructions"]}\n\nAllowed options:\n{options}\n\nAnswer as {{"answer": "<option letter>"}}.'
                identifier = f"{name}:rotation-{rotation}"
                item = {"id": identifier,
                        "system":"Make the requested decision from the supplied state. Follow the requested output format exactly.",
                        "user":user, "assistant_prefix":'{"answer": "', "codes":codes}
                inputs.append(item)
                started = time.perf_counter()
                response = engine.evaluate(request)
                seconds = time.perf_counter() - started
                choice = response["answers"]["move"]["choice"]
                records.append({"id":identifier, "request":request, "expected_choices":sorted(expected),
                                "keys":keys, "codes":codes, "turbojev":{"response":response,"choice":choice,
                                "correct":choice in expected,"seconds":seconds}})
                print(f"TurboJev {identifier}: {choice}",flush=True)
    inputs_path.write_text("".join(json.dumps(item) + "\n" for item in inputs), encoding="utf-8")
    dll_dir = Path(turbojev.__file__).resolve().parent
    subprocess.run([str(args.direct.resolve()), str(args.model.resolve()), str(inputs_path.resolve()), str(outputs_path.resolve())],
                   check=True, env={**os.environ,"PATH":str(dll_dir)+os.pathsep+os.environ.get("PATH","")})
    results = {item["id"]: item for item in map(json.loads, outputs_path.read_text(encoding="utf-8").splitlines())}
    for row in records:
        result = results[row["id"]]
        row["direct"] = result
        if "error" in result:
            row["parity_error"] = result["error"]
            continue
        prefill = result["exact_prefill"]
        row["direct_restricted_choice"] = row["keys"][row["codes"].index(prefill["restricted_choice_code"])]
        row["direct_restricted_correct"] = row["direct_restricted_choice"] in row["expected_choices"]
        row["choice_parity"] = row["direct_restricted_choice"] == row["turbojev"]["choice"]
        probs = row["turbojev"]["response"]["answers"]["move"]["probabilities"]
        row["probability_max_abs_error"] = max(abs(probs[key] - p) for key,p in zip(row["keys"], prefill["probabilities"]))
        for mode, generation in (("prefilled_generation", prefill["generation"]), ("free_generation", result["free_generation"])):
            choice,error = parse_answer(generation,row["codes"],row["keys"])
            row[mode] = {"choice":choice,"error":error,"correct":choice in row["expected_choices"]}
    groups = {}
    for group in ("coordinates","sensors","control"):
        selected = [row for row in records if row["id"].startswith(group+":")]
        groups[group] = {"total":len(selected), "turbojev_correct":sum(r["turbojev"]["correct"] for r in selected),
                         "direct_restricted_correct":sum(r.get("direct_restricted_correct",False) for r in selected),
                         "prefilled_generation_correct":sum(r.get("prefilled_generation",{}).get("correct",False) for r in selected),
                         "free_generation_correct":sum(r.get("free_generation",{}).get("correct",False) for r in selected)}
    report={"turbojev_version":package_version,"package_source":"PyPI; no local engine override",
            "model":str(args.model.resolve()),"model_sha256":checksum(args.model),"runtime_info":runtime_info,
            "runtime_dll_sha256":{p.name:checksum(p) for p in dll_dir.glob("*.dll")},
            "direct_source_sha256":checksum(Path(__file__).parent / "native-decision-audit/direct_decision.cpp"),
            "config":{"device":"cpu","nGpuLayers":0,"nCtx":4096,"nBatch":512,"nSeqMax":4,"nThreads":4,"thinking":False,"orderRobustness":"none"},
            "groups":groups,"choice_parity":sum(r.get("choice_parity",False) for r in records),
            "probability_max_abs_error":max((r.get("probability_max_abs_error",0) for r in records),default=0),
            "direct_errors":sum("error" in r["direct"] for r in records),"total":len(records),"records":records,
            "controls":"Identical model, native DLLs, system/user messages, Jinja settings and CPU context. Exact-prefill control bypasses TurboJev entirely. Free generation changes ONLY omission of assistant prefill; temperature zero, no grammar and 32-token budget. Invalid or unfinished answers count as failures. No movement replacement.",
            "limitations":"Small static decision audit; no general navigation qualification. Sequential timings include generation work and are not equal-work speed comparisons."}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"groups":groups,"choice_parity":report["choice_parity"],"total":len(records),"probability_max_abs_error":report["probability_max_abs_error"],"direct_errors":report["direct_errors"]}),flush=True)


if __name__ == "__main__":
    main()
