"""Reproducible real-model numeric and JevBench evaluations; no gold in prompts."""
from __future__ import annotations
import argparse
from collections import defaultdict
import ctypes as C
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent


class NativeClient:
    """Explicit development-only ABI path; normal runs use the published wheel."""
    def __init__(self, library, directories, config):
        self.directories = [os.add_dll_directory(str(p.resolve())) for p in directories] if os.name == "nt" else []
        self.lib = C.CDLL(str(library.resolve()))
        self.handle = C.c_void_p()
        self.lib.tj_engine_open_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.tj_engine_generate_json.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.tj_engine_evaluate_json.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.tj_engine_close.argtypes = [C.c_void_p]
        self.lib.tj_string_free.argtypes = [C.c_void_p]
        self.lib.tj_last_error_message.restype = C.c_char_p
        self.check(self.lib.tj_engine_open_json(json.dumps(config).encode(), C.byref(self.handle)))

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.tj_last_error_message().decode())

    def call(self, operation, request):
        output = C.c_void_p()
        try:
            self.check(getattr(self.lib, "tj_engine_" + operation + "_json")(
                self.handle, json.dumps(request, allow_nan=False).encode(), C.byref(output)))
            return json.loads(C.string_at(output).decode())
        finally:
            if output.value:
                self.lib.tj_string_free(output)

    def generate(self, request):
        return self.call("generate", request)

    def evaluate(self, request):
        return self.call("evaluate", request)

    def close(self):
        self.check(self.lib.tj_engine_close(self.handle))
        for directory in self.directories:
            directory.close()


def matched(actual, expected):
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        return not isinstance(actual, bool) and isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    return type(actual) is type(expected) and actual == expected


def numeric_request(case, args):
    questions = {"answer": {"type": case["type"], "instructions": case["question"]}}
    if "followup" in case:
        questions["is_positive"] = {"type": "boolean", "instructions": case["followup"]["question"], "depends_on": ["answer"]}
    thinking = "auto" if args.thinking == "auto" else args.thinking == "true"
    request = {"context": case["context"], "questions": questions, "thinking": thinking,
               "text_max_tokens": args.max_tokens, "numeric_max_tokens": args.max_tokens}
    if args.thinking_budget:
        request["thinking_budget"] = args.thinking_budget
    return request


def jev_request(case):
    question = dict(case["question"])
    # The existing noul contract has an instruction rather than named criteria.
    # Preserve upstream's explicit true/false rubric in that instruction.
    if question["type"] == "noul" and "criteria" in question:
        criteria = question.pop("criteria")
        question["instructions"] += f"\nTrue means: {criteria['true']}\nFalse means: {criteria['false']}"
    return {"state":case["state"],"questions":{"answer":question}}


def score_case(case, response, suite):
    if suite == "numeric":
        result = response["result"]
        checks = {"answer": matched(result.get("answer"), case["expected"])}
        if "followup" in case:
            checks["is_positive"] = matched(result.get("is_positive"), case["followup"]["expected"])
        return {"correct": all(checks.values()), "checks": checks, "predicted": result}
    answer = response["answers"]["answer"]
    kind = case["question"]["type"]
    if kind == "noul":
        probabilities = {"no": 1-answer["noul"], "yes": answer["noul"]}
    elif kind == "score":
        probabilities = {str(i): p for i, p in enumerate(answer["probabilities"])} if isinstance(answer["probabilities"], list) else answer["probabilities"]
    else:
        probabilities = answer["probabilities"]
    probabilities = {str(k): v for k, v in probabilities.items()}
    if set(probabilities) != set(map(str, case["labels"])) or any(not math.isfinite(v) or v < 0 or v > 1 for v in probabilities.values()) or abs(sum(probabilities.values())-1) > 1e-6:
        raise ValueError("Invalid label distribution")
    # Stable lexical tie breaking, as in the pinned JevBench/TypeLLM audit.
    predicted = min(probabilities, key=lambda key: (-probabilities[key], key))
    expected = str(case["expected"])
    result = {"correct": predicted == expected, "predicted": predicted, "probabilities": probabilities,
              "confidence": probabilities[predicted], "brier": sum((p-int(k == expected))**2 for k,p in probabilities.items())}
    if kind == "score":
        result["ordinal_mae"] = abs(answer["score"]-float(expected))
    return result


def summarize(records):
    durations = sorted(r["seconds"] for r in records)
    good = [r for r in records if "error" not in r]
    out = {"attempted": len(records), "completed": len(good), "errors": len(records)-len(good),
           "correct": sum(r.get("correct", False) for r in records),
           "accuracy": sum(r.get("correct", False) for r in records)/len(records) if records else None,
           "latency_seconds": {"mean": statistics.mean(durations) if durations else None,
                               "p50": statistics.median(durations) if durations else None,
                               "p95": durations[max(0, math.ceil(len(durations)*.95)-1)] if durations else None}}
    for metric in ("brier", "ordinal_mae"):
        values = [r[metric] for r in good if metric in r]
        if values:
            out[metric + "_mean"] = statistics.mean(values)
    bins = [[] for _ in range(10)]
    for r in good:
        if "confidence" in r:
            bins[min(9,int(r["confidence"]*10))].append(r)
    count = sum(map(len,bins))
    if count:
        out["ece_10_bins"] = sum(len(b)/count*abs(statistics.mean(r["confidence"] for r in b)-statistics.mean(r["correct"] for r in b)) for b in bins if b)
    return out


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["numeric","jevbench","all"], default="all")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--thinking", choices=["false","true","auto"], default="false")
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Per-suite smoke subset, never reported as a full eval")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run with identical model, configuration, data and requests")
    parser.add_argument("--native-library", type=Path, help="Explicit unpublished development build")
    parser.add_argument("--library-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.thinking == "auto" and args.thinking_budget:
        parser.error("auto thinking cannot have a fixed budget")
    if args.suite != "numeric" and args.thinking == "auto":
        parser.error("JevBench evaluates the existing evaluate API; auto is available in generate only")
    if args.limit is not None and args.limit < 1 or args.offset < 0:
        parser.error("limit must be positive and offset nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"model": str(args.model.resolve()), "device": args.device, "nCtx": 8192, "nSeqMax": 1,
              "nThreads": args.threads, "nThreadsBatch": args.threads,
              "thinking": args.thinking == "true", "maxThinkingTokens": args.thinking_budget or 512}
    if args.native_library:
        client = NativeClient(args.native_library, args.library_dir, config)
        source = {"kind":"unpublished-native-build","library":str(args.native_library.resolve()),"sha256":sha(args.native_library)}
    else:
        from turbojev import TurboJev
        client = TurboJev.load(config["model"], device=args.device, n_ctx=8192, n_seq_max=1,
                               n_threads=args.threads, n_threads_batch=args.threads,
                               thinking=config["thinking"], max_thinking_tokens=config["maxThinkingTokens"])
        source = {"kind":"registry-wheel","version":importlib.metadata.version("turbojev")}
    files = ([HERE/"data/numeric/cases.jsonl"] if args.suite in ("numeric","all") else []) + ([HERE/f"data/jevbench/{tier}.jsonl" for tier in ("original","easy","hard")] if args.suite in ("jevbench","all") else [])
    groups = defaultdict(list)
    for path in files:
        suite = "numeric" if path.parent.name == "numeric" else "jevbench"
        groups[suite].extend({**json.loads(line), "tier":path.stem} for line in path.read_text(encoding="utf-8").splitlines() if line)
    selected = [(suite,case) for suite,cases in groups.items() for case in cases[args.offset:args.offset+args.limit if args.limit else None]]
    manifest = {"source":source,"model":{"path":str(args.model.resolve()),"sha256":sha(args.model)},
                "config":config,"platform":platform.platform(),"machine":platform.machine(),"python":platform.python_version(),
                "datasets":{str(path.relative_to(HERE)):sha(path) for path in files},
                "subset":args.limit is not None or args.offset != 0,"offset":args.offset,"limit_per_suite":args.limit,
                "thinking":args.thinking,"max_tokens":args.max_tokens,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    records = []
    manifest_path=args.output/"manifest.json"
    result_path=args.output/"results.jsonl"
    if args.resume:
        previous=json.loads(manifest_path.read_text(encoding="utf-8"))
        source_keys=("kind","sha256") if args.native_library else ("kind","version")
        if any(previous['source'].get(k)!=source.get(k) for k in source_keys):
            raise ValueError("Cannot resume: executable/package identity changed")
        for key in ("model","config","datasets","subset","offset","limit_per_suite","thinking","max_tokens"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Cannot resume: {key} changed")
        records=[json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line]
        expected_requests={(suite,case['id']):(numeric_request(case,args) if suite=='numeric' else jev_request(case)) for suite,case in selected}
        seen=set()
        for record in records:
            identity=(record['suite'],record['id'])
            if identity in seen or expected_requests.get(identity) != record['request']:
                raise ValueError("Cannot resume changed/duplicate tasks")
            seen.add(identity)
        manifest=previous
        if args.native_library and previous['source']['library'] != source['library']:
            manifest.setdefault('relocated_libraries',[]).append(source)
        manifest.setdefault('resumed_utc',[]).append(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    elif manifest_path.exists() or result_path.exists():
        raise ValueError("Output already exists; use --resume or a new directory")
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    completed={(r['suite'],r['id']) for r in records}
    pending=[(suite,case) for suite,case in selected if (suite,case['id']) not in completed]
    start_all = time.perf_counter()
    try:
        with result_path.open("a" if args.resume else "w",encoding="utf-8") as stream:
            for index,(suite,case) in enumerate(pending):
                request = numeric_request(case,args) if suite == "numeric" else jev_request(case)
                record = {"id":case["id"],"suite":suite,"category":case.get("category",case.get("family")),"tier":case["tier"],"expected":case["expected"],"request":request}
                begin = time.perf_counter()
                try:
                    response = client.generate(request) if suite == "numeric" else client.evaluate(request)
                    record.update(response=response,**score_case(case,response,suite))
                except Exception as error:
                    record.update(error=str(error),correct=False)
                record["seconds"] = time.perf_counter()-begin
                records.append(record)
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
                stream.flush()
                fraction = len(records)/len(selected)
                filled = round(fraction*24)
                eta = (time.perf_counter()-start_all)/(index+1)*(len(pending)-index-1)
                sys.stdout.write(f"\r[{'#'*filled}{'-'*(24-filled)}] {len(records)}/{len(selected)}  {case['id']}  {record['seconds']:.2f}s  ETA {eta:.0f}s          ")
                sys.stdout.flush()
    finally:
        client.close()
        print()
    summary = {"overall":summarize(records),"suites":{s:summarize([r for r in records if r['suite']==s]) for s in groups},
               "categories":{c:summarize([r for r in records if r['category']==c]) for c in sorted({r['category'] for r in records})},
               "inference_seconds":sum(r['seconds'] for r in records),
               "current_segment_seconds":time.perf_counter()-start_all,"subset":manifest["subset"]}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary["overall"],indent=2))
    return int(any("error" in r for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
