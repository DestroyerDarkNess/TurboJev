"""Run real typed-field regression cases through the native C ABI."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--suite", choices=["fields", "reasoning", "all"], default="fields")
    args = parser.parse_args()
    directories = [os.add_dll_directory(str(p.resolve())) for p in args.library_dir] if os.name == "nt" else []
    lib = C.CDLL(str(args.library.resolve()))
    handle = C.c_void_p()
    lib.tj_engine_open_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
    lib.tj_engine_generate_json.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]
    lib.tj_engine_close.argtypes = [C.c_void_p]
    lib.tj_string_free.argtypes = [C.c_void_p]
    lib.tj_last_error_message.restype = C.c_char_p

    def check(status):
        if status:
            raise RuntimeError(lib.tj_last_error_message().decode())

    config = {"model": str(args.model.resolve()), "device": args.device, "nCtx": 8192, "nSeqMax": 1, "nThreads": 4, "nThreadsBatch": 4}
    check(lib.tj_engine_open_json(json.dumps(config).encode(), C.byref(handle)))
    cases = [
        ("string", "The merchant is Hilton London.", {"merchant": {"type": "string", "instructions": "Return only the merchant name."}}, {"merchant": "Hilton London"}),
        ("integer", "There are exactly 37 units.", {"count": {"type": "integer", "instructions": "How many units?"}}, {"count": 37}),
        ("number", "The total is 324.50 GBP.", {"total": {"type": "number", "instructions": "What is the total in GBP?"}}, {"total": 324.5}),
        ("nullable", "No invoice number is provided.", {"invoice": {"type": ["string", "null"], "instructions": "Return the invoice number, or null when absent."}}, {"invoice": None}),
        ("finite", "The service is down.", {"team": {"type": "string", "enum": ["outage", "billing", "feature"], "instructions": "Which team handles this?"}, "urgent": {"type": "boolean", "instructions": "Is this urgent?"}}, {"team": "outage", "urgent": True}),
        ("dependency", "There are exactly 7 units.", {"count": {"type": "integer", "instructions": "How many units?"}, "check": {"type": "boolean", "depends_on": ["count"], "instructions": "Is count greater than five?"}, "skip": {"type": "string", "when": {"count": {"lt": 5}}, "instructions": "Write a note."}}, {"count": 7, "check": True}),
        ("numeric_enum", "The severity level is 2.", {"severity": {"type": "integer", "enum": [0, 1, 2], "instructions": "What severity level is stated?", "return_probabilities": True}}, None),
        ("permutations", "A fair coin is tossed.", {"coin": {"type": "string", "enum": ["heads", "tails"], "permutations": "all", "return_probabilities": True}}, None),
    ]
    reasoning_cases = [
        ("thinking", "There are exactly seven items.", {"count": {"type": "integer", "instructions": "How many items?", "thinking": True, "thinking_budget": 512}}, {"count": 7}),
        ("auto", "The label is red.", {"color": {"type": "string", "enum": ["red", "blue"], "instructions": "Which color is stated?", "thinking": "auto"}}, {"color": "red"}),
        ("restored", "There are exactly seven items.", {"count": {"type": "integer", "instructions": "How many items?"}}, {"count": 7}),
    ]
    if args.suite == "reasoning":
        cases = reasoning_cases
    elif args.suite == "all":
        cases += reasoning_cases
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, context, questions, expected in cases:
            request = {"context": context, "questions": questions}
            start = time.perf_counter()
            output = C.c_void_p()
            try:
                check(lib.tj_engine_generate_json(handle, json.dumps(request).encode(), C.byref(output)))
                response = json.loads(C.string_at(output).decode())
                if name == "thinking" and not response["thinking"].get("count"):
                    raise AssertionError("No actual reasoning trace returned")
                if name == "auto" and response["thinking_effort"].get("color") not in {"none", "low", "medium", "high"}:
                    raise AssertionError("No valid automatic effort returned")
                if name == "restored" and response["thinking"]:
                    raise AssertionError("Reasoning options were not restored")
                record = {"id": name, "request": request, "response": response, "expected": expected, "matched": None if expected is None else response["result"] == expected}
            except Exception as error:
                record = {"id": name, "request": request, "error": str(error)}
            finally:
                if output.value:
                    lib.tj_string_free(output)
            record["seconds"] = time.perf_counter() - start
            records.append(record)
            args.output.write_text(json.dumps({"config": config, "suite": args.suite, "records": records}, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record), flush=True)
    finally:
        check(lib.tj_engine_close(handle))
        for directory in directories:
            directory.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": config, "suite": args.suite, "records": records}, indent=2) + "\n", encoding="utf-8")
    if any("error" in record or record.get("matched") is False for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
