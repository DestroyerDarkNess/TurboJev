"""Validate structured SDK features against a real GGUF through the C ABI.

The report separates schema/runtime contracts from model answer accuracy.
"""

import argparse
import hashlib
import ctypes
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = os.add_dll_directory(str(args.library_dir.resolve())) if os.name == "nt" else None
    library = ctypes.CDLL(str(args.library.resolve()))
    pointer = ctypes.c_void_p
    library.tj_engine_open_json.argtypes = [ctypes.c_char_p, ctypes.POINTER(pointer)]
    library.tj_engine_generate_json.argtypes = [pointer, ctypes.c_char_p, ctypes.POINTER(pointer)]
    library.tj_compile_schema_json.argtypes = [ctypes.c_char_p, ctypes.POINTER(pointer)]
    library.tj_engine_close.argtypes = [pointer]
    library.tj_string_free.argtypes = [pointer]
    library.tj_last_error_message.restype = ctypes.c_char_p

    def check(status):
        if status:
            raise RuntimeError(library.tj_last_error_message().decode())

    def call(function, *arguments):
        output = pointer()
        try:
            check(function(*arguments, ctypes.byref(output)))
            return json.loads(ctypes.string_at(output).decode())
        finally:
            if output.value:
                library.tj_string_free(output)

    compiled = call(
        library.tj_compile_schema_json,
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "person": {"type": "object", "properties": {"age": {"type": "integer"}}}
                },
            }
        ).encode(),
    )
    assert compiled[0]["path"] == ["person", "age"]
    config = {
        "model": str(args.model.resolve()),
        "device": "cpu",
        "nCtx": 8192,
        "nSeqMax": 1,
        "nThreads": 4,
        "nThreadsBatch": 4,
    }
    handle = pointer()
    check(library.tj_engine_open_json(json.dumps(config).encode(), ctypes.byref(handle)))
    cases = [
        (
            "nested",
            "Ada is 37 years old.",
            {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "instructions": "What is the person's name?"},
                        "age": {"type": "integer", "instructions": "What is the person's age?"},
                    },
                }
            },
            {"person": {"name": "Ada", "age": 37}},
        ),
        (
            "described_choice",
            "The service is down.",
            {
                "team": {
                    "type": "string",
                    "choices": [
                        {"value": "outage", "description": "Service unavailable"},
                        {"value": "billing", "description": "Payments"},
                    ],
                    "instructions": "Which team should handle this?",
                    "return_probabilities": True,
                    "permutations": 1,
                }
            },
            None,
        ),
        (
            "scalar_array",
            "The only listed skills are Rust and Python.",
            {
                "skills": {
                    "type": "array",
                    "instructions": "Extract the listed skills in their original order. Do not add any others.",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
            {"skills": ["Rust", "Python"]},
        ),
        (
            "object_array",
            "Invoice line: item Apple, quantity 3.",
            {
                "rows": {
                    "type": "array",
                    "instructions": "Extract invoice lines.",
                    "items": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}},
                    },
                    "minItems": 1,
                    "maxItems": 1,
                }
            },
            {"rows": [{"item": "Apple", "quantity": 3}]},
        ),
        (
            "continuation",
            "The only listed skills are Rust and Python.",
            {
                "skills": {
                    "type": "array",
                    "instructions": "Extract the next listed skill in order, without repeating existing skills.",
                    "items": {"type": "string"},
                    "continue_from": ["Rust"],
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
            {"skills": ["Rust", "Python"]},
        ),
        (
            "ordinal",
            "The customer explicitly says they are furious.",
            {
                "tone": {
                    "type": "number",
                    "instructions": "How upset is the customer?",
                    "levels": [{"label": "calm"}, {"label": "annoyed"}, {"label": "furious"}],
                    "return_probabilities": True,
                }
            },
            None,
        ),
        (
            "large_enum",
            "The stated code is category_17.",
            {
                "code": {
                    "type": "string",
                    "enum": [f"category_{index}" for index in range(32)],
                    "instructions": "Which code is explicitly stated?",
                    "return_probabilities": True,
                    "permutations": 1,
                }
            },
            None,
        ),
    ]
    cases.extend(
        [
            (
                "numeric_limits",
                "Reference 12345, amount 12.34.",
                {
                    "reference": {"type": "integer", "instructions": "Extract the reference."},
                    "amount": {"type": "number", "instructions": "Extract the amount."},
                },
                {"reference": 12345, "amount": 12.34},
            ),
            (
                "dice_sample",
                "A single roll of a fair die.",
                {
                    "roll": {
                        "type": "string",
                        "enum": ["one", "two", "three", "four", "five", "six"],
                        "instructions": "What number will come up on a single roll of a fair six-sided die?",
                        "permutations": "auto",
                        "return_probabilities": True,
                    },
                },
                None,
            ),
            (
                "actual_prompt",
                "The number is 7.",
                {
                    "n": {"type": "integer", "instructions": "Extract the stated number."},
                },
                {"n": 7},
            ),
        ]
    )
    case_options = {
        "numeric_limits": {"numeric_max_digits": 5},
        "dice_sample": {"mode": "sample", "temperature": 1, "seed": 42},
        "actual_prompt": {"print_final_prompt": True},
    }
    records = []
    try:
        for name, context, questions, expected in cases:
            start = time.perf_counter()
            try:
                response = call(
                    library.tj_engine_generate_json,
                    handle,
                    json.dumps(
                        {"context": context, "questions": questions, **case_options.get(name, {})}
                    ).encode(),
                )
                if name == "large_enum":
                    assert len(response["result"]["code"]["distribution"]) == 32
                if name == "dice_sample":
                    probabilities = response["result"]["roll"]["probabilities"]
                    assert len(probabilities) == 6 and abs(sum(probabilities.values()) - 1) < 1e-9
                if name == "actual_prompt":
                    assert (
                        response["prompts"]
                        and "Extract the stated number" in response["prompts"][0]
                    )
                if name == "ordinal":
                    assert 0 <= response["result"]["tone"]["value"] <= 2
                record = {
                    "case": name,
                    "response": response,
                    "contract_passed": True,
                    "answer_matched": None if expected is None else response["result"] == expected,
                }
            except Exception as error:
                record = {"case": name, "contract_passed": False, "error": str(error)}
            record["seconds"] = round(time.perf_counter() - start, 4)
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        check(library.tj_engine_close(handle))
        if directory:
            directory.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"config": config, "records": records,
                    "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
                    "shim_sha256": hashlib.sha256((args.library_dir / "turbojev_llama_shim.dll").read_bytes()).hexdigest(),
                    "model_sha256": hashlib.file_digest(args.model.open("rb"), "sha256").hexdigest()}, indent=2) + "\n", encoding="utf-8"
    )
    if any(not record["contract_passed"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
