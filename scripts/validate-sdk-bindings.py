"""Exercise the built Python extension and the source SDK, without substituting inference.

PYTHONPATH should point at the test installation containing this revision's native extension.
Model-free and fixture contract checks are explicitly separated from the optional real GGUF check.
"""

import argparse
import json
import threading

from turbojev import (
    ArrayField,
    Choice,
    GenerationCancelled,
    GenerationControl,
    GenerationTimeout,
    TurboJev,
    candidate_softmax,
    compile_json_schema,
    encode_image,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    args = parser.parse_args()
    schema = {
        "type": "object",
        "properties": {
            "team": Choice(
                "Which team?", {"outage": "Service down", "billing": "Payments"}
            ).to_schema(),
            "rows": {"type": "array", "items": {"type": "integer"}},
        },
    }
    compiled = compile_json_schema(schema)
    assert compiled[0].choices == ("outage", "billing")
    assert isinstance(compiled[1], ArrayField)
    assert candidate_softmax({"a": 10000, "b": 10000}, 0.0001) == {"a": 0.5, "b": 0.5}
    assert encode_image(b"image-bytes") == "aW1hZ2UtYnl0ZXM="
    with TurboJev.load("fixture://sdk-contract") as client:
        request = {
            "context": "fixture contract",
            "questions": {
                "person": {
                    "type": "object",
                    "properties": {"age": {"type": "integer", "enum": [37]}},
                }
            },
        }
        response = client.generate_typed(request)
        assert response.result == {"person": {"age": 37}}
        assert response.usage.decisions == 1
        control = GenerationControl()
        thread = threading.Thread(target=control.cancel)
        thread.start()
        thread.join()
        try:
            client.generate(request, control=control)
        except GenerationCancelled:
            pass
        else:
            raise AssertionError("Cancelled generation returned a result")
        try:
            client.generate(request, timeout=0)
        except GenerationTimeout:
            pass
        else:
            raise AssertionError("Expired generation returned a result")
        assert client.generate(request)["result"] == response.result
    print(
        "Python model-free compiler, typed results, image encoding and fixture control contracts passed."
    )
    if args.model:
        with TurboJev.load(
            args.model, device="cpu", n_threads=4, n_threads_batch=4, n_seq_max=1
        ) as client:
            response = client.generate_typed(
                context="Ada is 37 years old.",
                questions={
                    "person": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer", "instructions": "What is the person's age?"}
                        },
                    }
                },
            )
            assert response.result == {"person": {"age": 37}}
            # A Python timer must run during native inference: the extension releases
            # the GIL while preserving native thread affinity. The result is discarded
            # after the in-flight operation, not replaced by a fabricated answer.
            control = GenerationControl()
            timer = threading.Timer(0.1, control.cancel)
            timer.start()
            try:
                client.generate(
                    context="Ada is 37 years old.",
                    questions={"age": {"type": "integer"}},
                    control=control,
                )
            except GenerationCancelled:
                pass
            else:
                raise AssertionError("Cancellation did not run during native inference")
            finally:
                timer.join()
            print(
                json.dumps(
                    {
                        "real_model": args.model,
                        "response": response.result,
                        "in_flight_python_cancellation": "passed",
                    }
                )
            )


if __name__ == "__main__":
    main()
