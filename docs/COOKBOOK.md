# TurboJev cookbook

These workflows adapt all four articles in the [TypeLLM cookbook](https://typellm.ai/cookbook)
to local inference. Use a model suited to the evidence and decision: typed output does not
guarantee accuracy. This structured SDK extension is a source change awaiting publication;
build this revision for new APIs. Registry demos retain the last published packages.

## Extract a dense page in batches

Use a vision GGUF and its matching projector. The model supplies rows and the continuation
decision; host code only enforces a bounded loop and stops when no progress is made.

```python
from pathlib import Path
from turbojev import TurboJev

rows = []
with TurboJev.load("vision.gguf", mmproj="mmproj.gguf", device="cpu", n_ctx=8192) as client:
    for _ in range(10):
        response = client.generate(
            context="Extract every invoice line in reading order. Never repeat existing rows.",
            images=[Path("invoice.png")],
            text_max_tokens=512,
            questions={
                "rows": {
                    "type": "array",
                    "instructions": "Return the next invoice lines, then stop when none remain.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "amount": {"type": "number"},
                        },
                    },
                    "continue_from": rows,
                },
                "has_next": {
                    "type": "boolean",
                    "depends_on": ["rows"],
                    "instructions": "Are any printed invoice rows still absent from rows?",
                },
            },
        )
        next_rows = response["result"]["rows"]
        advanced = len(next_rows) > len(rows)
        rows = next_rows
        if not response["result"]["has_next"] or not advanced:
            break
print(rows)
```

Without a maximum, each call can append 50 new items. An explicit `maxItems` instead
limits the total array, including its prefix, to 1–50. Each object is actual constrained
generation; this is not single-token classification. The model and context limit determine
usable page complexity.

## More than 255 categories

Schemas accept up to 512 typed values. Native GGUF and text ONNX compare complete value
continuations for more than 26 values, with the complete set in the prompt.

```python
from turbojev import TurboJev

categories = [f"category_{index}" for index in range(418)]
with TurboJev.load("model.gguf", device="cpu", n_ctx=8192) as client:
    response = client.generate(
        context="The document explicitly identifies category_17.",
        questions={
            "category": {
                "type": "string",
                "enum": categories,
                "instructions": "Which exact category is stated?",
                "return_probabilities": True,
                "permutations": 1,
            }
        },
    )
    print(response["result"]["category"])
```

Replace identifiers with meaningful application categories/descriptions. This shows the API,
not measured 418-class accuracy. GGUF/WASM requires a value-scorer plugin and otherwise errors.
A large enum has a higher prompt/scoring cost; comparing only its first tokens is incorrect.

## Hierarchical classification

Choose a department, then ask only its category. Unchosen branches appear in `skipped`
and are absent from the result. Probability wrappers are unwrapped in dependent prompts.

```python
from turbojev import TurboJev

with TurboJev.load("model.gguf", device="cpu") as client:
    response = client.generate(
        context="I was charged twice and want a refund.",
        questions={
            "department": {
                "type": "string",
                "enum": ["billing", "technical"],
                "instructions": "Which department should handle this?",
            },
            "billing_category": {
                "type": "string",
                "enum": ["refund", "invoice", "payment_method"],
                "instructions": "What billing issue is described?",
                "when": {"department": "billing"},
            },
            "technical_category": {
                "type": "string",
                "enum": ["login", "outage", "performance"],
                "instructions": "What technical issue is described?",
                "when": {"department": "technical"},
            },
        },
    )
    print(response["result"], response["skipped"])
```

The described-choice/confidence variant is in
[hierarchical.json](../samples/typed-fields/hierarchical.json).

## Discover and reuse categories

Offer current categories plus `new`. Only if selected, ask for a category name.
The application reviews and stores the proposal; TurboJev does not mutate its taxonomy.

```python
from turbojev import TurboJev

categories = ["groceries", "transport", "utilities"]
with TurboJev.load("model.gguf", device="cpu") as client:
    response = client.generate(
        context="Receipt: dog grooming, 45.00.",
        questions={
            "category": {
                "type": "string",
                "enum": [*categories, "new"],
                "instructions": "Choose an existing expense category, or new if none fits.",
            },
            "new_category": {
                "type": "string",
                "instructions": "Propose a short reusable expense category name.",
                "when": {"category": "new"},
            },
        },
    )
    print(response["result"])
```

## Inspect the contract without loading a model

```python
from turbojev import Choice, compile_json_schema

schema = {
    "type": "object",
    "properties": {
        "team": Choice("Which team?", {"billing": "Payments", "outage": "Service down"}).to_schema()
    },
}
for decision in compile_json_schema(schema):
    print(decision.name, decision.choices)
```

Compilation runs Rust validation. The metadata contains semantic values rather than
unchecked model-specific token IDs. Generation still validates each request.

## Select latency and cancellation deliberately

Set `permutations: 1` for one finite evaluation. Probability requests without a budget
default to balanced layouts for small unordered fields. Ordered levels remain single-pass.

```python
from turbojev import GenerationCancelled, GenerationControl, GenerationTimeout, TurboJev

with TurboJev.load("model.gguf", device="cpu") as client:
    control = GenerationControl(timeout=30)
    try:
        response = client.generate(
            context="The service is down.",
            questions={"outage": {"type": "boolean", "permutations": 1}},
            control=control,
        )
        print(response["result"])
    except GenerationCancelled:
        print("Cancelled between model operations")
    except GenerationTimeout:
        print("Deadline expired between model operations")
```

Another thread may call `control.cancel()`. Native cancellation does not interrupt an
active decode; it aborts the workflow after that operation. Node/browser use
`client.generate(request, {timeoutMs: 30000, signal})`; C# accepts `TimeSpan` and
`CancellationToken`. An aborted workflow is not a model-selected null answer.
