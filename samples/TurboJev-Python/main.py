"""Idiomatic Python: context-managed local model and dictionary result."""
import sys
import json
from pathlib import Path
from turbojev import TurboJev

model = sys.argv[1]
message = sys.argv[2] if len(sys.argv) > 2 else "My package arrived broken."
if message == "--generate":
    request=json.loads(Path(__file__).resolve().parents[1].joinpath("typed-fields/request.json").read_text(encoding="utf-8"))
    with TurboJev.load(model, device="cpu", n_ctx=8192, n_seq_max=1, n_threads=4) as engine:
        print(json.dumps(engine.generate(request),indent=2))
    raise SystemExit(0)
# CPU is the default. Set device="gpu" to request GPU offload explicitly.
with TurboJev.load(model, device="cpu", n_gpu_layers=0) as engine:
    result = engine.evaluate({
        "state": {"message": message},
        "questions": {"department": {
            "type": "choice",
            "instructions": "Which department should handle this message?",
            "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "Product issue", "other": "Other"},
        }},
    })["answers"]["department"]
    print(f"Department: {result['choice']}")
    for label, probability in result["probabilities"].items():
        print(f"  {label:12} {probability:6.1%}")
