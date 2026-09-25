"""Idiomatic Python: context-managed local model and dictionary result."""
import sys
from turbojev import TurboJev

model = sys.argv[1]
message = sys.argv[2] if len(sys.argv) > 2 else "My package arrived broken."
# CPU is the default. Set device="gpu" to request GPU offload explicitly.
with TurboJev.load(model, device="cpu", n_gpu_layers=0) as engine:
    result = engine.classify(message, ["billing", "shipping", "technical", "other"])
    print(f"Department: {result['choice']}")
    for label, probability in result["probabilities"].items():
        print(f"  {label:12} {probability:6.1%}")
