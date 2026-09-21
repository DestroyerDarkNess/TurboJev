from __future__ import annotations
import json, sys
from pathlib import Path
from turbojev import TurboJev

if len(sys.argv) != 3:
    raise SystemExit("usage: run.py <model> <request.json>")
model, request_path = sys.argv[1:]
request = json.loads(Path(request_path).read_text(encoding="utf-8"))
with TurboJev.load(model, n_gpu_layers=0) as jev:
    response = jev.evaluate(request)
for identifier in ("route", "urgent", "severity"):
    if identifier not in response.get("answers", {}):
        raise SystemExit(f"missing typed answer: {identifier}")
print(json.dumps({"language": "python", "response": response}, ensure_ascii=False, separators=(",", ":")))
