from __future__ import annotations
import json, sys
from pathlib import Path
from turbojev import TurboJev

if len(sys.argv) not in (3, 4):
    raise SystemExit("usage: run.py <model> <request.json> [mmproj.gguf]")
model, request_path = sys.argv[1:3]
mmproj = sys.argv[3] if len(sys.argv) == 4 and sys.argv[3] else None
request = json.loads(Path(request_path).read_text(encoding="utf-8"))
options = {"n_gpu_layers": 0}
if mmproj:
    options["mmproj"] = mmproj
with TurboJev.load(model, **options) as jev:
    response = jev.evaluate(request)
print(json.dumps({"language": "python", "response": response}, ensure_ascii=False, separators=(",", ":")))
