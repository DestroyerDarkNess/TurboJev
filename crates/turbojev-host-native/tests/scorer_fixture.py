"""Protocol fixture only; this does not perform model inference."""
import json
import sys
import time

mode = sys.argv[1]
print(json.dumps({"contract": "turbojev.scorer.v1", "modalities": ["text", "image", "audio", "video"]}), flush=True)
if mode == "no_read":
    time.sleep(60)
for line in sys.stdin:
    batch = json.loads(line)
    if mode == "timeout":
        time.sleep(60)
    if mode == "error":
        print(json.dumps({"error": "unsupported codec"}), flush=True)
    else:
        scores = [{"id": task["id"], "logits": list(range(len(task["candidates"])))}
                  for task in batch["tasks"]]
        print(json.dumps({"model": "protocol-fixture", "scores": scores, "metrics": {}}), flush=True)
