"""Run with an installed wheel, including the oldest declared CPython version."""
import json
import platform
import struct
from typing import get_type_hints

from turbojev import Dice, GenerationTimeout, LoadOptions, TurboJev, candidate_softmax, compile_json_schema

assert get_type_hints(LoadOptions)["model"] is str
get_type_hints(TurboJev.generate)
get_type_hints(compile_json_schema)

schema = {"type": "object", "properties": {
    "label": {"type": "string", "enum": ["niño 🐈 中文"], "permutations": 1},
    "count": {"type": "integer", "enum": [7]},
}}
assert len(compile_json_schema(schema)) == 2
assert candidate_softmax({"café": 10000, "茶": 10000}) == {"café": 0.5, "茶": 0.5}
assert "six" in json.dumps(Dice(permutations=1).to_schema())
with TurboJev.load("fixture://version-compatibility") as engine:
    result = engine.generate_typed(context="niño 🐈 中文", schema=schema)
    assert result.result == {"label": "niño 🐈 中文", "count": 7}
    assert result.usage.decisions == 2
    try:
        engine.generate(context="facts", schema=schema, timeout=0)
        raise AssertionError("An expired call returned a result")
    except GenerationTimeout:
        pass

print(json.dumps({"language": "python", "version": platform.python_version(),
                  "bits": struct.calcsize("P") * 8, "passed": True}))
