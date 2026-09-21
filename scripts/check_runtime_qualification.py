#!/usr/bin/env python3
from __future__ import annotations
import json, py_compile
from pathlib import Path
from jsonschema import Draft202012Validator
ROOT=Path(__file__).resolve().parents[1]
schema=json.loads((ROOT/'schemas/turbojev.runtime-qualification.schema.json').read_text())
doc=json.loads((ROOT/'tests/qualification/llamacpp-cpu-qualified.json').read_text())
errors=list(Draft202012Validator(schema).iter_errors(doc))
if errors: raise SystemExit('; '.join(e.message for e in errors))
s=doc['summary']; t=doc['tolerance']
clean=(s['responseMismatches']==0 and s['isolationFailures']==0 and s['nonFiniteValues']==0 and s['maxAbsLogitDelta']<=t['absolute'] and s['maxRelativeLogitDelta']<=t['relative'])
assert clean == s['passed']
assert doc['fingerprint']['runtime']=='llama.cpp' and doc['fingerprint']['shimAbi']==5
assert doc['capabilities']['fusedBatch'] and doc['capabilities']['maxFusedItems']>=2
host=(ROOT/'crates/turbojev-host-native/src/lib.rs').read_text()
for snippet in ['llamacpp_qualification','runtime_fingerprint','manifest.authorize(current)','runtimeFingerprint.model.source']:
    assert snippet in host, snippet
py_compile.compile(str(ROOT/'scripts/runtime_qualification.py'),doraise=True)
print('OK: v0.26 runtime qualification evidence + host authorization PASS')
