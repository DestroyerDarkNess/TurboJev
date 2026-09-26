#!/usr/bin/env python3
from __future__ import annotations
import json, py_compile
from pathlib import Path
from jsonschema import Draft202012Validator
ROOT=Path(__file__).resolve().parents[1]
def load(p): return json.loads(p.read_text())
matrix=load(ROOT/'tests/e2e/matrix.json'); schema=load(ROOT/'schemas/turbojev.e2e-language-matrix.schema.json')
errors=list(Draft202012Validator(schema).iter_errors(matrix))
if errors: raise SystemExit('; '.join(e.message for e in errors))
ids={x['id'] for x in matrix['languages'] if x['required']}
assert {'node-ts','python','c','dotnet'} <= ids

plan=load(ROOT/'tests/e2e/final-plan.json')
plan_schema=load(ROOT/'schemas/turbojev.final-e2e-plan.schema.json')
plan_errors=list(Draft202012Validator(plan_schema).iter_errors(plan))
if plan_errors: raise SystemExit('; '.join(e.message for e in plan_errors))
assert set(plan['languages']) == {'node-ts','python','c','dotnet'}
assert plan['model']['sha256']=='478cca924101ed67bb2751f597be41532b2f4817ee2ab1b0d7cc379c76e79faf'
assert plan['model']['sizeBytes']==88202816
assert plan['rust']=='1.82.0' and plan['node']=='22' and plan['python']=='3.12' and plan['dotnet']=='8.0'
assert (ROOT/'crates/turbojev-runtime-llamacpp/native/CMakeLists.txt').is_file()
assert (ROOT/'crates/turbojev-runtime-llamacpp/examples/real_fused_qualification.rs').is_file()
assert (ROOT/'.github/workflows/real-e2e.yml').is_file()

request=load(ROOT/'tests/e2e/request.json')
questions=request.get('questions',{})
assert questions.get('route',{}).get('type')=='choice'
assert questions.get('urgent',{}).get('type') in {'boolean','noul'}
assert questions.get('severity',{}).get('type')=='score'
for rel in ['tests/e2e/node/run.mjs','tests/e2e/python/run.py','tests/e2e/c/main.c','tests/e2e/dotnet/Program.cs','tests/e2e/dotnet/TurboJev.E2E.csproj']:
    assert (ROOT/rel).is_file(), rel
for rel in ['scripts/e2e_language_matrix.py','scripts/final_e2e_linux.py']:
    if (ROOT/rel).exists(): py_compile.compile(str(ROOT/rel),doraise=True)
text=(ROOT/'scripts/e2e_language_matrix.py').read_text()
for snippet in ['turbojev.e2e-language-report.v2','semanticParity','responseSha256','generated_tokens']:
    assert snippet in text, snippet
assert '"-c", "Release", "--no-build"' in text, '.NET E2E must execute the already-built Release artifact'
final_text=(ROOT/'scripts/final_e2e_linux.py').read_text()
for snippet in ['turbojev.final-validation-report.v2','require_toolchain_versions','hardware_identity','--driver','cargoLockSha256','nodePackageLockSha256','cargo", "generate-lockfile','--locked','npm", "ci']:
    assert snippet in final_text, snippet
for rel in ['bindings/node/package.json','bindings/web/package.json','bindings/web-runtimes/transformersjs/package.json','bindings/web-runtimes/llamacpp-wasm/package.json']:
    package=load(ROOT/rel)
    assert not str(package.get('devDependencies',{}).get('typescript','')).startswith('^'), rel
workspace_cargo = (ROOT / 'Cargo.toml').read_text()
for version in ['=2.7.1', '=1.0.217', '=1.0.135', '=2.0.11']:
    assert version in workspace_cargo, f'workspace missing exact dependency pin {version}'

# Host/binding dependencies most sensitive to upstream MSRV/API drift are
# exact-pinned; the final run additionally freezes the complete transitive graph
# in Cargo.lock and records its SHA-256.
exact_cargo = {
    'crates/turbojev-node/Cargo.toml': ['=3.0.0', '=2.2.1'],
    'crates/turbojev-python/Cargo.toml': ['=0.23.4'],
    'crates/turbojev-wasm/Cargo.toml': ['=0.2.100', '=0.4.50', '=0.3.77', '=0.6.5'],
    'crates/turbojev-jni/Cargo.toml': ['=0.21.1'],
}
for rel, versions in exact_cargo.items():
    cargo_text = (ROOT / rel).read_text()
    for version in versions:
        assert version in cargo_text, f'{rel}: missing exact dependency pin {version}'

dotnet_text=(ROOT/'bindings/dotnet/TurboJev/TurboJevEngine.cs').read_text()
for snippet in ['catch {','queue.CompleteAdding()','thread.Join()','queue.Dispose()']:
    assert snippet in dotnet_text, f'.NET load-failure cleanup missing {snippet}'
assert 'finally {' in dotnet_text and 'failed native close' in dotnet_text, '.NET DisposeAsync must tear down its affinity worker even when native close fails'

workflow_text = (ROOT / '.github/workflows/real-e2e.yml').read_text()
assert '--skip-workspace-gates' not in workflow_text, 'real E2E workflow must run locked workspace gates inside final orchestrator'
assert 'scripts/final_e2e_linux.py' in workflow_text
print('OK: v0.28.2 cross-language E2E + reproducibility/lifecycle fixtures PASS')
