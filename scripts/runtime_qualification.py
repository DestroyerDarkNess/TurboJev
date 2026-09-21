#!/usr/bin/env python3
"""Build deterministic runtime fingerprint/qualification JSON for real-runtime evidence."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest='cmd',required=True)
    f=sub.add_parser('fingerprint')
    f.add_argument('--model',type=Path,required=True); f.add_argument('--runtime-revision',required=True)
    f.add_argument('--backend',required=True); f.add_argument('--device',required=True)
    f.add_argument('--backend-revision'); f.add_argument('--driver'); f.add_argument('--model-revision'); f.add_argument('--quantization')
    f.add_argument('--output',type=Path,required=True)
    m=sub.add_parser('manifest')
    m.add_argument('--fingerprint',type=Path,required=True); m.add_argument('--evidence',type=Path,required=True)
    m.add_argument('--id',required=True); m.add_argument('--harness-revision',default='turbojev-real-runtime-conformance-v1')
    m.add_argument('--absolute-tolerance',type=float,default=1e-5); m.add_argument('--relative-tolerance',type=float,default=1e-5)
    m.add_argument('--max-fused-items',type=int,required=True); m.add_argument('--estimated-fused-item-bytes',type=int)
    m.add_argument('--cooperative-cancellation',action='store_true'); m.add_argument('--hard-deadlines',action='store_true')
    m.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.cmd=='fingerprint':
        model=a.model.resolve()
        obj={'runtime':'llama.cpp','runtimeRevision':a.runtime_revision,'shimAbi':5,'backend':a.backend,'backendRevision':a.backend_revision,
             'hardware':{'architecture':platform.machine() or 'unknown','device':a.device,'driver':a.driver,'os':platform.system().lower() or None},
             'model':{'source':str(model),'sha256':sha256(model),'revision':a.model_revision,'quantization':a.quantization}}
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(obj,indent=2)+'\n')
        print(json.dumps(obj,indent=2)); return 0
    fingerprint=json.loads(a.fingerprint.read_text())
    evidence=json.loads(a.evidence.read_text())
    required=['rounds','requests','maxAbsLogitDelta','maxRelativeLogitDelta','responseMismatches','isolationFailures','nonFiniteValues']
    missing=[k for k in required if k not in evidence]
    if missing: raise SystemExit(f'missing evidence fields: {missing}')
    passed=(evidence['responseMismatches']==0 and evidence['isolationFailures']==0 and evidence['nonFiniteValues']==0
            and evidence['maxAbsLogitDelta']<=a.absolute_tolerance and evidence['maxRelativeLogitDelta']<=a.relative_tolerance)
    summary={k:evidence[k] for k in required}; summary['passed']=passed
    obj={'contract':'turbojev.runtime-qualification.v1','id':a.id,'fingerprint':fingerprint,'harnessRevision':a.harness_revision,
         'tolerance':{'absolute':a.absolute_tolerance,'relative':a.relative_tolerance},'summary':summary,
         'capabilities':{'fusedBatch':passed,'cooperativeCancellation':bool(a.cooperative_cancellation and passed),
                         'hardDeadlines':bool(a.hard_deadlines and passed),'maxFusedItems':a.max_fused_items if passed else None,
                         'estimatedFusedItemBytes':a.estimated_fused_item_bytes if passed else None},
         'evidenceSha256':sha256(a.evidence)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(obj,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
