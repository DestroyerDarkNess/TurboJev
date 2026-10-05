"""Verify real GGUF image field generation via an explicit development C ABI."""
import argparse
import base64
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"evals"))
from run import NativeClient, matched


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--mmproj",type=Path,required=True)
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--questions",type=Path,required=True)
    parser.add_argument("--expected",type=Path)
    parser.add_argument("--fields",help="Comma-separated explicit subset")
    parser.add_argument("--device",choices=["cpu","gpu"],default="cpu")
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--library-dir",type=Path,action="append",default=[])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    questions=json.loads(args.questions.read_text(encoding="utf-8"))
    if args.fields:
        names=set(args.fields.split(','))
        if names-set(questions): parser.error("unknown field")
        while True:
            expanded=names|{d for name in names for d in questions[name].get('depends_on',[])}
            if expanded==names: break
            names=expanded
        questions={n:q for n,q in questions.items() if n in names}
    expected=json.loads(args.expected.read_text(encoding="utf-8")) if args.expected else {}
    expected={n:v for n,v in expected.items() if n in questions}
    config={"model":str(args.model.resolve()),"mmproj":str(args.mmproj.resolve()),"device":args.device,"nCtx":8192,"nSeqMax":1,"nThreads":4,"nThreadsBatch":4}
    client=None
    request={"context":{"contract":"turbojev.media.v1","modality":"image","text":"Read the attached photo of a restaurant receipt.","base64":base64.b64encode(args.image.read_bytes()).decode()},
             "questions":questions,"text_max_tokens":64,"numeric_max_tokens":64}
    start=time.perf_counter()
    try:
        client=NativeClient(args.library,args.library_dir,config)
        response=client.generate(request)
        result={"config":config,"fields":list(questions),"subset":bool(args.fields),"response":response,"expected":expected,"matches":{n:matched(response['result'].get(n),v) for n,v in expected.items()}}
    except Exception as error:
        result={"config":config,"fields":list(questions),"error":str(error)}
    finally:
        if client: client.close()
    result["seconds"]=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    return int("error" in result)


if __name__=="__main__":
    raise SystemExit(main())
