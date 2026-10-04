"""Audit Snake with real C ABI inference and the demo's unchanged game physics.
No legal-action filtering, shortest-path hints, or host replacement is supplied to the model.
"""
import argparse, ast, ctypes as C, hashlib, json, os, time, types
from pathlib import Path

class NativeEngine:
    def __init__(self,args):
        self.paths=[os.add_dll_directory(str(p.resolve())) for p in args.library_dir] if os.name=='nt' else []
        self.lib=C.CDLL(str(args.library.resolve())); self.handle=C.c_void_p()
        self.lib.tj_engine_open_json.argtypes=[C.c_char_p,C.POINTER(C.c_void_p)]
        self.lib.tj_engine_evaluate_json.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_void_p)]
        self.lib.tj_engine_close.argtypes=[C.c_void_p];self.lib.tj_string_free.argtypes=[C.c_void_p]
        self.lib.tj_last_error_message.restype=C.c_char_p
        self.config={'model':str(args.model.resolve()),'device':'cpu','nCtx':4096,'nBatch':512,'nSeqMax':1,'nThreads':4,'nThreadsBatch':4,'thinking':args.thinking,'maxThinkingTokens':args.max_thinking_tokens,'execution':{'orderRobustness':'none'}}
        self.check(self.lib.tj_engine_open_json(json.dumps(self.config).encode(),C.byref(self.handle)))
    def check(self,status):
        if status:raise RuntimeError(self.lib.tj_last_error_message().decode())
    def evaluate(self,request):
        output=C.c_void_p();self.check(self.lib.tj_engine_evaluate_json(self.handle,json.dumps(request).encode(),C.byref(output)))
        try:return json.loads(C.string_at(output).decode())
        finally:self.lib.tj_string_free(output)
    def close(self):self.check(self.lib.tj_engine_close(self.handle))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--library-dir',type=Path,action='append',default=[])
    p.add_argument('--demo',type=Path,default=Path('other/TurboJev.Demos/snake-python/snake.py'))
    p.add_argument('--thinking',action='store_true');p.add_argument('--max-thinking-tokens',type=int,default=512)
    p.add_argument('--formats',nargs='+',choices=['coordinates','grid'],default=['coordinates','grid'])
    p.add_argument('--rotations',type=int,default=4);p.add_argument('--game-turns',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    # Import only the observation/game implementation, without loading the
    # installed package's older DLL into this source-built C ABI process.
    tree=ast.parse(args.demo.read_text(encoding='utf-8'))
    tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.module=='turbojev')]
    demo=types.ModuleType('snake_demo_validation')
    demo.__file__=str(args.demo.resolve())
    exec(compile(tree,str(args.demo.resolve()),'exec'),demo.__dict__)
    records=[];games=[]
    started=time.perf_counter();engine=NativeEngine(args);load_seconds=time.perf_counter()-started
    # UI suppression changes no request, choice, food spawn or game physics.
    demo.HAS_RICH=False;demo.console=None;demo.draw_legacy=lambda *a:None
    try:
        for fmt in args.formats:
            for rotation in range(args.rotations):
                req=demo.make_move_request([(6,0),(6,1)],'up',(11,0),fmt)
                q=req['questions']['move'];entries=list(q['criteria'].items());q['criteria']=dict(entries[rotation:]+entries[:rotation])
                started=time.perf_counter()
                try:
                    response=engine.evaluate(req);choice=response['answers']['move']['choice']
                    after,ate,collision=demo.apply_move([(6,0),(6,1)],'up',(11,0),choice)
                    record={'format':fmt,'rotation':rotation,'request':req,'response':response,'choice':choice,'expected':'right','correct':choice=='right','collision':collision,'seconds':time.perf_counter()-started}
                except Exception as e:
                    record={'format':fmt,'rotation':rotation,'request':req,'correct':False,'error':str(e),'seconds':time.perf_counter()-started}
                records.append(record);print(json.dumps({k:v for k,v in record.items() if k not in ['request','response']}),flush=True)
            if args.game_turns:
                trace=args.output.with_name(args.output.stem+'-'+fmt+'-game.jsonl')
                try:
                    summary=demo.play(engine,str(args.model),args.game_turns,load_seconds,'none',0,trace,fmt,args.thinking)
                    games.append({'format':fmt,'summary':summary,'trace':str(trace)})
                except Exception as e:games.append({'format':fmt,'error':str(e),'trace':str(trace)})
    finally:engine.close()
    with args.model.open('rb') as file:checksum=hashlib.file_digest(file,'sha256').hexdigest()
    report={'model':str(args.model.resolve()),'model_sha256':checksum,'config':engine.config,'demo_sha256':hashlib.sha256(args.demo.read_bytes()).hexdigest(),'correct':sum(r['correct'] for r in records),'total':len(records),'records':records,'games':games,'host_selects_or_replaces_actions':False,'inference':'source-built C ABI; demo still uses registry packages when run normally'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Saved {args.output}: {report['correct']}/{report['total']}",flush=True)

if __name__=='__main__':main()
