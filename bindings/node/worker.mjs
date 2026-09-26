import { parentPort, workerData } from "node:worker_threads";
import { createRequire } from "node:module";
const require=createRequire(import.meta.url); let engine; let initError;
try {
  const options=workerData.options;
  const candidates=[options.nativePath,process.env.TURBOJEV_NATIVE_PATH,"./turbojev.node",`@turbojev/node-${process.platform}-${process.arch}`].filter(Boolean);
  let mod; let last;
  for(const c of candidates){try{mod=require(c);break;}catch(e){last=e;}}
  if(!mod) throw last ?? new Error("native addon not found");
  engine=mod.NativeEngine.open(JSON.stringify(options));
}catch(e){initError=e;}
function answer(id,fn){try{if(initError)throw initError;parentPort.postMessage({id,ok:true,value:fn()});}catch(e){parentPort.postMessage({id,ok:false,error:e?.message??String(e)});}}
parentPort.on("message",({id,op,args})=>{switch(op){case"ready":answer(id,()=>"ok");break;case"evaluate":answer(id,()=>engine.evaluateJson(args.request));break;case"compatible":answer(id,()=>engine.evaluateCompatibleJson(args.profile,args.request));break;case"info":answer(id,()=>engine.infoJson());break;case"close":answer(id,()=>{engine?.close();engine=undefined;return"ok";});break;default:answer(id,()=>{throw new Error(`unknown op ${op}`)});}}
);
