let runtime=null, descriptor=null, engine=null;
const reply=(id,ok,value)=>postMessage(ok?{id,ok,value}:{id,ok,error:value?.message??String(value)});
onmessage=async({data:{id,op,args}})=>{try{
  if(op==="load"){
    const runtimeModule=await import(args.runtimeModuleUrl); runtime=await runtimeModule.createRuntime(); descriptor=await runtime.initialize({model:args.model,...(args.runtimeOptions??{})});
    const wasm=await import(args.wasmModuleUrl); if(typeof wasm.default==="function") await wasm.default(); if(!wasm.WasmEngine) throw new Error("WASM module does not export WasmEngine");
    engine=new wasm.WasmEngine(runtime,JSON.stringify(descriptor),args.prompt?JSON.stringify(args.prompt):undefined,args.cyclicOrderRobustness===true); return reply(id,true,descriptor);
  }
  if(!engine)throw new Error("ENGINE_NOT_LOADED");
  if(op==="evaluate")return reply(id,true,JSON.parse(await engine.evaluate_json(JSON.stringify(args.request))));
  if(op==="compatible")return reply(id,true,JSON.parse(await engine.evaluate_compatible_json(args.profile,JSON.stringify(args.request))));
  if(op==="info")return reply(id,true,descriptor);
  if(op==="close"){await runtime?.close?.();engine?.free?.();runtime=null;engine=null;return reply(id,true,null);}
  throw new Error(`unknown op ${op}`);
}catch(e){reply(id,false,e);}};
