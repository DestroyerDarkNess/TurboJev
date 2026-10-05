import { Worker } from "node:worker_threads";
import { resolveModel } from "./model.js";
import { normalizeError, TurboJevError } from "./errors.js";
import type { EvaluationRequest, EvaluationResponse, GenerateRequest, GenerateResponse, LoadOptions } from "./types.js";
export * from "./types.js"; export { TurboJevError } from "./errors.js";

type Pending={resolve:(v:any)=>void,reject:(e:any)=>void};
export class TurboJev {
  #worker:Worker; #next=1; #pending=new Map<number,Pending>(); #closed=false;
  private constructor(worker:Worker){ this.#worker=worker; worker.on("message",(m:any)=>{const p=this.#pending.get(m.id);if(!p)return;this.#pending.delete(m.id);m.ok?p.resolve(m.value):p.reject(normalizeError(m.error));}); worker.on("error",(e:any)=>this.#fail(e)); worker.on("exit",(c:number)=>{if(!this.#closed&&c!==0)this.#fail(new Error(`worker exited ${c}`));}); }
  static async load(options:LoadOptions):Promise<TurboJev>{ const model=await resolveModel(options); const worker=new Worker(new URL("../worker.mjs",import.meta.url),{workerData:{options:{...options,model}}}); const jev=new TurboJev(worker); try { await jev.#call("ready",{}); return jev; } catch (error) { await worker.terminate(); throw error; } }
  async evaluate(request:EvaluationRequest):Promise<EvaluationResponse>{return JSON.parse(await this.#call("evaluate",{request:JSON.stringify(request)}));}
  async generate(request:GenerateRequest):Promise<GenerateResponse>{return JSON.parse(await this.#call("generate",{request:JSON.stringify(request)}));}
  async evaluateCompatible(profile:string,request:EvaluationRequest):Promise<Record<string,unknown>>{return JSON.parse(await this.#call("compatible",{profile,request:JSON.stringify(request)}));}
  async info():Promise<Record<string,unknown>>{return JSON.parse(await this.#call("info",{}));}
  async close():Promise<void>{if(this.#closed)return;try{await this.#call("close",{});}finally{this.#closed=true;await this.#worker.terminate();}}
  #call(op:string,args:any):Promise<any>{ if(this.#closed)return Promise.reject(new TurboJevError("ENGINE_CLOSED","engine is closed")); const id=this.#next++; return new Promise((resolve,reject)=>{this.#pending.set(id,{resolve,reject});this.#worker.postMessage({id,op,args});}); }
  #fail(error:unknown){for(const p of this.#pending.values())p.reject(normalizeError(error));this.#pending.clear();}
}
