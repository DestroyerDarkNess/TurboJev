export { TurboJevRemote, mediaEvidence, videoFrameEvidence } from "./remote.js";
export type { MediaEvidence, MediaModality, VideoFrame, VideoFrameInput } from "./remote.js";
export interface BrowserRuntimeDescriptor { name:string; model:string; codecProfile?:string; scoringMode?:"tokens"|"decisions"; modalities?:Array<"text"|"image"|"audio"|"video">; promptProfile?:{profile:string;systemInstruction:string;bestMatchInstruction:string}; capabilities:{nextTokenLogits:boolean;sequenceScores:boolean;sharedPrefix:boolean;parallelSuffixes:boolean;selectiveOutputHead:boolean;localInference:boolean}; }
export interface BrowserRuntime { initialize(options:unknown):Promise<BrowserRuntimeDescriptor>; encodePrompt(messagesJson:string):string; encodePromptWithAnswer(messagesJson:string,answer:string):string; decodeTokens(tokenIdsJson:string):string; scoreRuntimeBatch(runtimeBatchJson:string,planJson:string):Promise<string>|string; close():Promise<void>|void; }
/** Model-independent scoring; the provider owns media decoding and preprocessing. */
export interface BrowserDecisionRuntime {
  initialize(options:unknown):Promise<BrowserRuntimeDescriptor & {scoringMode:"decisions";modalities:Array<"text"|"image"|"audio"|"video">}>;
  scoreDecisionBatch(batchJson:string,planJson:string):Promise<string>|string;
  close():Promise<void>|void;
}
export interface WebLoadOptions { model:string; runtimeModuleUrl:string; wasmModuleUrl:string; runtimeOptions?:unknown; cyclicOrderRobustness?:boolean; prompt?:{profile?:string;systemInstruction?:string;bestMatchInstruction?:string}; workerUrl?:string; }
export class TurboJevWeb {
  #worker:Worker; #id=1; #pending=new Map<number,{resolve:(v:any)=>void,reject:(e:any)=>void}>(); #closed=false;
  private constructor(worker:Worker){this.#worker=worker;worker.onmessage=({data})=>{const p=this.#pending.get(data.id);if(!p)return;this.#pending.delete(data.id);data.ok?p.resolve(data.value):p.reject(new Error(data.error));};worker.onerror=e=>{for(const p of this.#pending.values())p.reject(e);this.#pending.clear();};}
  static async load(options:WebLoadOptions){const worker=new Worker(options.workerUrl??new URL("../worker.js",import.meta.url),{type:"module"});const x=new TurboJevWeb(worker);try{await x.#call("load",options);return x;}catch(error){worker.terminate();throw error;}}
  async evaluate(request:Record<string,unknown>){return this.#call("evaluate",{request});}
  async evaluateCompatible(profile:string,request:Record<string,unknown>){return this.#call("compatible",{profile,request});}
  async info(){return this.#call("info",{});}
  async close(){if(this.#closed)return;try{await this.#call("close",{});}finally{this.#closed=true;this.#worker.terminate();}}
  #call(op:string,args:any){if(this.#closed)return Promise.reject(new Error("ENGINE_CLOSED"));const id=this.#id++;return new Promise((resolve,reject)=>{this.#pending.set(id,{resolve,reject});this.#worker.postMessage({id,op,args});});}
}
