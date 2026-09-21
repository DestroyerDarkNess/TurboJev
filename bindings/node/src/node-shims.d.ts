declare const process: { pid:number; platform:string; arch:string; cwd():string; env:Record<string,string|undefined> };
declare module 'node:worker_threads' { export class Worker { constructor(url:URL, options?:any); on(event:string, cb:(...args:any[])=>void):this; postMessage(value:any):void; terminate():Promise<number>; } }
declare module 'node:fs' { export function existsSync(path:string):boolean; export function createWriteStream(path:string,opts?:any):any; export function mkdirSync(path:string,opts?:any):void; export function renameSync(a:string,b:string):void; export function rmSync(path:string,opts?:any):void; }
declare module 'node:module' { export function createRequire(url:string): (id:string)=>any; }
declare module 'node:path' { export function resolve(...parts:string[]):string; export function join(...parts:string[]):string; export function basename(path:string):string; }
declare module 'node:stream/promises' { export function pipeline(...streams:any[]):Promise<void>; }
declare module 'node:stream' { export class Readable { static fromWeb(stream:any):any; } }
declare module 'node:crypto' { export function createHash(name:string): { update(data:string):any; digest(enc:string):string }; }
