import { createWriteStream, existsSync, mkdirSync, renameSync, rmSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import type { LoadOptions } from "./types.js";

function hfUrl(spec:string):string {
  const body=spec.slice(5); const parts=body.split("/");
  if (parts.length<3) throw new Error("INVALID_OPTIONS: hf:// source must be hf://owner/repo/file");
  const owner=parts.shift()!, repo=parts.shift()!, file=parts.join("/");
  return `https://huggingface.co/${owner}/${repo}/resolve/main/${file}`;
}
export async function resolveModel(options:LoadOptions):Promise<string> {
  const source=options.model;
  if (source.startsWith("fixture://")) return source;
  if (source.startsWith("file://")) return fileURLToPath(source);
  if (!source.startsWith("http://") && !source.startsWith("https://") && !source.startsWith("hf://")) return resolve(source);
  const url=source.startsWith("hf://") ? hfUrl(source) : source;
  const cache=resolve(options.cacheDir ?? join(process.env.HOME ?? process.cwd(), ".cache", "turbojev")); mkdirSync(cache,{recursive:true});
  const hash=createHash("sha256").update(url).digest("hex").slice(0,24); const target=join(cache,`${hash}-${basename(new URL(url).pathname)||"model.bin"}`);
  if (existsSync(target)) return target;
  const tmp=`${target}.part-${process.pid}-${Date.now()}`;
  const response=await fetch(url); if(!response.ok||!response.body) throw new Error(`DOWNLOAD_ERROR: ${response.status} ${response.statusText}`);
  try { await pipeline(Readable.fromWeb(response.body as any), createWriteStream(tmp,{flags:"wx"})); renameSync(tmp,target); }
  catch(e){ rmSync(tmp,{force:true}); throw e; }
  return target;
}
