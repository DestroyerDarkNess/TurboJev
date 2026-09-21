import test from "node:test"; import assert from "node:assert/strict"; import { mkdtemp, writeFile } from "node:fs/promises"; import { tmpdir } from "node:os"; import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveModel } from "../dist/model.js";
test("local and file model resolution is host-owned",async()=>{const d=await mkdtemp(join(tmpdir(),"tj-"));const p=join(d,"m.gguf");await writeFile(p,"x");assert.equal(await resolveModel({model:p}),p);assert.equal(await resolveModel({model:pathToFileURL(p).href}),p);});
