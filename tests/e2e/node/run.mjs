import { readFile } from "node:fs/promises";
import { TurboJev } from "../../../bindings/node/dist/index.js";
const [model, requestPath, nativePath] = process.argv.slice(2);
if (!model || !requestPath || !nativePath) throw new Error("usage: node run.mjs <model> <request.json> <native.node>");
const request = JSON.parse(await readFile(requestPath, "utf8"));
const jev = await TurboJev.load({ model, nativePath, nGpuLayers: 0 });
try {
  const response = await jev.evaluate(request);
  for (const id of ["route", "urgent", "severity"]) {
    if (!response?.answers?.[id]) throw new Error(`missing typed answer: ${id}`);
  }
  console.log(JSON.stringify({ language: "node-ts", response }));
} finally {
  await jev.close();
}
