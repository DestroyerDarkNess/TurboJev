import { join } from "node:path";
import { pathToFileURL } from "node:url";
import type { TurboJev as Engine } from "../../../bindings/node/dist/index.js";

const stage = process.env.TURBOJEV_ARTIFACTS;
if (!stage) throw new Error("Use python samples/run.py ts");
const { TurboJev } = await import(pathToFileURL(join(stage, "node/dist/index.js")).href) as { TurboJev: typeof Engine };
const engine = await TurboJev.load({model: process.argv[2], nativePath: join(stage, "turbojev.node"), nGpuLayers: 0, nThreads: 4});
try {
  const result = await engine.classify(process.argv[3] ?? "My package arrived broken", ["billing", "shipping", "technical", "other"]);
  console.log(`Department: ${result.choice}`);
  console.table(Object.entries(result.probabilities).map(([department, p]) => ({department, probability: `${(p * 100).toFixed(1)}%`})));
} finally {
  await engine.close();
}
