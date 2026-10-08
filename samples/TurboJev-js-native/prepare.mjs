import { build } from "esbuild";
import { mkdir, copyFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";
const vendor = resolve("public/vendor");
await mkdir(`${vendor}/web/dist`, { recursive: true });
await build({
  entryPoints: ["../../bindings/web/src/index.ts"],
  outfile: `${vendor}/web/dist/index.js`,
  bundle: true,
  format: "esm",
  target: "es2022",
});
await copyFile("../../bindings/web/worker.js", `${vendor}/web/worker.js`);
await build({
  entryPoints: ["runtime.ts"],
  outfile: `${vendor}/runtime.js`,
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  nodePaths: [resolve("node_modules")],
  define: { "process.env.NODE_ENV": '"production"' },
});
await mkdir(`${vendor}/onnx`, { recursive: true });
for (const name of await readdir("node_modules/onnxruntime-web/dist")) {
  if (name.endsWith(".wasm") || name.endsWith(".mjs"))
    await copyFile(`node_modules/onnxruntime-web/dist/${name}`, `${vendor}/onnx/${name}`);
}
