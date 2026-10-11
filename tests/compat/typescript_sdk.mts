import { TurboJev, Dice, type GenerateRequest } from "../../bindings/node/dist/index.js";
import { TurboJevWeb } from "../../bindings/web/dist/index.js";

const request: GenerateRequest = {
  context: "A fair six-sided die.",
  questions: { roll: new Dice(1).toSchema() },
};

export async function nativeExample(model: string): Promise<void> {
  const engine = await TurboJev.load({ model, device: "cpu" });
  try {
    const response = await engine.generate(request);
    console.log(response.result.roll, response.usage.decisions);
  } finally {
    await engine.close();
  }
}

export async function browserExample(model: string): Promise<void> {
  const engine = await TurboJevWeb.load({
    model,
    runtimeModuleUrl: "/runtime/index.js",
    wasmModuleUrl: "/turbojev_wasm.js",
  });
  try {
    const response = await engine.generate(request);
    console.log(response.result.roll, response.usage.decisions);
  } finally {
    await engine.close();
  }
}
