import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import type { TurboJev as Engine } from "../../../bindings/node/dist/index.js";

const stage = process.env.TURBOJEV_ARTIFACTS;
if (!stage) throw new Error("Use python samples/run.py ts");

const { TurboJev } = (await import(pathToFileURL(join(stage, "node/dist/index.js")).href)) as {
  TurboJev: typeof Engine;
};

const model = process.argv[2];
const argumentsAfterModel = process.argv.slice(3);
const mediaFlag = argumentsAfterModel.indexOf("--media");

function valueAfter(flag: string, fallback?: string): string | undefined {
  const index = argumentsAfterModel.indexOf(flag);
  return index >= 0 ? argumentsAfterModel[index + 1] : fallback;
}

if (argumentsAfterModel.includes("--generate")) {
  const engine = await TurboJev.load({
    model,
    nativePath: join(stage, "turbojev.node"),
    device: "cpu",
    nCtx: 8192,
    nSeqMax: 1,
    nThreads: 4,
  });
  try {
    const request = JSON.parse(
      readFileSync(new URL("../../typed-fields/request.json", import.meta.url), "utf8"),
    );
    console.log(JSON.stringify(await engine.generate(request), null, 2));
  } finally {
    await engine.close();
  }
} else if (mediaFlag >= 0) {
  const mediaPath = argumentsAfterModel[mediaFlag + 1];
  const modality = valueAfter("--modality", "image");
  const question = valueAfter("--question", "What is shown?");
  const optionsFlag = argumentsAfterModel.indexOf("--options");
  const options =
    optionsFlag >= 0 ? argumentsAfterModel.slice(optionsFlag + 1) : ["A cat", "A dog"];
  const workerPython = process.env.TURBOJEV_MULTIMODAL_PYTHON;
  const workerPath = process.env.TURBOJEV_MULTIMODAL_WORKER;
  const workerModel = process.env.TURBOJEV_MULTIMODAL_MODEL;

  if (!mediaPath || !workerPython || !workerPath || !workerModel) {
    throw new Error("First run: python samples/multimodal/setup.py vision");
  }
  if (modality !== "image" && modality !== "video") {
    throw new Error("The SmolVLM2 sample accepts --modality image or video");
  }
  if (options.length < 2 || options.length > 26 || new Set(options).size !== options.length) {
    throw new Error("Provide 2 to 26 distinct options after --options");
  }
  if (statSync(mediaPath).size > 24 * 1024 * 1024) {
    throw new Error("Media must be at most 24 MiB");
  }

  const engine = await TurboJev.load({
    model: workerModel,
    nativePath: join(stage, "turbojev.node"),
    worker: {
      program: workerPython,
      args: [workerPath, "vision", workerModel],
      timeoutSeconds: 600,
    },
  });
  try {
    const result = await engine.evaluate({
      state: {
        contract: "turbojev.media.v1",
        modality,
        base64: readFileSync(mediaPath).toString("base64"),
      },
      questions: {
        scene: {
          type: "choice",
          instructions: question,
          criteria: Object.fromEntries(
            options.map((option, index) => [`option_${index + 1}`, option]),
          ),
        },
      },
    });
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await engine.close();
  }
} else {
  const engine = await TurboJev.load({
    model,
    nativePath: join(stage, "turbojev.node"),
    // CPU is the default. Set device: "gpu" to request GPU offload explicitly.
    device: "cpu",
    nGpuLayers: 0,
    nThreads: 4,
  });
  try {
    const result = await engine.evaluate({
      state: { message: argumentsAfterModel[0] ?? "My package arrived broken" },
      questions: {
        department: {
          type: "choice",
          instructions: "Which department should handle this message?",
          criteria: {
            billing: "Payments and refunds",
            shipping: "Delivery",
            technical: "Product issue",
            other: "Other",
          },
        },
      },
    });
    const answer = (
      result as {
        answers: Record<string, { choice: string; probabilities: Record<string, number> }>;
      }
    ).answers.department;
    console.log(`Department: ${answer.choice}`);
    console.table(
      Object.entries(answer.probabilities).map(([department, probability]) => ({
        department,
        probability: `${(probability * 100).toFixed(1)}%`,
      })),
    );
  } finally {
    await engine.close();
  }
}
