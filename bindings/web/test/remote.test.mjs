import test from "node:test";
import assert from "node:assert/strict";
import { mediaEvidence, videoFrameEvidence, TurboJevRemote } from "../dist/remote.js";

test("media preserves binary bytes and rejects empty inputs", async () => {
  const result = await mediaEvidence(new Blob([new Uint8Array([0, 255])]), "image", "context");
  assert.equal(result.base64, "AP8=");
  assert.equal(result.contract, "turbojev.media.v1");
  await assert.rejects(mediaEvidence(new Blob([]), "audio"));
});

test("video media evidence carries ordered decoded frames and timestamps", async () => {
  const result = await videoFrameEvidence([
    { timestampMs: 0, image: new Blob([new Uint8Array([0, 255])]) },
    { timestampMs: 500, image: new Blob([new Uint8Array([1, 2])]) },
  ], "describe the clip");
  assert.deepEqual(result, {
    contract: "turbojev.media.v1",
    text: "describe the clip",
    modality: "video",
    frames: [{ timestamp_ms: 0, base64: "AP8=" }, { timestamp_ms: 500, base64: "AQI=" }],
  });
  await assert.rejects(videoFrameEvidence([
    { timestampMs: 500, image: new Blob(["a"]) },
    { timestampMs: 500, image: new Blob(["b"]) },
  ]), /increasing/);
});

test("remote forwards canonical request and returns server errors", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, "http://localhost:3217/v1/evaluate");
      assert.deepEqual(JSON.parse(options.body), { state: "example" });
      return new Response(JSON.stringify({ error: "unsupported modality" }), { status: 400 });
    };
    await assert.rejects(new TurboJevRemote("http://localhost:3217/").evaluate({ state: "example" }), /unsupported modality/);
  } finally { globalThis.fetch = original; }
});
