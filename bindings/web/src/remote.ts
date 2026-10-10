/** Model-independent media evidence for local WASM or explicitly selected servers. */
export type MediaModality = "image" | "audio" | "video";
export type MediaEvidence =
  | { contract: "turbojev.media.v1"; text: string; modality: "image" | "audio"; base64: string }
  | { contract: "turbojev.media.v1"; text: string; modality: "image"; images: string[] }
  | { contract: "turbojev.media.v1"; text: string; modality: "video"; frames: VideoFrame[] };

export interface VideoFrame {
  timestamp_ms: number;
  base64: string;
}

export interface VideoFrameInput {
  timestampMs: number;
  image: Blob;
}

const MAX_MEDIA_BYTES = 24 * 1024 * 1024;

async function blobBase64(file: Blob): Promise<string> {
  if (!file.size || file.size > MAX_MEDIA_BYTES)
    throw new Error("Media must contain 1 byte..24 MiB");
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  }
  return btoa(binary);
}

/** Encode a browser File/Blob without exposing a server-side filesystem path. */
export async function mediaEvidence(
  file: Blob,
  modality: "image" | "audio",
  text = "",
): Promise<MediaEvidence> {
  return { contract: "turbojev.media.v1", text, modality, base64: await blobBase64(file) };
}

/** Encode several still images in caller order, without an implicit network request. */
export async function imageEvidence(images: readonly Blob[], text = ""): Promise<MediaEvidence> {
  if (!images.length || images.length > 32) throw new Error("Supply 1..32 images");
  if (images.reduce((total, image) => total + image.size, 0) > MAX_MEDIA_BYTES) {
    throw new Error("Combined images must not exceed 24 MiB");
  }
  return {
    contract: "turbojev.media.v1",
    text,
    modality: "image",
    images: await Promise.all(images.map(blobBase64)),
  };
}

/** Encode ordered, already-decoded still frames; video containers are not decoded by TurboJev. */
export async function videoFrameEvidence(
  frames: readonly VideoFrameInput[],
  text = "",
): Promise<MediaEvidence> {
  if (!frames.length || frames.length > 32)
    throw new Error("Video must contain 1..32 timestamped frames");
  let previous = -1;
  let totalBytes = 0;
  const encoded: VideoFrame[] = [];
  for (const frame of frames) {
    if (
      !Number.isSafeInteger(frame.timestampMs) ||
      frame.timestampMs < 0 ||
      frame.timestampMs <= previous
    ) {
      throw new Error(
        "Video frame timestamps must be increasing non-negative integers in milliseconds",
      );
    }
    previous = frame.timestampMs;
    if (!frame.image.size || frame.image.size > MAX_MEDIA_BYTES)
      throw new Error("Video frame must contain 1 byte..24 MiB");
    totalBytes += frame.image.size;
    if (totalBytes > MAX_MEDIA_BYTES)
      throw new Error("Combined video frames must not exceed 24 MiB");
    encoded.push({ timestamp_ms: frame.timestampMs, base64: await blobBase64(frame.image) });
  }
  return { contract: "turbojev.media.v1", text, modality: "video", frames: encoded };
}

/** Use a same-origin reverse proxy, or a server with explicitly configured CORS. */
export class TurboJevRemote {
  constructor(private readonly baseUrl: string) {}
  async evaluate(
    request: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<Record<string, unknown>> {
    const response = await fetch(`${this.baseUrl.replace(/\/$/, "")}/v1/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
    const body = await response.json();
    if (!response.ok) throw new Error(`TurboJev ${response.status}: ${JSON.stringify(body)}`);
    return body;
  }
}
