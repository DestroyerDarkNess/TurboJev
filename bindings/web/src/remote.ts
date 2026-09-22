/** Model-independent media evidence for local WASM or explicitly selected servers. */
export type MediaModality = "image" | "audio" | "video";
export interface MediaEvidence {
  contract: "turbojev.media.v1";
  text: string;
  modality: MediaModality;
  base64: string;
}

/** Encode a browser File/Blob without exposing a server-side filesystem path. */
export async function mediaEvidence(file: Blob, modality: MediaModality, text = ""): Promise<MediaEvidence> {
  if (!file.size || file.size > 24 * 1024 * 1024) throw new Error("Media must contain 1 byte..24 MiB");
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  }
  return { contract: "turbojev.media.v1", text, modality, base64: btoa(binary) };
}

/** Use a same-origin reverse proxy, or a server with explicitly configured CORS. */
export class TurboJevRemote {
  constructor(private readonly baseUrl: string) {}
  async evaluate(request: Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const response = await fetch(`${this.baseUrl.replace(/\/$/, "")}/v1/evaluate`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request), signal,
    });
    const body = await response.json();
    if (!response.ok) throw new Error(`TurboJev ${response.status}: ${JSON.stringify(body)}`);
    return body;
  }
}
