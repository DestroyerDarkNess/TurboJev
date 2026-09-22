/** Model-independent media evidence for local WASM or explicitly selected servers. */
export type MediaModality = "image" | "audio" | "video";
export interface MediaEvidence {
    contract: "turbojev.media.v1";
    text: string;
    modality: MediaModality;
    base64: string;
}
/** Encode a browser File/Blob without exposing a server-side filesystem path. */
export declare function mediaEvidence(file: Blob, modality: MediaModality, text?: string): Promise<MediaEvidence>;
/** Use a same-origin reverse proxy, or a server with explicitly configured CORS. */
export declare class TurboJevRemote {
    private readonly baseUrl;
    constructor(baseUrl: string);
    evaluate(request: Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>>;
}
