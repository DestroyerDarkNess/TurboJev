/** Model-independent media evidence for local WASM or explicitly selected servers. */
export type MediaModality = "image" | "audio" | "video";
export type MediaEvidence = {
    contract: "turbojev.media.v1";
    text: string;
    modality: "image" | "audio";
    base64: string;
} | {
    contract: "turbojev.media.v1";
    text: string;
    modality: "video";
    frames: VideoFrame[];
};
export interface VideoFrame {
    timestamp_ms: number;
    base64: string;
}
export interface VideoFrameInput {
    timestampMs: number;
    image: Blob;
}
/** Encode a browser File/Blob without exposing a server-side filesystem path. */
export declare function mediaEvidence(file: Blob, modality: "image" | "audio", text?: string): Promise<MediaEvidence>;
/** Encode ordered, already-decoded still frames; video containers are not decoded by TurboJev. */
export declare function videoFrameEvidence(frames: readonly VideoFrameInput[], text?: string): Promise<MediaEvidence>;
/** Use a same-origin reverse proxy, or a server with explicitly configured CORS. */
export declare class TurboJevRemote {
    private readonly baseUrl;
    constructor(baseUrl: string);
    evaluate(request: Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>>;
}
