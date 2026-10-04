export interface TransformersRuntimeOptions {
    model: string;
    device?: string;
    dtype?: string;
    thinking?: boolean;
    progressCallback?: (p: unknown) => void;
}
export declare function createRuntime(): Promise<{
    initialize(input: TransformersRuntimeOptions): Promise<{
        name: string;
        model: string;
        codecProfile: string;
        capabilities: {
            nextTokenLogits: boolean;
            sequenceScores: boolean;
            sharedPrefix: boolean;
            parallelSuffixes: boolean;
            selectiveOutputHead: boolean;
            localInference: boolean;
        };
    }>;
    encodePrompt(messagesJson: string): string;
    encodePromptWithAnswer(messagesJson: string, answer: string): string;
    decodeTokens(tokenIdsJson: string): any;
    scoreRuntimeBatch(batchJson: string, _planJson: string): Promise<string>;
    close(): Promise<void>;
}>;
