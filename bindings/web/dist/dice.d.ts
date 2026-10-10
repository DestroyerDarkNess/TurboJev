import type { ScalarField } from "./fields.js";
/** Six-face example schema. The model's probabilities are never made uniform. */
export declare class Dice {
    readonly permutations: number | "auto" | "all";
    readonly returnProbabilities: boolean;
    constructor(permutations?: number | "auto" | "all", returnProbabilities?: boolean);
    toSchema(): ScalarField;
}
