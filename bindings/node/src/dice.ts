import type { ScalarField } from "./types.js";

/** Six-face example schema. The model's probabilities are never made uniform. */
export class Dice {
  constructor(
    public readonly permutations: number | "auto" | "all" = "auto",
    public readonly returnProbabilities = true,
  ) {}

  toSchema(): ScalarField {
    return {
      type: "string",
      enum: ["one", "two", "three", "four", "five", "six"],
      instructions: "What number will come up on a single roll of a fair six-sided die?",
      permutations: this.permutations,
      return_probabilities: this.returnProbabilities,
    };
  }
}
