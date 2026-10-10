/** Six-face example schema. The model's probabilities are never made uniform. */
export class Dice {
    permutations;
    returnProbabilities;
    constructor(permutations = "auto", returnProbabilities = true) {
        this.permutations = permutations;
        this.returnProbabilities = returnProbabilities;
    }
    toSchema() {
        return {
            type: "string",
            enum: ["one", "two", "three", "four", "five", "six"],
            instructions: "What number will come up on a single roll of a fair six-sided die?",
            permutations: this.permutations,
            return_probabilities: this.returnProbabilities,
        };
    }
}
