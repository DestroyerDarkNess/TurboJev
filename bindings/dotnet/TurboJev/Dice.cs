namespace TurboJev;

/// <summary>A six-face example schema. The model distribution is never forced to be uniform.</summary>
public sealed class Dice(object? permutations = null, bool returnProbabilities = true)
{
    public object ToSchema() => new
    {
        type = "string",
        @enum = new[] { "one", "two", "three", "four", "five", "six" },
        instructions = "What number will come up on a single roll of a fair six-sided die?",
        permutations = permutations ?? "auto",
        return_probabilities = returnProbabilities,
    };
}
