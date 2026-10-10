using System.Text.Json;
using System.Text.Json.Serialization;

namespace TurboJev;

/// JSON elements preserve nested objects, arrays and probability answers without lossy conversions.
public sealed record Generation
{
    [JsonPropertyName("prompts")]
    public string[] Prompts { get; init; } = System.Array.Empty<string>();
    [JsonPropertyName("result")]
    public Dictionary<string, JsonElement> Result { get; init; } = new();

    [JsonPropertyName("thinking")]
    public Dictionary<string, string> Thinking { get; init; } = new();

    [JsonPropertyName("thinking_effort")]
    public Dictionary<string, string> ThinkingEffort { get; init; } = new();

    [JsonPropertyName("skipped")]
    public string[] Skipped { get; init; } = System.Array.Empty<string>();

    [JsonPropertyName("usage")]
    public GenerationUsage Usage { get; init; } = new();
}

public sealed record GenerationUsage
{
    [JsonPropertyName("decisions")]
    public int Decisions { get; init; }

    [JsonPropertyName("input_tokens")]
    public long? InputTokens { get; init; }

    [JsonPropertyName("generated_tokens")]
    public long GeneratedTokens { get; init; }
}

public sealed record CompiledDecision
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = "";

    [JsonPropertyName("path")]
    public string[] Path { get; init; } = System.Array.Empty<string>();

    [JsonPropertyName("question")]
    public string Question { get; init; } = "";

    [JsonPropertyName("choices")]
    public JsonElement Choices { get; init; }

    [JsonPropertyName("array")]
    public JsonElement Array { get; init; }

    [JsonPropertyName("output_schema")]
    public JsonElement OutputSchema { get; init; }

    [JsonExtensionData]
    public Dictionary<string, JsonElement> Options { get; init; } = new();
}
