using System.Text.Json;
using TurboJev;

if (args.Length is < 2 or > 3) throw new ArgumentException("usage: TurboJev.E2E <model> <request.json> [mmproj.gguf]");
await using var jev = await TurboJevEngine.LoadAsync(new { model = args[0], mmproj = args.Length == 3 ? args[2] : null, nGpuLayers = 0 });
var output = await jev.EvaluateJsonAsync(await File.ReadAllTextAsync(args[1]));
using var doc = JsonDocument.Parse(output);
if (!doc.RootElement.TryGetProperty("answers", out var answers) || answers.ValueKind != JsonValueKind.Object || !answers.EnumerateObject().Any())
    throw new Exception("missing typed answers");
Console.WriteLine(JsonSerializer.Serialize(new { language = "dotnet", response = doc.RootElement }));
