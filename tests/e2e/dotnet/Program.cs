using System.Text.Json;
using TurboJev;

if (args.Length != 2) throw new ArgumentException("usage: TurboJev.E2E <model> <request.json>");
await using var jev = await TurboJevEngine.LoadAsync(new { model = args[0], nGpuLayers = 0 });
var output = await jev.EvaluateJsonAsync(await File.ReadAllTextAsync(args[1]));
using var doc = JsonDocument.Parse(output);
foreach (var id in new[] { "route", "urgent", "severity" }) {
    if (!doc.RootElement.GetProperty("answers").TryGetProperty(id, out _))
        throw new Exception($"missing typed answer: {id}");
}
Console.WriteLine(JsonSerializer.Serialize(new { language = "dotnet", response = doc.RootElement }));
