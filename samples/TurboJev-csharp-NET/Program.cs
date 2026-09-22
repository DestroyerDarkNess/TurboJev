using System.Text.Json;
using TurboJev;

if (args.Length == 0) throw new ArgumentException("Pass a GGUF model path.");
await using var engine = await TurboJevEngine.LoadAsync(new { model = args[0], nGpuLayers = 0, nThreads = 4 });
var json = await engine.ClassifyJsonAsync(args.Length > 1 ? args[1] : "My package arrived broken",
    ["billing", "shipping", "technical", "other"]);
using var result = JsonDocument.Parse(json);
Console.WriteLine($"Department: {result.RootElement.GetProperty("choice").GetString()}");
foreach (var item in result.RootElement.GetProperty("probabilities").EnumerateObject())
    Console.WriteLine($"  {item.Name,-12} {item.Value.GetDouble(),6:P1}");
