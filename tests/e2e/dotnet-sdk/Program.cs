using System.Text.Json;
using TurboJev;

var schema = new {
    type = "object",
    properties = new {
        person = new { type = "object", properties = new { age = new { type = "integer", @enum = new[] { 37 } } } }
    }
};
var compiled = TurboJevEngine.CompileSchema(schema);
if (compiled[0].Name != "person.age") throw new Exception("Nested schema compilation failed");
var distribution = TurboJevEngine.CandidateSoftmax(new Dictionary<string, double> { ["a"] = 10000, ["b"] = 10000 });
if (distribution["a"] != 0.5) throw new Exception("Candidate normalization failed");

// Fixture checks prove binding contracts, not model intelligence.
await using var engine = await TurboJevEngine.LoadAsync(new { model = "fixture://sdk-contract" });
var request = new { context = "fixture contract", schema };
var response = await engine.GenerateAsync(request);
if (response.Result["person"].GetProperty("age").GetInt32() != 37) throw new Exception("Nested result failed");
if (response.Usage.Decisions != 1) throw new Exception("Typed usage failed");

var wire = JsonSerializer.Serialize(request);
try {
    await engine.GenerateJsonAsync(wire, TimeSpan.Zero);
    throw new Exception("Expired call returned a result");
} catch (TurboJevException error) when (error.Code == "GENERATION_TIMEOUT") { }

using var cancellation = new CancellationTokenSource();
cancellation.Cancel();
try {
    await engine.GenerateJsonAsync(wire, null, cancellation.Token);
    throw new Exception("Cancelled call returned a result");
} catch (TurboJevException error) when (error.Code == "GENERATION_CANCELLED") { }

Console.WriteLine(".NET compiler, typed nested response, normalization and control contracts passed.");
