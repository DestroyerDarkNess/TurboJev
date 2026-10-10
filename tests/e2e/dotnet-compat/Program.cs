using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text.Json;
using TurboJev;

var model = ReadOption("--model");
var report = ReadOption("--report");
var timer = Stopwatch.StartNew();
var expectedTarget = typeof(Program).Assembly.GetCustomAttribute<TargetFrameworkAttribute>()!.FrameworkName;
var sdkTarget = typeof(TurboJevEngine).Assembly.GetCustomAttribute<TargetFrameworkAttribute>()!.FrameworkName;
var expectedSdk = expectedTarget.StartsWith(".NETFramework") ? ".NETFramework,Version=v4.8"
    : expectedTarget.StartsWith(".NETCoreApp,Version=v2") || expectedTarget.StartsWith(".NETCoreApp,Version=v3")
      || expectedTarget.StartsWith(".NETCoreApp,Version=v5") || expectedTarget.StartsWith(".NETCoreApp,Version=v6")
      || expectedTarget.StartsWith(".NETCoreApp,Version=v7") ? ".NETStandard,Version=v2.0" : ".NETCoreApp,Version=v8.0";
Require(sdkTarget == expectedSdk, "NuGet selected the wrong managed SDK asset");

var schema = new
{
    type = "object",
    properties = new
    {
        person = new { type = "object", properties = new { age = new { type = "integer", @enum = new[] { 37 } } } },
        label = new { type = "string", @enum = new[] { "niño 🐈 中文" }, permutations = 1 }
    }
};
var compiled = TurboJevEngine.CompileSchema(schema);
Require(compiled.Count == 2 && compiled[0].Name == "person.age", "Nested schema compilation failed");
var softmax = TurboJevEngine.CandidateSoftmax(new Dictionary<string, double> { ["café"] = 10000, ["茶"] = 10000 });
Require(softmax["café"] == 0.5 && softmax["茶"] == 0.5, "UTF-8 candidate normalization failed");
var dice = JsonSerializer.Serialize(new Dice(1).ToSchema());
Require(dice.Contains("six"), "Dice schema failed");

using (var engine = await TurboJevEngine.LoadAsync(new { model = "fixture://dotnet-compat" }))
{
    var request = new { context = "niño 🐈 中文", schema };
    var response = await engine.GenerateAsync(request);
    Require(response.Result["person"].GetProperty("age").GetInt32() == 37, "Typed records failed");
    Require(response.Result["label"].GetString() == "niño 🐈 中文", "Native UTF-8 output was corrupted");
    Require(response.Usage.Decisions == 2, "Usage deserialization failed");

    // Send literal UTF-8, not just the serializer's escaped ASCII representation.
    var raw = "{\"context\":\"niño 🐈 中文\",\"questions\":{\"café\":{\"type\":\"string\",\"enum\":[\"mañana 🐈\"],\"permutations\":1}}}";
    using (var output = JsonDocument.Parse(await engine.GenerateJsonAsync(raw)))
        Require(output.RootElement.GetProperty("result").GetProperty("café").GetString() == "mañana 🐈", "Native UTF-8 input was corrupted");

    try
    {
        await engine.GenerateJsonAsync(raw, TimeSpan.Zero);
        throw new Exception("Expired call returned a result");
    }
    catch (TurboJevException error) when (error.Code == "GENERATION_TIMEOUT") { }

    using (var cancellation = new CancellationTokenSource())
    {
        cancellation.Cancel();
        try
        {
            await engine.GenerateJsonAsync(raw, null, cancellation.Token);
            throw new Exception("Cancelled call returned a result");
        }
        catch (TurboJevException error) when (error.Code == "GENERATION_CANCELLED") { }
    }
}

await using (var engine = await TurboJevEngine.LoadAsync(new { model = "fixture://async-dispose" })) { }

// A WinForms/WPF-style context must not deadlock a synchronous using block.
Exception? workerError = null;
var context = new NonPumpingContext();
var worker = new Thread(() =>
{
    try
    {
        SynchronizationContext.SetSynchronizationContext(context);
        using (var engine = TurboJevEngine.LoadAsync(new { model = "fixture://ui-dispose" }).GetAwaiter().GetResult())
            engine.GenerateAsync(new { context = "facts", schema }).GetAwaiter().GetResult();
    }
    catch (Exception error) { workerError = error; }
}) { IsBackground = true };
worker.Start();
Require(worker.Join(TimeSpan.FromSeconds(10)), "SDK captured a UI context and deadlocked");
if (workerError != null) throw workerError;
Require(context.Posts == 0, "SDK posted a continuation to the caller's UI context");

long? realEvaluationMs = null;
if (model != null)
{
    using (var engine = await TurboJevEngine.LoadAsync(new { model, device = "cpu", nThreads = 4, nCtx = 4096 }))
    {
        var realTimer = Stopwatch.StartNew();
        var raw = "{\"state\":\"The animal says meow.\",\"questions\":{\"animal\":{\"type\":\"choice\",\"instructions\":\"Which animal is described?\",\"criteria\":{\"gato\":\"A cat\",\"perro\":\"A dog\"}}}}";
        using (var response = JsonDocument.Parse(await engine.EvaluateJsonAsync(raw)))
        {
            var answer = response.RootElement.GetProperty("answers").GetProperty("animal");
            Require(new[] { "gato", "perro" }.Contains(answer.GetProperty("choice").GetString()), "Real GGUF returned an invalid choice");
        }
        realEvaluationMs = realTimer.ElapsedMilliseconds;
    }
}

var record = JsonSerializer.Serialize(new
{
    target = expectedTarget,
    sdk_target = sdkTarget,
    runtime = RuntimeInformation.FrameworkDescription,
    architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    passed = true,
    contracts = new[] { "package_asset_selection", "native_loading", "schema", "softmax", "typed_records", "dice", "utf8", "timeout", "cancellation", "sync_async_disposal", "ui_context" },
    real_model = model == null ? null : Path.GetFileName(model),
    real_evaluation_ms = realEvaluationMs,
    elapsed_ms = timer.ElapsedMilliseconds
});
if (report != null) File.WriteAllText(report, record);
Console.WriteLine(record);

string? ReadOption(string name)
{
    var index = Array.IndexOf(args, name);
    return index < 0 ? null : args[index + 1];
}

void Require(bool valid, string message)
{
    if (!valid) throw new Exception(message);
}

sealed class NonPumpingContext : SynchronizationContext
{
    public int Posts;
    public override void Post(SendOrPostCallback callback, object? state) => Interlocked.Increment(ref Posts);
}
