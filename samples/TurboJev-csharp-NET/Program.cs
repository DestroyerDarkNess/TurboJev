using System.Text.Json;
using TurboJev;

if (args.Length == 0) throw new ArgumentException("Pass a GGUF model path.");
if (Array.IndexOf(args, "--generate") >= 0) {
    var requestPath = Path.Combine(AppContext.BaseDirectory, "typed-fields-request.json");
    await using var fieldsEngine = await TurboJevEngine.LoadAsync(new {model=args[0],device="cpu",nCtx=8192,nSeqMax=1,nThreads=4});
    Console.WriteLine(await fieldsEngine.GenerateJsonAsync(await File.ReadAllTextAsync(requestPath)));
    return;
}

var mediaIndex = Array.IndexOf(args, "--media");
if (mediaIndex >= 0) {
    if (mediaIndex + 1 >= args.Length) throw new ArgumentException("Pass a file after --media.");

    var mediaPath = args[mediaIndex + 1];
    var modality = ValueAfter(args, "--modality") ?? "image";
    var question = ValueAfter(args, "--question") ?? "What is shown?";
    var options = ValuesAfter(args, "--options");
    if (options.Length == 0) options = ["A cat", "A dog"];
    if (modality is not ("image" or "video"))
        throw new ArgumentException("The SmolVLM2 sample accepts --modality image or video.");
    if (options.Length is < 2 or > 26 || options.Distinct().Count() != options.Length)
        throw new ArgumentException("Provide 2 to 26 distinct options after --options.");

    var file = new FileInfo(mediaPath);
    if (!file.Exists || file.Length == 0 || file.Length > 24L * 1024 * 1024)
        throw new ArgumentException("Pass an existing nonempty media file of at most 24 MiB.");

    var workerPython = Environment.GetEnvironmentVariable("TURBOJEV_MULTIMODAL_PYTHON");
    var workerPath = Environment.GetEnvironmentVariable("TURBOJEV_MULTIMODAL_WORKER");
    var workerModel = Environment.GetEnvironmentVariable("TURBOJEV_MULTIMODAL_MODEL");
    if (workerPython is null || workerPath is null || workerModel is null)
        throw new InvalidOperationException("First run: python samples/multimodal/setup.py vision");

    await using var engine = await TurboJevEngine.LoadAsync(new {
        model = workerModel,
        worker = new { program = workerPython, args = new[] { workerPath, "vision", workerModel }, timeoutSeconds = 600 },
    });
    var criteria = options.Select((option, index) => new KeyValuePair<string, string>($"option_{index + 1}", option))
        .ToDictionary(pair => pair.Key, pair => pair.Value);
    var request = new {
        state = new {
            contract = "turbojev.media.v1",
            modality,
            base64 = Convert.ToBase64String(await File.ReadAllBytesAsync(mediaPath)),
        },
        questions = new Dictionary<string, object> {
            ["scene"] = new { type = "choice", instructions = question, criteria },
        },
    };
    Console.WriteLine(await engine.EvaluateJsonAsync(JsonSerializer.Serialize(request)));
} else {
    // CPU is the default. Set device = "gpu" to request GPU offload explicitly.
    await using var engine = await TurboJevEngine.LoadAsync(new { model = args[0], device = "cpu", nGpuLayers = 0, nThreads = 4 });
    var request = new {
        state = new { message = args.Length > 1 ? args[1] : "My package arrived broken" },
        questions = new {
            department = new {
                type = "choice",
                instructions = "Which department should handle this message?",
                criteria = new { billing = "Payments and refunds", shipping = "Delivery", technical = "Product issue", other = "Other" },
            },
        },
    };
    var json = await engine.EvaluateJsonAsync(JsonSerializer.Serialize(request));
    using var result = JsonDocument.Parse(json);
    var answer = result.RootElement.GetProperty("answers").GetProperty("department");
    Console.WriteLine($"Department: {answer.GetProperty("choice").GetString()}");
    foreach (var item in answer.GetProperty("probabilities").EnumerateObject())
        Console.WriteLine($"  {item.Name,-12} {item.Value.GetDouble(),6:P1}");
}

static string? ValueAfter(string[] arguments, string flag) {
    var index = Array.IndexOf(arguments, flag);
    return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
}

static string[] ValuesAfter(string[] arguments, string flag) {
    var index = Array.IndexOf(arguments, flag);
    return index >= 0 ? arguments[(index + 1)..] : [];
}
