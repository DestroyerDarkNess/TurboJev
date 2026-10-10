using System.Collections.Concurrent;
using System.Text.Json;

namespace TurboJev;

public sealed class TurboJevEngine : IAsyncDisposable, IDisposable
{
    readonly BlockingCollection<Action> queue = new();
    readonly Thread thread;
    nint handle;
    bool closed;

    TurboJevEngine()
    {
        thread = new Thread(() =>
        {
            foreach (var job in queue.GetConsumingEnumerable()) job();
        })
        { IsBackground = true, Name = "TurboJevNative" };
        thread.Start();
    }

    static Task<T> Run<T>(BlockingCollection<Action> queue, Func<T> fn)
    {
        var completion = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        queue.Add(() =>
        {
            try { completion.SetResult(fn()); }
            catch (Exception error) { completion.SetException(error); }
        });
        return completion.Task;
    }

    public static async Task<TurboJevEngine> LoadAsync(object options)
    {
        var json = JsonSerializer.Serialize(options);
        var engine = new TurboJevEngine();
        try
        {
            engine.handle = await Run(engine.queue, () =>
            {
                Native.Check(Native.tj_engine_open_json(json, out var handle));
                return handle;
            }).ConfigureAwait(false);
            return engine;
        }
        catch
        {
            // Opening can fail before a native handle exists (bad model, ABI mismatch, etc.).
            // Always stop the dedicated affinity thread before propagating the error.
            engine.closed = true;
            engine.queue.CompleteAdding();
            engine.thread.Join();
            engine.queue.Dispose();
            throw;
        }
    }

    public Task<string> EvaluateJsonAsync(string request) =>
        Run(queue, () => Call((out nint output) => Native.tj_engine_evaluate_json(handle, request, out output)));
    public static string CompileSchemaJson(string schema) =>
        Call((out nint output) => Native.tj_compile_schema_json(schema, out output));

    public Task<string> GenerateJsonAsync(string request) =>
        Run(queue, () => Call((out nint output) => Native.tj_engine_generate_json(handle, request, out output)));

    /// Cancellation is checked between native operations; it does not interrupt an active decode.
    public async Task<string> GenerateJsonAsync(string request, TimeSpan? timeout, CancellationToken cancellationToken = default)
    {
        if (timeout is { } duration && (duration < TimeSpan.Zero || duration.TotalMilliseconds > long.MaxValue))
        {
            throw new ArgumentOutOfRangeException(nameof(timeout));
        }
        Native.Check(Native.tj_generation_control_open(timeout is { } limit ? (long)limit.TotalMilliseconds : -1, out var control));
        using var registration = cancellationToken.Register(() => Native.tj_generation_control_cancel(control));
        try
        {
            return await Run(queue, () => Call((out nint output) => Native.tj_engine_generate_json_controlled(handle, request, control, out output))).ConfigureAwait(false);
        }
        finally
        {
            // Dispose the registration first so a concurrent callback cannot use a freed control.
            registration.Dispose();
            Native.tj_generation_control_free(control);
        }
    }

    public async Task<Generation> GenerateAsync(object request, TimeSpan? timeout = null, CancellationToken cancellationToken = default)
    {
        var response = await GenerateJsonAsync(JsonSerializer.Serialize(request), timeout, cancellationToken).ConfigureAwait(false);
        return JsonSerializer.Deserialize<Generation>(response)!;
    }

    public static IReadOnlyList<CompiledDecision> CompileSchema(object schema) =>
        JsonSerializer.Deserialize<CompiledDecision[]>(CompileSchemaJson(JsonSerializer.Serialize(schema)))!;

    public static IReadOnlyDictionary<string, double> CandidateSoftmax(IReadOnlyDictionary<string, double> logits, double temperature = 1)
    {
        var input = JsonSerializer.Serialize(new { logits, temperature });
        var output = Call((out nint pointer) => Native.tj_candidate_softmax_json(input, out pointer));
        return JsonSerializer.Deserialize<Dictionary<string, double>>(output)!;
    }

    public Task<string> EvaluateCompatibleJsonAsync(string profile, string request) =>
        Run(queue, () => Call((out nint output) => Native.tj_engine_evaluate_compatible_json(handle, profile, request, out output)));

    delegate int NativeCall(out nint pointer);

    static string Call(NativeCall fn)
    {
        Native.Check(fn(out var pointer));
        try { return Native.Read(pointer); }
        finally { Native.tj_string_free(pointer); }
    }

    public async ValueTask DisposeAsync()
    {
        if (closed) return;
        try
        {
            await Run(queue, () =>
            {
                Native.Check(Native.tj_engine_close(handle));
                handle = 0;
                return 0;
            }).ConfigureAwait(false);
        }
        finally
        {
            // The managed affinity worker must never survive a failed native close.
            closed = true;
            queue.CompleteAdding();
            thread.Join();
            queue.Dispose();
        }
    }

    /// <summary>Closes the native engine and its worker; usable with traditional using statements.</summary>
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();
}
