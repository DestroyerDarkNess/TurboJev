using System.Collections.Concurrent;
using System.Text.Json;

namespace TurboJev;

public sealed class TurboJevEngine : IAsyncDisposable {
    readonly BlockingCollection<Action> queue = new();
    readonly Thread thread;
    nint handle;
    bool closed;

    TurboJevEngine() {
        thread = new Thread(() => {
            foreach (var job in queue.GetConsumingEnumerable()) job();
        }) { IsBackground = true, Name = "TurboJevNative" };
        thread.Start();
    }

    static Task<T> Run<T>(BlockingCollection<Action> queue, Func<T> fn) {
        var completion = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        queue.Add(() => {
            try { completion.SetResult(fn()); }
            catch (Exception error) { completion.SetException(error); }
        });
        return completion.Task;
    }

    public static async Task<TurboJevEngine> LoadAsync(object options) {
        var engine = new TurboJevEngine();
        var json = JsonSerializer.Serialize(options);
        try {
            engine.handle = await Run(engine.queue, () => {
                Native.Check(Native.tj_engine_open_json(json, out var handle));
                return handle;
            });
            return engine;
        } catch {
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
    public Task<string> GenerateJsonAsync(string request) =>
        Run(queue, () => Call((out nint output) => Native.tj_engine_generate_json(handle, request, out output)));

    public Task<string> EvaluateCompatibleJsonAsync(string profile, string request) =>
        Run(queue, () => Call((out nint output) => Native.tj_engine_evaluate_compatible_json(handle, profile, request, out output)));

    delegate int NativeCall(out nint pointer);

    static string Call(NativeCall fn) {
        Native.Check(fn(out var pointer));
        try { return Native.Read(pointer); }
        finally { Native.tj_string_free(pointer); }
    }

    public async ValueTask DisposeAsync() {
        if (closed) return;
        try {
            await Run(queue, () => {
                Native.Check(Native.tj_engine_close(handle));
                handle = 0;
                return 0;
            });
        } finally {
            // The managed affinity worker must never survive a failed native close.
            closed = true;
            queue.CompleteAdding();
            thread.Join();
            queue.Dispose();
        }
    }
}
