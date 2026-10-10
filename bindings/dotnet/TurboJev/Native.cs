using System.Runtime.InteropServices;
using System.Text.Json;

namespace TurboJev;

internal static class Native
{
    private const string Lib = "turbojev_capi";
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_compile_schema_json([MarshalAs(UnmanagedType.LPUTF8Str)] string schema, out nint output);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint tj_abi_version();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_engine_open_json([MarshalAs(UnmanagedType.LPUTF8Str)] string o, out nint h);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_engine_evaluate_json(nint h, [MarshalAs(UnmanagedType.LPUTF8Str)] string r, out nint o);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_engine_generate_json(nint h, [MarshalAs(UnmanagedType.LPUTF8Str)] string r, out nint o);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_engine_evaluate_compatible_json(nint h, [MarshalAs(UnmanagedType.LPUTF8Str)] string p, [MarshalAs(UnmanagedType.LPUTF8Str)] string r, out nint o);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_engine_close(nint h);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void tj_string_free(nint p);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern nint tj_last_error_code();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern nint tj_last_error_message();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_generation_control_open(long timeoutMs, out nint control);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void tj_generation_control_cancel(nint control);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void tj_generation_control_free(nint control);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_engine_generate_json_controlled(nint engine, [MarshalAs(UnmanagedType.LPUTF8Str)] string request, nint control, out nint output);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int tj_candidate_softmax_json([MarshalAs(UnmanagedType.LPUTF8Str)] string input, out nint output);
    internal static string Read(nint p) => Marshal.PtrToStringUTF8(p) ?? "";

    internal static void Check(int status)
    {
        if (status != 0)
        {
            throw new TurboJevException(Read(tj_last_error_code()), Read(tj_last_error_message()));
        }
    }
}

public sealed class TurboJevException : Exception
{
    public string Code { get; }
    public JsonElement? Usage { get; }

    public TurboJevException(string code, string message)
        : base(message.Split("\nTURBOJEV_USAGE: ")[0])
    {
        Code = code;
        var parts = message.Split("\nTURBOJEV_USAGE: ");
        if (parts.Length > 1)
        {
            using var document = JsonDocument.Parse(parts[1]);
            Usage = document.RootElement.Clone();
        }
    }
}
