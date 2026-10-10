using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace TurboJev;

internal static class Native
{
    private const string Lib = "turbojev_capi";
#if NETFRAMEWORK
    // Keep the module loaded for the process lifetime, as with normal P/Invoke.
    private static readonly nint frameworkLibrary;

    static Native()
    {
        // .NET Framework does not resolve NuGet RID assets like modern .NET.
        var architecture = RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant();
        var path = Path.Combine(AppContext.BaseDirectory, "runtimes", "win-" + architecture, "native", Lib + ".dll");
        if (File.Exists(path))
        {
            // Dependencies resolve beside this DLL without changing the process PATH.
            frameworkLibrary = LoadLibraryEx(path, 0, 0x00000100 | 0x00001000);
            if (frameworkLibrary == 0)
            {
                throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "Cannot load the packaged TurboJev native runtime.");
            }
        }
    }

    [DllImport("kernel32.dll", EntryPoint = "LoadLibraryExW", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern nint LoadLibraryEx(string path, nint file, uint flags);
#endif
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    private static extern int tj_compile_schema_json([In] byte[] schema, out nint output);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint tj_abi_version();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    private static extern int tj_engine_open_json([In] byte[] o, out nint h);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    private static extern int tj_engine_evaluate_json(nint h, [In] byte[] r, out nint o);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    private static extern int tj_engine_generate_json(nint h, [In] byte[] r, out nint o);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    private static extern int tj_engine_evaluate_compatible_json(nint h, [In] byte[] p, [In] byte[] r, out nint o);
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
    private static extern int tj_engine_generate_json_controlled(nint engine, [In] byte[] request, nint control, out nint output);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    private static extern int tj_candidate_softmax_json([In] byte[] input, out nint output);

    // Explicit null-terminated UTF-8 works on Framework and .NET Standard 2.0.
    private static byte[] Utf8(string value)
    {
        var bytes = new byte[Encoding.UTF8.GetByteCount(value) + 1];
        Encoding.UTF8.GetBytes(value, 0, value.Length, bytes, 0);
        return bytes;
    }

    internal static int tj_compile_schema_json(string schema, out nint output) =>
        tj_compile_schema_json(Utf8(schema), out output);
    internal static int tj_engine_open_json(string options, out nint handle) =>
        tj_engine_open_json(Utf8(options), out handle);
    internal static int tj_engine_evaluate_json(nint handle, string request, out nint output) =>
        tj_engine_evaluate_json(handle, Utf8(request), out output);
    internal static int tj_engine_generate_json(nint handle, string request, out nint output) =>
        tj_engine_generate_json(handle, Utf8(request), out output);
    internal static int tj_engine_evaluate_compatible_json(nint handle, string profile, string request, out nint output) =>
        tj_engine_evaluate_compatible_json(handle, Utf8(profile), Utf8(request), out output);
    internal static int tj_engine_generate_json_controlled(nint handle, string request, nint control, out nint output) =>
        tj_engine_generate_json_controlled(handle, Utf8(request), control, out output);
    internal static int tj_candidate_softmax_json(string input, out nint output) =>
        tj_candidate_softmax_json(Utf8(input), out output);

    internal static string Read(nint p)
    {
        if (p == 0) return "";
        var length = 0;
        while (Marshal.ReadByte(p, length) != 0) length++;
        var bytes = new byte[length];
        Marshal.Copy(p, bytes, 0, length);
        return Encoding.UTF8.GetString(bytes);
    }

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
        : base(message.Split(new[] { "\nTURBOJEV_USAGE: " }, StringSplitOptions.None)[0])
    {
        Code = code;
        var parts = message.Split(new[] { "\nTURBOJEV_USAGE: " }, StringSplitOptions.None);
        if (parts.Length > 1)
        {
            using var document = JsonDocument.Parse(parts[1]);
            Usage = document.RootElement.Clone();
        }
    }
}
