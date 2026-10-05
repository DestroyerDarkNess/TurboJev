using System.Runtime.InteropServices;
namespace TurboJev;
internal static class Native {
 const string Lib="turbojev_capi";
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern uint tj_abi_version();
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern int tj_engine_open_json([MarshalAs(UnmanagedType.LPUTF8Str)]string o,out nint h);
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern int tj_engine_evaluate_json(nint h,[MarshalAs(UnmanagedType.LPUTF8Str)]string r,out nint o);
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern int tj_engine_generate_json(nint h,[MarshalAs(UnmanagedType.LPUTF8Str)]string r,out nint o);
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern int tj_engine_evaluate_compatible_json(nint h,[MarshalAs(UnmanagedType.LPUTF8Str)]string p,[MarshalAs(UnmanagedType.LPUTF8Str)]string r,out nint o);
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern int tj_engine_close(nint h);
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern void tj_string_free(nint p);
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern nint tj_last_error_code();
 [DllImport(Lib,CallingConvention=CallingConvention.Cdecl)] internal static extern nint tj_last_error_message();
 internal static string Read(nint p)=>Marshal.PtrToStringUTF8(p)??"";
 internal static void Check(int rc){if(rc==0)return;throw new TurboJevException(Read(tj_last_error_code()),Read(tj_last_error_message()));}
}
public sealed class TurboJevException(string code,string message):Exception(message){public string Code{get;}=code;}
