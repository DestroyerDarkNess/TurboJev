#if NETSTANDARD2_0 || NETFRAMEWORK
namespace System.Runtime.CompilerServices;

// Enables the existing immutable SDK records on older framework references.
internal static class IsExternalInit { }
#endif
