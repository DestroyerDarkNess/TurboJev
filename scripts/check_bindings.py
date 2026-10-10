#!/usr/bin/env python3
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
required=[
'crates/turbojev-host-native/src/lib.rs','crates/turbojev-node/src/lib.rs','crates/turbojev-wasm/src/lib.rs','crates/turbojev-capi/src/lib.rs','crates/turbojev-python/src/lib.rs','crates/turbojev-jni/src/lib.rs',
'bindings/node/package.json','bindings/web/package.json','bindings/web-runtimes/transformersjs/package.json','bindings/web-runtimes/llamacpp-wasm/package.json','bindings/c/include/turbojev.h','bindings/python/turbojev/__init__.py','bindings/dotnet/TurboJev/TurboJevEngine.cs','bindings/android/src/main/kotlin/ai/turbojev/TurboJev.kt','bindings/swift/Sources/TurboJev/TurboJev.swift']
missing=[x for x in required if not (ROOT/x).is_file()]
if missing: raise SystemExit('BINDING CHECK FAILED: missing '+', '.join(missing))
android=(ROOT/'bindings/android/src/main/kotlin/ai/turbojev/TurboJev.kt').read_text(); swift=(ROOT/'bindings/swift/Sources/TurboJev/TurboJev.swift').read_text();
if 'HandlerThread' not in android: raise SystemExit('Android binding must preserve native thread affinity')
if 'NativeThread' not in swift or 'DispatchQueue' in swift: raise SystemExit('Swift binding must use a dedicated Thread, not merely a serial DispatchQueue')
if 'OpaquePointer' not in swift or 'UnsafeMutablePointer<tj_engine>' in swift: raise SystemExit('Swift binding must use Clang-imported opaque C handles')
dotnet=(ROOT/'bindings/dotnet/TurboJev/TurboJevEngine.cs').read_text()
if 'BlockingCollection<Action>' not in dotnet or 'Thread' not in dotnet: raise SystemExit('.NET binding must preserve native thread affinity')
if not re.search(r'catch\s*\{', dotnet) or 'CompleteAdding()' not in dotnet or 'thread.Join()' not in dotnet: raise SystemExit('.NET LoadAsync failure path must stop the affinity thread')
print(f'OK: {len(required)} native/browser/mobile binding artifacts present; mobile/.NET affinity + opaque-handle guards pass')
