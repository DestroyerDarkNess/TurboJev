# Mobile bindings

TurboJev mobile bindings preserve the same architecture as desktop hosts:

```text
Android app -> HandlerThread -> JNI -> turbojev-host-native -> runtime adapter
Swift app   -> dedicated Thread -> C ABI -> turbojev-host-native -> runtime adapter
```

Neither wrapper implements Choice/Noul/Score, candidate discovery, calibration or model logic.
The app decides which native runtime build to ship. Android JNI and Swift/C are host-only layers;
portable core crates remain free of `jni`, Foundation, Android, iOS, target-OS branching and device APIs.

## Thread affinity

Native model contexts are allowed to be thread-affine. Android therefore owns a `HandlerThread`, and
the Swift wrapper owns a long-lived Foundation `Thread`. A serial dispatch queue is intentionally not
used as an affinity guarantee because GCD may execute serial work on different worker threads.

## Wrapper requirements

The Android source module targets API 24+ (Android 7.0), the minimum for its
`CompletableFuture` API. It builds Java 8 bytecode with Kotlin 2.0.21 and uses
Android's built-in `org.json`. Repositories are configured in
`settings.gradle.kts`; declaring them again in the module conflicts with
`FAIL_ON_PROJECT_REPOS`. Supply `libturbojev_jni.so` and its runtime dependencies
under the appropriate `jniLibs/<ABI>` directories.

The Swift source package requires Swift tools 5.7+ and declares iOS 12+ / macOS
10.13+ deployment targets. These are wrapper requirements; the supplied C ABI,
llama.cpp build and GPU backend may require a higher OS version. Supply matching
native libraries for the intended Apple device or simulator. Windows DLLs cannot
be used in an Apple application. These lower deployment targets have not been
executed on an Apple host.

Native error strings are thread-local. Swift checks open and close failures on
the same dedicated thread as the C ABI call and stops that worker when opening
or closing fails. The wrapper does not redirect native errors to another thread.

Neither mobile binding currently ships a published AAR or XCFramework with a
bundled model runtime. See [binding compatibility](BINDING_COMPATIBILITY.md)
for measured coverage and remaining platform validation.
