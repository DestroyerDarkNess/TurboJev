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
