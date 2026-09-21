# ADR 0016 — Mobile preserves thread-affine runtimes

Android uses a HandlerThread and Swift a dedicated Foundation Thread. Core code has no target-OS branches; JNI/C are outer adapters only.
