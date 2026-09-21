# ADR 0012 — Node host worker

Node model I/O/cache belongs to `@turbojev/node`; native inference is isolated to one worker thread and delegates to `turbojev-host-native`.
