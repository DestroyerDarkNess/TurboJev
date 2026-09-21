# ADR 0015 — Shared native host layer

Node, Python, C/.NET and mobile native bindings delegate model/session behavior to `turbojev-host-native`; bindings only own host-language lifetime, I/O and ABI concerns.
