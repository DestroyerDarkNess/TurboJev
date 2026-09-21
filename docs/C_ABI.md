# C ABI

The C ABI uses opaque `tj_engine*` handles, UTF-8 JSON, explicit `tj_string_free`, thread-local error strings and panic containment. Handles are thread-affine.
