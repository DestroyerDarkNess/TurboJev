fn main() {
    println!("cargo:rerun-if-env-changed=TURBOJEV_LLAMA_SHIM_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_FFI").is_none() {
        return;
    }
    match std::env::var_os("TURBOJEV_LLAMA_SHIM_LIB_DIR") {
        Some(path) => println!("cargo:rustc-link-search=native={}", std::path::PathBuf::from(path).display()),
        None => println!(
            "cargo:warning=ffi enabled without TURBOJEV_LLAMA_SHIM_LIB_DIR; cargo check works, but final linking requires the built turbojev_llama_shim library"
        ),
    }
}
