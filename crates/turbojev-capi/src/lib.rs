#![deny(rust_2018_idioms)]
#![deny(unsafe_op_in_unsafe_fn)]

//! Stable TurboJev C ABI. No Rust layout is exposed across the boundary.

use std::{
    cell::RefCell,
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    thread::ThreadId,
};
use turbojev_host_native::{HostError, NativeSession};

pub const TJ_OK: i32 = 0;
pub const TJ_E_INVALID_ARGUMENT: i32 = 1;
pub const TJ_E_RUNTIME: i32 = 2;
pub const TJ_E_THREAD_AFFINITY: i32 = 3;
pub const TJ_E_PANIC: i32 = 255;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("OK").expect("literal has no NUL"));
    static LAST_CODE: RefCell<CString> = RefCell::new(CString::new("OK").expect("literal has no NUL"));
}

/// Opaque engine handle. C callers only ever hold a pointer to this type.
#[repr(C)]
pub struct tj_engine {
    session: NativeSession,
    creator: ThreadId,
}

fn set_error(code: &str, message: impl AsRef<str>) -> i32 {
    let sanitized = message.as_ref().replace('\0', "\\0");
    LAST_CODE.with(|v| *v.borrow_mut() = CString::new(code).expect("sanitized code"));
    LAST_ERROR.with(|v| *v.borrow_mut() = CString::new(sanitized).expect("sanitized message"));
    match code {
        "INVALID_ARGUMENT" | "INVALID_OPTIONS" | "JSON_ERROR" | "WIRE_ERROR" => TJ_E_INVALID_ARGUMENT,
        "THREAD_AFFINITY" => TJ_E_THREAD_AFFINITY,
        "PANIC" => TJ_E_PANIC,
        _ => TJ_E_RUNTIME,
    }
}

fn clear_error() { let _ = set_error("OK", "OK"); }

unsafe fn input<'a>(value: *const c_char, name: &str) -> Result<&'a str, i32> {
    if value.is_null() { return Err(set_error("INVALID_ARGUMENT", format!("{name} must not be null"))); }
    // SAFETY: caller contract requires a valid NUL-terminated UTF-8 string for the duration of the call.
    let bytes = unsafe { CStr::from_ptr(value) };
    bytes.to_str().map_err(|_| set_error("INVALID_ARGUMENT", format!("{name} must be UTF-8")))
}

fn host_error(error: HostError) -> i32 { set_error(error.code(), error.to_string()) }

unsafe fn handle_ref<'a>(engine: *mut tj_engine) -> Result<&'a tj_engine, i32> {
    // SAFETY: checked non-null and the opaque pointer is created only by tj_engine_open_json.
    let handle = unsafe { engine.as_ref() }.ok_or_else(|| set_error("INVALID_ARGUMENT", "engine must not be null"))?;
    if handle.creator != std::thread::current().id() {
        return Err(set_error("THREAD_AFFINITY", "TurboJev native handles must be used from the thread that created them"));
    }
    Ok(handle)
}

fn write_output(value: String, out: *mut *mut c_char) -> i32 {
    if out.is_null() { return set_error("INVALID_ARGUMENT", "output pointer must not be null"); }
    let string = match CString::new(value) { Ok(v) => v, Err(_) => return set_error("RUNTIME_ERROR", "output unexpectedly contained NUL") };
    // SAFETY: out was checked non-null and ownership is transferred to caller until tj_string_free.
    unsafe { *out = string.into_raw(); }
    clear_error();
    TJ_OK
}

#[no_mangle]
pub unsafe extern "C" fn tj_engine_open_json(options_json: *const c_char, out_engine: *mut *mut tj_engine) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        if out_engine.is_null() { return set_error("INVALID_ARGUMENT", "out_engine must not be null"); }
        // SAFETY: validated by input().
        let options = match unsafe { input(options_json, "options_json") } { Ok(v) => v, Err(c) => return c };
        let session = match NativeSession::open_json(options) { Ok(v) => v, Err(e) => return host_error(e) };
        let boxed = Box::new(tj_engine { session, creator: std::thread::current().id() });
        // SAFETY: caller supplied valid output slot.
        unsafe { *out_engine = Box::into_raw(boxed); }
        clear_error(); TJ_OK
    })) { Ok(code) => code, Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary") }
}

#[no_mangle]
pub unsafe extern "C" fn tj_engine_evaluate_json(engine: *mut tj_engine, request_json: *const c_char, out_json: *mut *mut c_char) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: opaque pointer and C strings validated below.
        let handle = match unsafe { handle_ref(engine) } { Ok(v) => v, Err(c) => return c };
        let request = match unsafe { input(request_json, "request_json") } { Ok(v) => v, Err(c) => return c };
        match handle.session.evaluate_json(request) { Ok(v) => write_output(v, out_json), Err(e) => host_error(e) }
    })) { Ok(code) => code, Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary") }
}

#[no_mangle]
pub unsafe extern "C" fn tj_engine_evaluate_compatible_json(engine: *mut tj_engine, profile: *const c_char, request_json: *const c_char, out_json: *mut *mut c_char) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let handle = match unsafe { handle_ref(engine) } { Ok(v) => v, Err(c) => return c };
        let profile = match unsafe { input(profile, "profile") } { Ok(v) => v, Err(c) => return c };
        let request = match unsafe { input(request_json, "request_json") } { Ok(v) => v, Err(c) => return c };
        match handle.session.evaluate_compatible_json(profile, request) { Ok(v) => write_output(v, out_json), Err(e) => host_error(e) }
    })) { Ok(code) => code, Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary") }
}

#[no_mangle]
pub unsafe extern "C" fn tj_engine_classify_json(engine: *mut tj_engine, state: *const c_char, options_json: *const c_char, out_json: *mut *mut c_char) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let handle = match unsafe { handle_ref(engine) } { Ok(v) => v, Err(c) => return c };
        let state = match unsafe { input(state, "state") } { Ok(v) => v, Err(c) => return c };
        let options = match unsafe { input(options_json, "options_json") } { Ok(v) => v, Err(c) => return c };
        match handle.session.classify_json(state, options) { Ok(v) => write_output(v, out_json), Err(e) => host_error(e) }
    })) { Ok(code) => code, Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary") }
}

#[no_mangle]
pub unsafe extern "C" fn tj_engine_info_json(engine: *mut tj_engine, out_json: *mut *mut c_char) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let handle = match unsafe { handle_ref(engine) } { Ok(v) => v, Err(c) => return c };
        match handle.session.info_json() { Ok(v) => write_output(v, out_json), Err(e) => host_error(e) }
    })) { Ok(code) => code, Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary") }
}

#[no_mangle]
pub unsafe extern "C" fn tj_engine_close(engine: *mut tj_engine) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        if engine.is_null() { clear_error(); return TJ_OK; }
        // Check affinity before reclaiming ownership.
        if let Err(code) = unsafe { handle_ref(engine) } { return code; }
        // SAFETY: pointer was allocated by Box::into_raw in open and is consumed exactly once here.
        unsafe { drop(Box::from_raw(engine)); }
        clear_error(); TJ_OK
    })) { Ok(code) => code, Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary") }
}

#[no_mangle]
pub unsafe extern "C" fn tj_string_free(value: *mut c_char) {
    if !value.is_null() {
        // SAFETY: only strings returned by this ABI may be passed here, once.
        unsafe { drop(CString::from_raw(value)); }
    }
}

#[no_mangle]
pub extern "C" fn tj_last_error_code() -> *const c_char {
    LAST_CODE.with(|v| v.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn tj_last_error_message() -> *const c_char {
    LAST_ERROR.with(|v| v.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn tj_abi_version() -> u32 { 1 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_codes_are_nonzero_for_errors() {
        assert_ne!(set_error("INVALID_ARGUMENT", "bad"), TJ_OK);
        assert_eq!(tj_abi_version(), 1);
    }
}
