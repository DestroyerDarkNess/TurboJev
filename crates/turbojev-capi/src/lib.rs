#![deny(rust_2018_idioms)]
#![deny(unsafe_op_in_unsafe_fn)]

//! Stable TurboJev C ABI. No Rust layout is exposed across the boundary.

use std::{
    cell::RefCell,
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    thread::ThreadId,
};
use turbojev_host_native::{GenerationControl, HostError, NativeSession};

/// Thread-safe cancellation state; the engine itself remains thread-affine.
#[repr(C)]
pub struct tj_generation_control {
    inner: GenerationControl,
}

/// Create control. Negative timeout disables the deadline; zero expires immediately.
///
/// # Safety
/// `out_control` must be a writable pointer slot. Free the returned handle once.
#[no_mangle]
pub unsafe extern "C" fn tj_generation_control_open(
    timeout_ms: i64,
    out_control: *mut *mut tj_generation_control,
) -> i32 {
    if out_control.is_null() {
        return set_error("INVALID_ARGUMENT", "out_control must not be null");
    }
    let timeout = (timeout_ms >= 0).then(|| std::time::Duration::from_millis(timeout_ms as u64));
    match GenerationControl::new(timeout) {
        Ok(inner) => {
            // SAFETY: the caller provides writable output storage.
            unsafe {
                *out_control = Box::into_raw(Box::new(tj_generation_control { inner }));
            }
            clear_error();
            TJ_OK
        }
        Err(error) => host_error(error),
    }
}

/// Cancel from any thread. Control must remain alive until generation returns.
///
/// # Safety
/// `control` must be a live handle; no concurrent call may free it.
#[no_mangle]
pub unsafe extern "C" fn tj_generation_control_cancel(control: *const tj_generation_control) {
    // SAFETY: live handle required by the caller contract.
    if let Some(control) = unsafe { control.as_ref() } {
        control.inner.cancel();
    }
}

/// Release control after every generation/cancel operation has completed.
///
/// # Safety
/// Free a handle returned by open exactly once, with no outstanding uses.
#[no_mangle]
pub unsafe extern "C" fn tj_generation_control_free(control: *mut tj_generation_control) {
    if !control.is_null() {
        // SAFETY: unique owned handle required by the caller contract.
        drop(unsafe { Box::from_raw(control) });
    }
}

/// Generate with cooperative checks before/after each complete model operation.
///
/// # Safety
/// Engine affinity and input/output ownership match `tj_engine_generate_json`.
/// Control must remain live until this function returns. Cancellation may run concurrently.
#[no_mangle]
pub unsafe extern "C" fn tj_engine_generate_json_controlled(
    engine: *mut tj_engine,
    request_json: *const c_char,
    control: *const tj_generation_control,
    out_json: *mut *mut c_char,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: checked borrowed pointers under the caller's live-handle contract.
        let handle = match unsafe { handle_ref(engine) } {
            Ok(value) => value,
            Err(code) => return code,
        };
        // SAFETY: UTF-8 C string required for the duration of the call.
        let request = match unsafe { input(request_json, "request_json") } {
            Ok(value) => value,
            Err(code) => return code,
        };
        // SAFETY: control may be read concurrently but cannot be freed during this call.
        let Some(control) = (unsafe { control.as_ref() }) else {
            return set_error("INVALID_ARGUMENT", "control must not be null");
        };
        match handle
            .session
            .generate_json_controlled(request, &control.inner)
        {
            Ok(value) => write_output(value, out_json),
            Err(error) => host_error(error),
        }
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary"),
    }
}

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
        "INVALID_ARGUMENT" | "INVALID_OPTIONS" | "JSON_ERROR" | "WIRE_ERROR" => {
            TJ_E_INVALID_ARGUMENT
        }
        "THREAD_AFFINITY" => TJ_E_THREAD_AFFINITY,
        "PANIC" => TJ_E_PANIC,
        _ => TJ_E_RUNTIME,
    }
}

fn clear_error() {
    let _ = set_error("OK", "OK");
}

unsafe fn input<'a>(value: *const c_char, name: &str) -> Result<&'a str, i32> {
    if value.is_null() {
        return Err(set_error(
            "INVALID_ARGUMENT",
            format!("{name} must not be null"),
        ));
    }
    // SAFETY: caller contract requires a valid NUL-terminated UTF-8 string for the duration of the call.
    let bytes = unsafe { CStr::from_ptr(value) };
    bytes
        .to_str()
        .map_err(|_| set_error("INVALID_ARGUMENT", format!("{name} must be UTF-8")))
}

fn host_error(error: HostError) -> i32 {
    set_error(error.code(), error.to_string())
}

unsafe fn handle_ref<'a>(engine: *mut tj_engine) -> Result<&'a tj_engine, i32> {
    // SAFETY: checked non-null and the opaque pointer is created only by tj_engine_open_json.
    let handle = unsafe { engine.as_ref() }
        .ok_or_else(|| set_error("INVALID_ARGUMENT", "engine must not be null"))?;
    if handle.creator != std::thread::current().id() {
        return Err(set_error(
            "THREAD_AFFINITY",
            "TurboJev native handles must be used from the thread that created them",
        ));
    }
    Ok(handle)
}

fn write_output(value: String, out: *mut *mut c_char) -> i32 {
    if out.is_null() {
        return set_error("INVALID_ARGUMENT", "output pointer must not be null");
    }
    let string = match CString::new(value) {
        Ok(v) => v,
        Err(_) => return set_error("RUNTIME_ERROR", "output unexpectedly contained NUL"),
    };
    // SAFETY: out was checked non-null and ownership is transferred to caller until tj_string_free.
    unsafe {
        *out = string.into_raw();
    }
    clear_error();
    TJ_OK
}

/// Open an engine from a JSON configuration.
///
/// # Safety
/// `options_json` must point to a readable NUL-terminated UTF-8 string and
/// `out_engine` to an aligned, writable pointer slot. Keep that slot disjoint
/// from the input. Close a successfully returned handle once on this thread.
#[no_mangle]
pub unsafe extern "C" fn tj_engine_open_json(
    options_json: *const c_char,
    out_engine: *mut *mut tj_engine,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        if out_engine.is_null() {
            return set_error("INVALID_ARGUMENT", "out_engine must not be null");
        }
        // SAFETY: validated by input().
        let options = match unsafe { input(options_json, "options_json") } {
            Ok(v) => v,
            Err(c) => return c,
        };
        let session = match NativeSession::open_json(options) {
            Ok(v) => v,
            Err(e) => return host_error(e),
        };
        let boxed = Box::new(tj_engine {
            session,
            creator: std::thread::current().id(),
        });
        // SAFETY: caller supplied valid output slot.
        unsafe {
            *out_engine = Box::into_raw(boxed);
        }
        clear_error();
        TJ_OK
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary"),
    }
}

/// Evaluate a typed decision request.
///
/// # Safety
/// `engine` must be a live handle returned by `tj_engine_open_json`, used only
/// on its creating thread without concurrent access. `request_json` must be a
/// readable NUL-terminated UTF-8 string; `out_json` must be an aligned, writable
/// pointer slot disjoint from inputs. Free returned strings with `tj_string_free`.
#[no_mangle]
pub unsafe extern "C" fn tj_engine_evaluate_json(
    engine: *mut tj_engine,
    request_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: opaque pointer and C strings validated below.
        let handle = match unsafe { handle_ref(engine) } {
            Ok(v) => v,
            Err(c) => return c,
        };
        let request = match unsafe { input(request_json, "request_json") } {
            Ok(v) => v,
            Err(c) => return c,
        };
        match handle.session.evaluate_json(request) {
            Ok(v) => write_output(v, out_json),
            Err(e) => host_error(e),
        }
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary"),
    }
}

/// Evaluate through a named compatibility profile.
///
/// # Safety
/// The engine and output slot must satisfy `tj_engine_evaluate_json`'s safety
/// contract. Both `profile` and `request_json` must point to readable
/// NUL-terminated UTF-8 strings throughout this call.
#[no_mangle]
pub unsafe extern "C" fn tj_engine_evaluate_compatible_json(
    engine: *mut tj_engine,
    profile: *const c_char,
    request_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let handle = match unsafe { handle_ref(engine) } {
            Ok(v) => v,
            Err(c) => return c,
        };
        let profile = match unsafe { input(profile, "profile") } {
            Ok(v) => v,
            Err(c) => return c,
        };
        let request = match unsafe { input(request_json, "request_json") } {
            Ok(v) => v,
            Err(c) => return c,
        };
        match handle.session.evaluate_compatible_json(profile, request) {
            Ok(v) => write_output(v, out_json),
            Err(e) => host_error(e),
        }
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary"),
    }
}

/// Generate typed JSON Schema fields through the same thread-affine session.
/// Generate dependency-aware typed fields.
///
/// # Safety
/// The handle, request string and output slot must satisfy
/// `tj_engine_evaluate_json`'s safety contract, including thread affinity,
/// exclusive access and string ownership.
#[no_mangle]
pub unsafe extern "C" fn tj_engine_generate_json(
    engine: *mut tj_engine,
    request_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: validate the owned handle and borrowed UTF-8 C string.
        let handle = match unsafe { handle_ref(engine) } {
            Ok(v) => v,
            Err(c) => return c,
        };
        let request = match unsafe { input(request_json, "request_json") } {
            Ok(v) => v,
            Err(c) => return c,
        };
        match handle.session.generate_json(request) {
            Ok(v) => write_output(v, out_json),
            Err(e) => host_error(e),
        }
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary"),
    }
}

/// Normalize named candidate logits without loading a model.
///
/// # Safety
/// Input must be valid NUL-terminated UTF-8. Output must be a writable slot.
/// Free the returned string with `tj_string_free`.
#[no_mangle]
pub unsafe extern "C" fn tj_candidate_softmax_json(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: borrowed string required by the caller contract.
        let input = match unsafe { input(input_json, "input_json") } {
            Ok(value) => value,
            Err(code) => return code,
        };
        match NativeSession::candidate_softmax_json(input) {
            Ok(value) => write_output(value, out_json),
            Err(error) => host_error(error),
        }
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary"),
    }
}

/// Compile a schema without creating an engine or invoking a model.
///
/// # Safety
/// Input must be valid NUL-terminated UTF-8. Output must be a writable pointer
/// slot; release its returned owned string with `tj_string_free`.
#[no_mangle]
pub unsafe extern "C" fn tj_compile_schema_json(
    schema_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let schema = match unsafe { input(schema_json, "schema_json") } {
            Ok(value) => value,
            Err(code) => return code,
        };
        match NativeSession::compile_schema_json(schema) {
            Ok(value) => write_output(value, out_json),
            Err(error) => host_error(error),
        }
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at schema compiler boundary"),
    }
}

/// Return engine information as an owned JSON string.
///
/// # Safety
/// `engine` must be a live handle accessed exclusively on its creating thread.
/// `out_json` must point to an aligned, writable pointer slot disjoint from the
/// engine. Free a returned string exactly once with `tj_string_free`.
#[no_mangle]
pub unsafe extern "C" fn tj_engine_info_json(
    engine: *mut tj_engine,
    out_json: *mut *mut c_char,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let handle = match unsafe { handle_ref(engine) } {
            Ok(v) => v,
            Err(c) => return c,
        };
        match handle.session.info_json() {
            Ok(v) => write_output(v, out_json),
            Err(e) => host_error(e),
        }
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary"),
    }
}

/// Close an engine and consume its handle.
///
/// # Safety
/// `engine` must be null or a live handle returned by `tj_engine_open_json`.
/// Close it exclusively on its creating thread, exactly once; never use the
/// handle after a successful close.
#[no_mangle]
pub unsafe extern "C" fn tj_engine_close(engine: *mut tj_engine) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        if engine.is_null() {
            clear_error();
            return TJ_OK;
        }
        // Check affinity before reclaiming ownership.
        if let Err(code) = unsafe { handle_ref(engine) } {
            return code;
        }
        // SAFETY: pointer was allocated by Box::into_raw in open and is consumed exactly once here.
        unsafe {
            drop(Box::from_raw(engine));
        }
        clear_error();
        TJ_OK
    })) {
        Ok(code) => code,
        Err(_) => set_error("PANIC", "panic contained at TurboJev C ABI boundary"),
    }
}

/// Release an owned string returned by this library.
///
/// # Safety
/// `value` must be null or an unmodified string pointer returned by TurboJev.
/// Free it exactly once, without concurrent readers; never use it afterward.
#[no_mangle]
pub unsafe extern "C" fn tj_string_free(value: *mut c_char) {
    if !value.is_null() {
        // SAFETY: only strings returned by this ABI may be passed here, once.
        unsafe {
            drop(CString::from_raw(value));
        }
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
pub extern "C" fn tj_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_codes_are_nonzero_for_errors() {
        assert_ne!(set_error("INVALID_ARGUMENT", "bad"), TJ_OK);
        assert_eq!(tj_abi_version(), 1);
    }
}
