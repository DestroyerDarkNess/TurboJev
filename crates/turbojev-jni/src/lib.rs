#![deny(rust_2018_idioms)]
#![deny(unsafe_op_in_unsafe_fn)]
use jni::{
    objects::{JClass, JString},
    sys::{jlong, jstring},
    JNIEnv,
};
use std::thread::ThreadId;
use turbojev_host_native::NativeSession;
struct Handle {
    session: NativeSession,
    creator: ThreadId,
}
fn throw(env: &mut JNIEnv<'_>, message: impl AsRef<str>) {
    let _ = env.throw_new("ai/turbojev/TurboJevException", message.as_ref());
}
fn read(env: &mut JNIEnv<'_>, s: JString<'_>) -> Option<String> {
    match env.get_string(&s) {
        Ok(v) => Some(v.into()),
        Err(e) => {
            throw(env, e.to_string());
            None
        }
    }
}
unsafe fn h<'a>(env: &mut JNIEnv<'_>, raw: jlong) -> Option<&'a mut Handle> {
    if raw == 0 {
        throw(env, "engine handle is closed");
        return None;
    }
    let p = raw as *mut Handle;
    let v = unsafe { p.as_mut() };
    match v {
        Some(v) if v.creator == std::thread::current().id() => Some(v),
        Some(_) => {
            throw(env, "TurboJev handle used from a different thread");
            None
        }
        None => {
            throw(env, "invalid engine handle");
            None
        }
    }
}
fn out(env: &mut JNIEnv<'_>, value: String) -> jstring {
    match env.new_string(value) {
        Ok(v) => v.into_raw(),
        Err(e) => {
            throw(env, e.to_string());
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub extern "system" fn Java_ai_turbojev_NativeBridge_open(
    mut env: JNIEnv<'_>,
    _: JClass<'_>,
    opts: JString<'_>,
) -> jlong {
    let Some(s) = read(&mut env, opts) else {
        return 0;
    };
    match NativeSession::open_json(&s) {
        Ok(session) => Box::into_raw(Box::new(Handle {
            session,
            creator: std::thread::current().id(),
        })) as jlong,
        Err(e) => {
            throw(&mut env, format!("{}: {}", e.code(), e));
            0
        }
    }
}
#[no_mangle]
pub extern "system" fn Java_ai_turbojev_NativeBridge_evaluate(
    mut env: JNIEnv<'_>,
    _: JClass<'_>,
    raw: jlong,
    req: JString<'_>,
) -> jstring {
    let Some(req) = read(&mut env, req) else {
        return std::ptr::null_mut();
    };
    let Some(x) = (unsafe { h(&mut env, raw) }) else {
        return std::ptr::null_mut();
    };
    match x.session.evaluate_json(&req) {
        Ok(v) => out(&mut env, v),
        Err(e) => {
            throw(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub extern "system" fn Java_ai_turbojev_NativeBridge_compatible(
    mut env: JNIEnv<'_>,
    _: JClass<'_>,
    raw: jlong,
    p: JString<'_>,
    req: JString<'_>,
) -> jstring {
    let Some(p) = read(&mut env, p) else {
        return std::ptr::null_mut();
    };
    let Some(req) = read(&mut env, req) else {
        return std::ptr::null_mut();
    };
    let Some(x) = (unsafe { h(&mut env, raw) }) else {
        return std::ptr::null_mut();
    };
    match x.session.evaluate_compatible_json(&p, &req) {
        Ok(v) => out(&mut env, v),
        Err(e) => {
            throw(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub extern "system" fn Java_ai_turbojev_NativeBridge_close(
    mut env: JNIEnv<'_>,
    _: JClass<'_>,
    raw: jlong,
) {
    if raw == 0 {
        return;
    }
    let p = raw as *mut Handle;
    let creator = unsafe { p.as_ref() }.map(|v| v.creator);
    if creator != Some(std::thread::current().id()) {
        throw(
            &mut env,
            "TurboJev handle must be closed on its creator thread",
        );
        return;
    }
    unsafe {
        drop(Box::from_raw(p));
    }
}
