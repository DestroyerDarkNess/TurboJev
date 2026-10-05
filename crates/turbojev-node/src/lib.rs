// N-API macros locally allow the unsafe registration glue they generate.
#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]

use napi::{Error, Result, Status};
use napi_derive::napi;
use turbojev_host_native::{HostError, NativeSession};

fn map_error(error: HostError) -> Error {
    Error::new(
        Status::GenericFailure,
        format!("{}: {}", error.code(), error),
    )
}

#[napi]
pub struct NativeEngine {
    inner: Option<NativeSession>,
}

#[napi]
impl NativeEngine {
    #[napi(factory)]
    pub fn open(options_json: String) -> Result<Self> {
        NativeSession::open_json(&options_json)
            .map(|inner| Self { inner: Some(inner) })
            .map_err(map_error)
    }
    #[napi]
    pub fn evaluate_json(&self, request_json: String) -> Result<String> {
        self.inner
            .as_ref()
            .ok_or_else(|| Error::from_reason("ENGINE_CLOSED"))?
            .evaluate_json(&request_json)
            .map_err(map_error)
    }
    #[napi]
    pub fn evaluate_compatible_json(
        &self,
        profile: String,
        request_json: String,
    ) -> Result<String> {
        self.inner
            .as_ref()
            .ok_or_else(|| Error::from_reason("ENGINE_CLOSED"))?
            .evaluate_compatible_json(&profile, &request_json)
            .map_err(map_error)
    }
    #[napi]
    pub fn generate_json(&self, request_json: String) -> Result<String> {
        self.inner
            .as_ref()
            .ok_or_else(|| Error::from_reason("ENGINE_CLOSED"))?
            .generate_json(&request_json)
            .map_err(map_error)
    }
    #[napi]
    pub fn info_json(&self) -> Result<String> {
        self.inner
            .as_ref()
            .ok_or_else(|| Error::from_reason("ENGINE_CLOSED"))?
            .info_json()
            .map_err(map_error)
    }
    #[napi]
    pub fn close(&mut self) {
        self.inner.take();
    }
}
