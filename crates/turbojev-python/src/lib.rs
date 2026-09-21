#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use pyo3::{exceptions::PyRuntimeError, prelude::*};
use turbojev_host_native::{HostError, NativeSession};

fn py_error(error: HostError) -> PyErr {
    PyRuntimeError::new_err(format!("{}: {}", error.code(), error))
}

#[pyclass(name = "NativeEngine", unsendable)]
pub struct PyNativeEngine { inner: Option<NativeSession> }

#[pymethods]
impl PyNativeEngine {
    #[staticmethod]
    fn load_json(options_json: &str) -> PyResult<Self> {
        NativeSession::open_json(options_json).map(|inner| Self { inner: Some(inner) }).map_err(py_error)
    }

    fn evaluate_json(&self, request_json: &str) -> PyResult<String> {
        self.inner.as_ref().ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?.evaluate_json(request_json).map_err(py_error)
    }
    fn evaluate_compatible_json(&self, profile: &str, request_json: &str) -> PyResult<String> {
        self.inner.as_ref().ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?.evaluate_compatible_json(profile, request_json).map_err(py_error)
    }
    fn classify_json(&self, state: &str, options_json: &str) -> PyResult<String> {
        self.inner.as_ref().ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?.classify_json(state, options_json).map_err(py_error)
    }
    fn info_json(&self) -> PyResult<String> {
        self.inner.as_ref().ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?.info_json().map_err(py_error)
    }
    fn close(&mut self) { self.inner.take(); }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNativeEngine>()?;
    Ok(())
}
