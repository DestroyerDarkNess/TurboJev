#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]

use pyo3::{exceptions::PyRuntimeError, prelude::*};
use turbojev_host_native::{GenerationControl, HostError, NativeSession};

// PyO3's Send bound also covers synchronous GIL release. The native session must
// stay on its creator thread. This private guard checks that before every access.
struct AffinityBorrow<'a> {
    session: &'a NativeSession,
    creator: std::thread::ThreadId,
}

// SAFETY: no native access is possible through this type on a different thread.
// The session remains borrowed throughout Python::allow_threads, which runs the
// closure synchronously on the caller thread; it never transfers the session.
#[allow(unsafe_code)]
unsafe impl Send for AffinityBorrow<'_> {}

impl AffinityBorrow<'_> {
    fn generate(self, request: &str, control: &GenerationControl) -> Result<String, HostError> {
        if self.creator != std::thread::current().id() {
            return Err(HostError::Runtime("native session changed threads".into()));
        }
        self.session.generate_json_controlled(request, control)
    }
}

#[pyclass(name = "NativeGenerationControl")]
pub struct PyGenerationControl {
    inner: GenerationControl,
}

#[pymethods]
impl PyGenerationControl {
    #[new]
    #[pyo3(signature = (timeout=None))]
    fn new(timeout: Option<f64>) -> PyResult<Self> {
        let duration = timeout
            .map(std::time::Duration::try_from_secs_f64)
            .transpose()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: GenerationControl::new(duration).map_err(py_error)?,
        })
    }

    fn cancel(&self) {
        self.inner.cancel();
    }
}

fn py_error(error: HostError) -> PyErr {
    PyRuntimeError::new_err(format!("{}: {}", error.code(), error))
}

#[pyclass(name = "NativeEngine", unsendable)]
pub struct PyNativeEngine {
    inner: Option<NativeSession>,
}

#[pymethods]
impl PyNativeEngine {
    #[staticmethod]
    fn candidate_softmax_json(input: &str) -> PyResult<String> {
        NativeSession::candidate_softmax_json(input).map_err(py_error)
    }

    #[staticmethod]
    fn compile_schema_json(schema_json: &str) -> PyResult<String> {
        NativeSession::compile_schema_json(schema_json).map_err(py_error)
    }

    #[staticmethod]
    fn load_json(options_json: &str) -> PyResult<Self> {
        NativeSession::open_json(options_json)
            .map(|inner| Self { inner: Some(inner) })
            .map_err(py_error)
    }

    fn evaluate_json(&self, request_json: &str) -> PyResult<String> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?
            .evaluate_json(request_json)
            .map_err(py_error)
    }
    fn generate_json(&self, request_json: &str) -> PyResult<String> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?
            .generate_json(request_json)
            .map_err(py_error)
    }

    fn generate_json_controlled(
        &self,
        py: Python<'_>,
        request_json: &str,
        control: &PyGenerationControl,
    ) -> PyResult<String> {
        let session = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?;
        let affinity = AffinityBorrow {
            session,
            creator: std::thread::current().id(),
        };
        py.allow_threads(move || affinity.generate(request_json, &control.inner))
            .map_err(py_error)
    }
    fn evaluate_compatible_json(&self, profile: &str, request_json: &str) -> PyResult<String> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?
            .evaluate_compatible_json(profile, request_json)
            .map_err(py_error)
    }
    fn info_json(&self) -> PyResult<String> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("engine is closed"))?
            .info_json()
            .map_err(py_error)
    }
    fn close(&mut self) {
        self.inner.take();
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNativeEngine>()?;
    m.add_class::<PyGenerationControl>()?;
    Ok(())
}
