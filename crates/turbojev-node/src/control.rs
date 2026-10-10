use napi::{Error, Result};
use napi_derive::napi;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex, OnceLock,
};
use turbojev_host_native::GenerationControl;

static NEXT: AtomicU64 = AtomicU64::new(1);
static CONTROLS: OnceLock<Mutex<HashMap<String, GenerationControl>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<String, GenerationControl>> {
    CONTROLS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn lookup(id: &str) -> Result<GenerationControl> {
    registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(id)
        .cloned()
        .ok_or_else(|| {
            Error::from_reason(
                "INVALID_OPTIONS: generation control is closed or belongs to another addon",
            )
        })
}

/// Controls can cross worker boundaries by ID. Loaded engines never cross threads.
#[napi]
pub struct NativeGenerationControl {
    id: String,
    inner: GenerationControl,
}

#[napi]
impl NativeGenerationControl {
    #[napi(constructor)]
    pub fn new(timeout_ms: Option<f64>) -> Result<Self> {
        let duration = timeout_ms
            .map(|value| std::time::Duration::try_from_secs_f64(value / 1000.0))
            .transpose()
            .map_err(|error| Error::from_reason(format!("INVALID_OPTIONS: {error}")))?;
        let inner = GenerationControl::new(duration).map_err(super::map_error)?;
        let id = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::from_reason("generation control identifiers exhausted"))?
            .to_string();
        registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(id.clone(), inner.clone());
        Ok(Self { id, inner })
    }

    #[napi(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    #[napi]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[napi]
    pub fn close(&self) {
        registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

impl Drop for NativeGenerationControl {
    fn drop(&mut self) {
        self.close();
    }
}
