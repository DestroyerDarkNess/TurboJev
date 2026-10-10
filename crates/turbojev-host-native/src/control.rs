use crate::HostError;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

/// Host-owned control. Cancellation is thread-safe; a loaded engine remains thread-affine.
#[derive(Clone)]
pub struct GenerationControl {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl GenerationControl {
    pub fn new(timeout: Option<Duration>) -> Result<Self, HostError> {
        let deadline = timeout
            .map(|duration| {
                Instant::now()
                    .checked_add(duration)
                    .ok_or_else(|| HostError::InvalidOptions("timeout is too large".into()))
            })
            .transpose()?;
        Ok(Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline,
        })
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn check(&self) -> Result<(), HostError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(HostError::GenerationCancelled);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(HostError::GenerationTimeout);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_deadline_and_shared_cancellation() {
        let deadline = GenerationControl::new(Some(Duration::ZERO)).unwrap();
        assert!(matches!(
            deadline.check(),
            Err(HostError::GenerationTimeout)
        ));
        let control = GenerationControl::new(None).unwrap();
        let other = control.clone();
        std::thread::spawn(move || other.cancel()).join().unwrap();
        assert!(matches!(
            control.check(),
            Err(HostError::GenerationCancelled)
        ));
    }
}
