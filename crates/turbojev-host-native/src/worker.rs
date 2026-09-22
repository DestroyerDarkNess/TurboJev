//! Opt-in local JSON-lines scorer. No shell is used; Rust owns response semantics.
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use turbojev_backend::{
    BackendCapabilities, BackendError, BackendOutput, DecisionBatch, ExecutionPlan, ScoringBackend,
};

/// A trusted executable configured by the application, never by an evaluation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerOptions {
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}
fn default_timeout() -> u64 {
    300
}

struct Process {
    child: Child,
    requests: mpsc::SyncSender<Vec<u8>>,
    written: mpsc::Receiver<Result<(), String>>,
    lines: mpsc::Receiver<Result<String, String>>,
}
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Process {
    fn receive(&mut self, timeout: Duration) -> Result<String, BackendError> {
        match self.lines.recv_timeout(timeout) {
            Ok(Ok(line)) => Ok(line),
            error => {
                let _ = self.child.kill();
                Err(BackendError::Inference(format!(
                    "worker disconnected or timed out: {error:?}"
                )))
            }
        }
    }
}

pub struct WorkerBackend {
    process: Mutex<Process>,
    timeout: Duration,
    modalities: Vec<String>,
}
impl WorkerBackend {
    pub fn open(options: &WorkerOptions) -> Result<Self, BackendError> {
        if options.program.trim().is_empty() || options.timeout_seconds == 0 {
            return Err(BackendError::InvalidInput(
                "worker program and positive timeout required".into(),
            ));
        }
        let mut child = Command::new(&options.program)
            .args(&options.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| BackendError::NotReady(e.to_string()))?;
        let mut input = child.stdin.take().expect("piped stdin");
        let output = child.stdout.take().expect("piped stdout");
        let (requests, pending) = mpsc::sync_channel::<Vec<u8>>(1);
        let (writes, written) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            while let Ok(payload) = pending.recv() {
                let result = input
                    .write_all(&payload)
                    .and_then(|()| input.write_all(b"\n"))
                    .and_then(|()| input.flush())
                    .map_err(|e| e.to_string());
                let failed = result.is_err();
                if writes.send(result).is_err() || failed {
                    break;
                }
            }
        });
        let (tx, lines) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(output);
            loop {
                let mut line = String::new();
                let result = reader.by_ref().take(1024 * 1024 + 1).read_line(&mut line);
                match result {
                    Ok(0) => break,
                    Ok(_) if line.len() <= 1024 * 1024 && line.ends_with('\n') => {
                        if tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    _ => {
                        let _ = tx.send(Err("invalid or oversized worker response".into()));
                        break;
                    }
                }
            }
        });
        let timeout = Duration::from_secs(options.timeout_seconds);
        let mut process = Process {
            child,
            requests,
            written,
            lines,
        };
        let hello: serde_json::Value = serde_json::from_str(&process.receive(timeout)?)
            .map_err(|e| BackendError::NotReady(e.to_string()))?;
        if hello["contract"] != "turbojev.scorer.v1" {
            return Err(BackendError::NotReady(
                "worker handshake contract mismatch".into(),
            ));
        }
        let modalities: Vec<String> = serde_json::from_value(hello["modalities"].clone())
            .map_err(|e| BackendError::NotReady(e.to_string()))?;
        Ok(Self {
            process: Mutex::new(process),
            timeout,
            modalities,
        })
    }
    pub fn modalities(&self) -> &[String] {
        &self.modalities
    }
}
impl ScoringBackend for WorkerBackend {
    fn name(&self) -> &'static str {
        "local-worker"
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            local_inference: true,
            ..Default::default()
        }
    }
    fn score(
        &self,
        batch: &DecisionBatch,
        plan: ExecutionPlan,
    ) -> Result<BackendOutput, BackendError> {
        if !self.capabilities().supports(plan) {
            return Err(BackendError::UnsupportedPlan);
        }
        let media = turbojev_schema::media::MediaEvidence::from_state(&batch.state)
            .map_err(BackendError::InvalidInput)?;
        let modality = media
            .as_ref()
            .map(|m| match m.modality {
                turbojev_schema::media::Modality::Image => "image",
                turbojev_schema::media::Modality::Audio => "audio",
                turbojev_schema::media::Modality::Video => "video",
            })
            .unwrap_or("text");
        if !self.modalities.iter().any(|m| m == modality) {
            return Err(BackendError::Unsupported(format!(
                "worker does not support {modality}"
            )));
        }
        let payload =
            serde_json::to_vec(batch).map_err(|e| BackendError::InvalidInput(e.to_string()))?;
        let mut process = self
            .process
            .lock()
            .map_err(|e| BackendError::Inference(e.to_string()))?;
        process
            .requests
            .try_send(payload)
            .map_err(|e| BackendError::Inference(e.to_string()))?;
        match process.written.recv_timeout(self.timeout) {
            Ok(Ok(())) => (),
            error => {
                let _ = process.child.kill();
                return Err(BackendError::Inference(format!(
                    "worker input failed or timed out: {error:?}"
                )));
            }
        }
        let line = process.receive(self.timeout)?;
        let value: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| BackendError::Inference(e.to_string()))?;
        if let Some(error) = value.get("error") {
            return Err(BackendError::Inference(error.to_string()));
        }
        serde_json::from_value(value).map_err(|e| BackendError::Inference(e.to_string()))
    }
}
