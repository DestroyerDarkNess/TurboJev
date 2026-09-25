#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Local HTTP host for TurboJev.
//!
//! Networking and thread ownership live here, outside every portable crate.
//! One dedicated worker owns the thread-affine `NativeSession`; concurrent
//! clients communicate with it through a bounded queue. v0.24 adds a concrete
//! fused conformance path, resource-aware fused planning, and end-to-end batch
//! parity gates while preserving truthful production-runtime capabilities.

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use turbojev_host_native::{HostError, NativeLoadOptions, NativeSession};
use turbojev_schema::{EvaluationRequest, EvaluationResponse};
use turbojev_server_execution::{
    execute_turn, plan_cross_request_with_budget, BatchPlanningPolicy, CrossRequestBudget,
    CrossRequestCapabilities, CrossRequestExecutor, CrossRequestMode, ExecutionControl, TurnError,
    CROSS_REQUEST_EXECUTION_CONTRACT,
};
use turbojev_server_protocol::{
    decode_command, encode_batch_response_with_execution, encode_error, encode_evaluation_response,
    ProtocolError, ServerCommand, ServerLimits, SERVER_ERROR_CONTRACT,
};

const DEFAULT_BIND: &str = "127.0.0.1:3217";
const DEFAULT_MAX_CONNECTIONS: usize = 64;
const DEFAULT_MAX_HEADER_BYTES: usize = 16 * 1024;
const DEFAULT_QUEUE_CAPACITY: usize = 128;
const DEFAULT_IO_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_SESSION_TIMEOUT_MS: u64 = 120_000;
const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_FAIRNESS_QUANTUM: usize = 1;
const DEFAULT_MAX_FUSED_ITEMS: usize = 8;
const ACCEPT_POLL_MS: u64 = 10;
const REQUEST_TIMEOUT_HEADER: &str = "x-turbojev-timeout-ms";
const TURBOJEV_REQUEST_ID_HEADER: &str = "x-turbojev-request-id";
const GENERIC_REQUEST_ID_HEADER: &str = "x-request-id";
const FORWARDED_PROTO_HEADER: &str = "x-forwarded-proto";
static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);

fn default_bind() -> String {
    DEFAULT_BIND.to_owned()
}
fn default_max_connections() -> usize {
    DEFAULT_MAX_CONNECTIONS
}
fn default_max_header_bytes() -> usize {
    DEFAULT_MAX_HEADER_BYTES
}
fn default_queue_capacity() -> usize {
    DEFAULT_QUEUE_CAPACITY
}
fn default_io_timeout_ms() -> u64 {
    DEFAULT_IO_TIMEOUT_MS
}
fn default_session_timeout_ms() -> u64 {
    DEFAULT_SESSION_TIMEOUT_MS
}
fn default_drain_timeout_ms() -> u64 {
    DEFAULT_DRAIN_TIMEOUT_MS
}
fn default_fairness_quantum() -> usize {
    DEFAULT_FAIRNESS_QUANTUM
}
fn default_max_fused_items() -> usize {
    DEFAULT_MAX_FUSED_ITEMS
}

/// Host-only transport controls. They never enter portable inference crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TransportConfig {
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_max_header_bytes")]
    pub max_header_bytes: usize,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
    #[serde(default = "default_io_timeout_ms")]
    pub io_timeout_ms: u64,
    /// Upper bound for one request's total queue + execution wait.
    #[serde(default = "default_session_timeout_ms")]
    pub session_timeout_ms: u64,
    /// Grace period for already-admitted work after shutdown begins.
    #[serde(default = "default_drain_timeout_ms")]
    pub drain_timeout_ms: u64,
    /// Maximum request items one job may execute before another queued job gets a turn.
    #[serde(default = "default_fairness_quantum")]
    pub fairness_quantum: usize,
    /// Host cap for an executor that truthfully advertises fused batching.
    #[serde(default = "default_max_fused_items")]
    pub max_fused_items: usize,
    /// Optional host memory budget for one fused cross-request turn.
    #[serde(default)]
    pub fused_memory_budget_bytes: Option<u64>,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            max_connections: DEFAULT_MAX_CONNECTIONS,
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            io_timeout_ms: DEFAULT_IO_TIMEOUT_MS,
            session_timeout_ms: DEFAULT_SESSION_TIMEOUT_MS,
            drain_timeout_ms: DEFAULT_DRAIN_TIMEOUT_MS,
            fairness_quantum: DEFAULT_FAIRNESS_QUANTUM,
            max_fused_items: DEFAULT_MAX_FUSED_ITEMS,
            fused_memory_budget_bytes: None,
        }
    }
}

/// Deployment posture for the local server. Both modes keep TurboJev bound to
/// loopback; `reverse-proxy` adds validation for the trusted local proxy hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum DeploymentMode {
    #[default]
    Loopback,
    ReverseProxy,
}

impl DeploymentMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Loopback => "loopback",
            Self::ReverseProxy => "reverse-proxy",
        }
    }
}

/// Reverse-proxy deployment contract. TurboJev still does not terminate TLS or
/// implement public authentication; those remain proxy responsibilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeploymentConfig {
    #[serde(default)]
    pub mode: DeploymentMode,
    #[serde(default = "default_true")]
    pub require_forwarded_https: bool,
}

fn default_true() -> bool {
    true
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            mode: DeploymentMode::Loopback,
            require_forwarded_https: true,
        }
    }
}

/// Complete local server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_bind")]
    pub bind: String,
    pub load: NativeLoadOptions,
    #[serde(default)]
    pub limits: ServerLimits,
    #[serde(default)]
    pub transport: TransportConfig,
    #[serde(default)]
    pub deployment: DeploymentConfig,
}

impl ServerConfig {
    pub fn validate(&self) -> Result<SocketAddr, ServerError> {
        let address: SocketAddr = self.bind.parse().map_err(|_| {
            ServerError::InvalidConfig("bind must be an explicit IP:port socket address".to_owned())
        })?;
        if !address.ip().is_loopback() {
            return Err(ServerError::InvalidConfig(
                "TurboJev server only permits loopback bind addresses; expose remote access through a colocated reverse proxy".to_owned(),
            ));
        }
        self.limits.validate().map_err(ServerError::Protocol)?;
        if self.transport.max_connections == 0
            || self.transport.max_header_bytes == 0
            || self.transport.queue_capacity == 0
            || self.transport.io_timeout_ms == 0
            || self.transport.session_timeout_ms == 0
            || self.transport.drain_timeout_ms == 0
            || self.transport.fairness_quantum == 0
            || self.transport.max_fused_items == 0
            || matches!(self.transport.fused_memory_budget_bytes, Some(0))
        {
            return Err(ServerError::InvalidConfig(
                "all transport limits/timeouts must be greater than zero".to_owned(),
            ));
        }
        Ok(address)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum LifecyclePhase {
    Ready = 0,
    Draining = 1,
    Stopped = 2,
}

impl LifecyclePhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Stopped => "stopped",
        }
    }
}

#[derive(Clone, Default)]
struct ServerLifecycle(Arc<AtomicU8>);

impl ServerLifecycle {
    fn phase(&self) -> LifecyclePhase {
        match self.0.load(Ordering::Acquire) {
            0 => LifecyclePhase::Ready,
            1 => LifecyclePhase::Draining,
            _ => LifecyclePhase::Stopped,
        }
    }

    fn is_ready(&self) -> bool {
        self.phase() == LifecyclePhase::Ready
    }

    fn begin_drain(&self) {
        let _ = self.0.compare_exchange(
            LifecyclePhase::Ready as u8,
            LifecyclePhase::Draining as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn mark_stopped(&self) {
        self.0
            .store(LifecyclePhase::Stopped as u8, Ordering::Release);
    }
}

/// Programmatic graceful-shutdown signal for embedding the HTTP host.
#[derive(Clone, Default)]
pub struct ShutdownToken(Arc<AtomicBool>);

impl ShutdownToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn request(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_requested(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Default)]
struct MetricsInner {
    submitted_jobs: AtomicUsize,
    submitted_requests: AtomicUsize,
    completed_jobs: AtomicUsize,
    completed_requests: AtomicUsize,
    rejected_jobs: AtomicUsize,
    caller_deadlines: AtomicUsize,
    cancelled_before_execution: AtomicUsize,
    serial_turns: AtomicUsize,
    fused_turns: AtomicUsize,
    inflight_jobs: AtomicUsize,
    inflight_high_water: AtomicUsize,
}

#[derive(Clone, Default)]
struct ServerMetrics(Arc<MetricsInner>);

impl ServerMetrics {
    fn record_submit(&self, requests: usize) {
        self.0.submitted_jobs.fetch_add(1, Ordering::Relaxed);
        self.0
            .submitted_requests
            .fetch_add(requests, Ordering::Relaxed);
        let current = self.0.inflight_jobs.fetch_add(1, Ordering::AcqRel) + 1;
        update_high_water(&self.0.inflight_high_water, current);
    }

    fn record_rejected(&self) {
        self.0.rejected_jobs.fetch_add(1, Ordering::Relaxed);
        self.0.inflight_jobs.fetch_sub(1, Ordering::AcqRel);
    }

    fn record_complete(&self, requests: usize) {
        self.0.completed_jobs.fetch_add(1, Ordering::Relaxed);
        self.0
            .completed_requests
            .fetch_add(requests, Ordering::Relaxed);
        self.0.inflight_jobs.fetch_sub(1, Ordering::AcqRel);
    }

    fn record_failed(&self) {
        self.0.inflight_jobs.fetch_sub(1, Ordering::AcqRel);
    }

    fn snapshot(&self, capabilities: CrossRequestCapabilities, phase: LifecyclePhase) -> Value {
        json!({
            "contract": "turbojev.server-metrics.v1",
            "lifecycle": phase.as_str(),
            "executionContract": CROSS_REQUEST_EXECUTION_CONTRACT,
            "capabilities": capabilities,
            "submittedJobs": self.0.submitted_jobs.load(Ordering::Relaxed),
            "submittedRequests": self.0.submitted_requests.load(Ordering::Relaxed),
            "completedJobs": self.0.completed_jobs.load(Ordering::Relaxed),
            "completedRequests": self.0.completed_requests.load(Ordering::Relaxed),
            "rejectedJobs": self.0.rejected_jobs.load(Ordering::Relaxed),
            "callerDeadlines": self.0.caller_deadlines.load(Ordering::Relaxed),
            "cancelledBeforeExecution": self.0.cancelled_before_execution.load(Ordering::Relaxed),
            "serialTurns": self.0.serial_turns.load(Ordering::Relaxed),
            "fusedTurns": self.0.fused_turns.load(Ordering::Relaxed),
            "inflightJobs": self.0.inflight_jobs.load(Ordering::Relaxed),
            "inflightHighWater": self.0.inflight_high_water.load(Ordering::Relaxed)
        })
    }
}

fn update_high_water(target: &AtomicUsize, candidate: usize) {
    let mut current = target.load(Ordering::Acquire);
    while candidate > current {
        match target.compare_exchange_weak(current, candidate, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

#[derive(Debug)]
struct WorkerFailure {
    code: String,
    message: String,
}

#[derive(Debug)]
struct BatchExecutionResult {
    responses: Vec<EvaluationResponse>,
    mode: CrossRequestMode,
}

#[derive(Clone)]
struct JobControl {
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
}

impl JobControl {
    fn new(timeout: Duration) -> Result<Self, ServerError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| ServerError::InvalidConfig("request timeout is too large".to_owned()))?;
        Ok(Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline,
        })
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl ExecutionControl for JobControl {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn deadline_expired(&self) -> bool {
        Instant::now() >= self.deadline
    }
}

struct NativeSessionExecutor {
    session: NativeSession,
}

impl CrossRequestExecutor for NativeSessionExecutor {
    type Error = HostError;

    fn capabilities(&self) -> CrossRequestCapabilities {
        let native = self.session.execution_capabilities();
        CrossRequestCapabilities {
            fused_batch: native.fused_batch,
            cooperative_cancellation: native.cooperative_cancellation,
            hard_deadlines: native.hard_deadlines,
            max_fused_items: native.max_fused_items,
            estimated_fused_item_bytes: native.estimated_fused_item_bytes,
        }
    }

    fn execute_one(
        &mut self,
        request: &EvaluationRequest,
        _control: &dyn ExecutionControl,
    ) -> Result<EvaluationResponse, Self::Error> {
        self.session.evaluate(request.clone())
    }

    fn can_execute_fused(
        &mut self,
        requests: &[EvaluationRequest],
        _control: &dyn ExecutionControl,
    ) -> Result<bool, Self::Error> {
        self.session.can_evaluate_fused(requests)
    }

    fn execute_fused(
        &mut self,
        requests: &[EvaluationRequest],
        _control: &dyn ExecutionControl,
    ) -> Result<Vec<EvaluationResponse>, Self::Error> {
        self.session.evaluate_fused(requests)
    }
}

struct WorkerJob {
    requests: Vec<EvaluationRequest>,
    next: usize,
    responses: Vec<EvaluationResponse>,
    mode: CrossRequestMode,
    reply: mpsc::Sender<Result<BatchExecutionResult, WorkerFailure>>,
    control: JobControl,
}

impl WorkerJob {
    fn remaining(&self) -> usize {
        self.requests.len().saturating_sub(self.next)
    }
}

enum WorkerCommand {
    Submit(WorkerJob),
}

#[derive(Clone)]
struct SessionBroker {
    sender: mpsc::SyncSender<WorkerCommand>,
    default_timeout: Duration,
    info: Arc<Value>,
    capabilities: CrossRequestCapabilities,
    metrics: ServerMetrics,
}

impl SessionBroker {
    fn start(options: NativeLoadOptions, transport: &TransportConfig) -> Result<Self, ServerError> {
        let (sender, receiver) = mpsc::sync_channel(transport.queue_capacity);
        let (init_sender, init_receiver) = mpsc::channel();
        let timeout = Duration::from_millis(transport.session_timeout_ms);
        let fairness_quantum = transport.fairness_quantum;
        let max_fused_items = transport.max_fused_items;
        let fused_memory_budget_bytes = transport.fused_memory_budget_bytes;
        let pending_capacity = transport.queue_capacity;
        let metrics = ServerMetrics::default();
        let metrics_for_worker = metrics.clone();

        thread::Builder::new()
            .name("turbojev-session".to_owned())
            .spawn(move || {
                let session = match NativeSession::open(options) {
                    Ok(session) => session,
                    Err(error) => {
                        let _ = init_sender.send(Err(WorkerFailure {
                            code: error.code().to_owned(),
                            message: error.to_string(),
                        }));
                        return;
                    }
                };
                let info = match session
                    .info_json()
                    .and_then(|raw| serde_json::from_str(&raw).map_err(Into::into))
                {
                    Ok(info) => info,
                    Err(error) => {
                        let _ = init_sender.send(Err(WorkerFailure {
                            code: error.code().to_owned(),
                            message: error.to_string(),
                        }));
                        return;
                    }
                };
                let mut executor = NativeSessionExecutor { session };
                let capabilities = match executor.capabilities().validate() {
                    Ok(capabilities) => capabilities,
                    Err(error) => {
                        let _ = init_sender.send(Err(WorkerFailure {
                            code: "INVALID_EXECUTION_CAPABILITIES".to_owned(),
                            message: error.to_string(),
                        }));
                        return;
                    }
                };
                if init_sender.send(Ok((info, capabilities))).is_err() {
                    return;
                }

                run_scheduler(
                    &mut executor,
                    receiver,
                    fairness_quantum,
                    max_fused_items,
                    fused_memory_budget_bytes,
                    pending_capacity,
                    metrics_for_worker,
                );
            })
            .map_err(ServerError::Spawn)?;

        let (info, capabilities) = init_receiver
            .recv_timeout(timeout)
            .map_err(|_| ServerError::SessionTimeout)?
            .map_err(|error| ServerError::Session {
                code: error.code,
                message: error.message,
            })?;
        Ok(Self {
            sender,
            default_timeout: timeout,
            info: Arc::new(info),
            capabilities,
            metrics,
        })
    }

    fn evaluate(
        &self,
        requests: Vec<EvaluationRequest>,
        requested_timeout: Option<Duration>,
    ) -> Result<BatchExecutionResult, ServerError> {
        let timeout = requested_timeout
            .map(|value| value.min(self.default_timeout))
            .unwrap_or(self.default_timeout);
        let control = JobControl::new(timeout)?;
        let (reply, receiver) = mpsc::channel();
        self.metrics.record_submit(requests.len());
        let job = WorkerJob {
            responses: Vec::with_capacity(requests.len()),
            requests,
            next: 0,
            mode: CrossRequestMode::Serial,
            reply,
            control: control.clone(),
        };
        self.sender
            .try_send(WorkerCommand::Submit(job))
            .map_err(|error| {
                self.metrics.record_rejected();
                match error {
                    mpsc::TrySendError::Full(_) => ServerError::Busy,
                    mpsc::TrySendError::Disconnected(_) => ServerError::SessionUnavailable,
                }
            })?;
        match receiver.recv_timeout(timeout) {
            Ok(result) => result.map_err(map_worker_failure),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                control.cancel();
                self.metrics
                    .0
                    .caller_deadlines
                    .fetch_add(1, Ordering::Relaxed);
                Err(ServerError::RequestDeadline)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(ServerError::SessionUnavailable),
        }
    }

    fn metrics_json(&self, phase: LifecyclePhase) -> String {
        self.metrics.snapshot(self.capabilities, phase).to_string()
    }

    fn inflight_jobs(&self) -> usize {
        self.metrics.0.inflight_jobs.load(Ordering::Acquire)
    }
}

fn map_worker_failure(error: WorkerFailure) -> ServerError {
    match error.code.as_str() {
        "REQUEST_CANCELLED" => ServerError::RequestCancelled,
        "REQUEST_DEADLINE_EXCEEDED" => ServerError::RequestDeadline,
        _ => ServerError::Session {
            code: error.code,
            message: error.message,
        },
    }
}

fn run_scheduler<E>(
    executor: &mut E,
    receiver: mpsc::Receiver<WorkerCommand>,
    fairness_quantum: usize,
    max_fused_items: usize,
    fused_memory_budget_bytes: Option<u64>,
    pending_capacity: usize,
    metrics: ServerMetrics,
) where
    E: CrossRequestExecutor<Error = HostError>,
{
    let mut pending = VecDeque::<WorkerJob>::new();
    let mut disconnected = false;

    loop {
        if pending.is_empty() {
            if disconnected {
                break;
            }
            match receiver.recv() {
                Ok(WorkerCommand::Submit(job)) => pending.push_back(job),
                Err(_) => {
                    disconnected = true;
                    continue;
                }
            }
        }

        if !disconnected && pending.len() < pending_capacity {
            match receiver.try_recv() {
                Ok(WorkerCommand::Submit(job)) => pending.push_back(job),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => disconnected = true,
            }
        }

        let Some(mut job) = pending.pop_front() else {
            continue;
        };
        if job.control.is_cancelled() {
            metrics
                .0
                .cancelled_before_execution
                .fetch_add(job.remaining(), Ordering::Relaxed);
            metrics.record_failed();
            let _ = job.reply.send(Err(WorkerFailure {
                code: "REQUEST_CANCELLED".to_owned(),
                message: "request was cancelled before remaining work started".to_owned(),
            }));
            continue;
        }
        if job.control.deadline_expired() {
            metrics
                .0
                .cancelled_before_execution
                .fetch_add(job.remaining(), Ordering::Relaxed);
            metrics.record_failed();
            let _ = job.reply.send(Err(WorkerFailure {
                code: "REQUEST_DEADLINE_EXCEEDED".to_owned(),
                message: "request deadline expired before remaining work started".to_owned(),
            }));
            continue;
        }

        let mut plan = match plan_cross_request_with_budget(
            BatchPlanningPolicy::Auto,
            executor.capabilities(),
            job.remaining(),
            fairness_quantum,
            CrossRequestBudget {
                max_fused_items,
                memory_budget_bytes: fused_memory_budget_bytes,
            },
        ) {
            Ok(plan) => plan,
            Err(error) => {
                metrics.record_failed();
                let _ = job.reply.send(Err(WorkerFailure {
                    code: "EXECUTION_PLAN_ERROR".to_owned(),
                    message: error.to_string(),
                }));
                continue;
            }
        };

        let remaining = &job.requests[job.next..];
        if plan.mode == CrossRequestMode::Fused {
            let slice = &remaining[..plan.turn_items.min(remaining.len())];
            match executor.can_execute_fused(slice, &job.control) {
                Ok(true) => {}
                Ok(false) => {
                    plan.mode = CrossRequestMode::Serial;
                }
                Err(error) => {
                    metrics.record_failed();
                    let _ = job.reply.send(Err(WorkerFailure {
                        code: error.code().to_owned(),
                        message: error.to_string(),
                    }));
                    continue;
                }
            }
        }
        match execute_turn(executor, remaining, plan, &job.control) {
            Ok(turn_responses) => {
                match plan.mode {
                    CrossRequestMode::Serial => {
                        metrics.0.serial_turns.fetch_add(1, Ordering::Relaxed);
                    }
                    CrossRequestMode::Fused => {
                        metrics.0.fused_turns.fetch_add(1, Ordering::Relaxed);
                        job.mode = CrossRequestMode::Fused;
                    }
                }
                job.next += turn_responses.len();
                job.responses.extend(turn_responses);
                if job.next == job.requests.len() {
                    let request_count = job.requests.len();
                    metrics.record_complete(request_count);
                    let _ = job.reply.send(Ok(BatchExecutionResult {
                        responses: job.responses,
                        mode: job.mode,
                    }));
                } else {
                    pending.push_back(job);
                }
            }
            Err(error) => {
                let (code, message) = match error {
                    TurnError::Cancelled => (
                        "REQUEST_CANCELLED".to_owned(),
                        "request was cancelled".to_owned(),
                    ),
                    TurnError::DeadlineExceeded => (
                        "REQUEST_DEADLINE_EXCEEDED".to_owned(),
                        "request deadline expired before the next scheduler turn".to_owned(),
                    ),
                    TurnError::Contract(error) => {
                        ("EXECUTION_CONTRACT_ERROR".to_owned(), error.to_string())
                    }
                    TurnError::Executor(error) => (error.code().to_owned(), error.to_string()),
                };
                metrics.record_failed();
                let _ = job.reply.send(Err(WorkerFailure { code, message }));
            }
        }
    }
}

/// Transport-independent host application backed by one native session worker.
#[derive(Clone)]
pub struct ServerApp {
    broker: SessionBroker,
    limits: ServerLimits,
    lifecycle: ServerLifecycle,
}

impl ServerApp {
    pub fn open(config: &ServerConfig) -> Result<Self, ServerError> {
        config.validate()?;
        let broker = SessionBroker::start(config.load.clone(), &config.transport)?;
        Ok(Self {
            broker,
            limits: config.limits,
            lifecycle: ServerLifecycle::default(),
        })
    }

    /// Stop admitting new inference work while allowing already-admitted jobs to finish.
    pub fn begin_drain(&self) {
        self.lifecycle.begin_drain();
    }

    pub fn handle(&self, method: &str, path: &str, body: &str) -> HttpResponse {
        self.handle_with_timeout(method, path, body, None)
    }

    fn handle_with_timeout(
        &self,
        method: &str,
        path: &str,
        body: &str,
        timeout: Option<Duration>,
    ) -> HttpResponse {
        let command = match decode_command(method, path, body, self.limits) {
            Ok(command) => command,
            Err(error) => return protocol_error_response(&error),
        };
        match self.execute(command, timeout) {
            Ok(response) => response,
            Err(error) => server_error_response(&error),
        }
    }

    fn execute(
        &self,
        command: ServerCommand,
        timeout: Option<Duration>,
    ) -> Result<HttpResponse, ServerError> {
        match command {
            ServerCommand::Health => Ok(HttpResponse::json(
                200,
                json!({
                    "status": "ok",
                    "lifecycle": self.lifecycle.phase().as_str(),
                    "_turbojev": {"contract": "turbojev.server-health.v1"}
                })
                .to_string(),
            )),
            ServerCommand::Ready => {
                let ready = self.lifecycle.is_ready();
                Ok(HttpResponse::json(
                    if ready { 200 } else { 503 },
                    json!({
                        "status": if ready { "ready" } else { "not-ready" },
                        "lifecycle": self.lifecycle.phase().as_str(),
                        "_turbojev": {"contract": "turbojev.server-readiness.v1"}
                    })
                    .to_string(),
                ))
            }
            ServerCommand::Models => {
                let model = self.broker.info.as_ref().clone();
                Ok(HttpResponse::json(
                    200,
                    json!({
                        "object": "list",
                        "data": [model],
                        "_turbojev": {"contract": "turbojev.server-models.v1"}
                    })
                    .to_string(),
                ))
            }
            ServerCommand::Metrics => Ok(HttpResponse::json(
                200,
                self.broker.metrics_json(self.lifecycle.phase()),
            )),
            ServerCommand::Evaluate { profile, request } => {
                if !self.lifecycle.is_ready() {
                    return Err(ServerError::Draining);
                }
                let result = self.broker.evaluate(vec![request.clone()], timeout)?;
                let response = result
                    .responses
                    .into_iter()
                    .next()
                    .ok_or(ServerError::EmptyWorkerResponse)?;
                let body = encode_evaluation_response(profile, &request, &response)
                    .map_err(ServerError::Protocol)?;
                Ok(HttpResponse::json(200, body))
            }
            ServerCommand::Batch { profile, requests } => {
                if !self.lifecycle.is_ready() {
                    return Err(ServerError::Draining);
                }
                let result = self.broker.evaluate(requests.clone(), timeout)?;
                let body = encode_batch_response_with_execution(
                    profile,
                    &requests,
                    &result.responses,
                    result.mode,
                )
                .map_err(ServerError::Protocol)?;
                Ok(HttpResponse::json(200, body))
            }
        }
    }
}
/// Minimal HTTP response used by both the TCP transport and tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
    pub headers: BTreeMap<String, String>,
}

impl HttpResponse {
    fn json(status: u16, body: String) -> Self {
        Self {
            status,
            body,
            headers: BTreeMap::from([
                (
                    "Content-Type".to_owned(),
                    "application/json; charset=utf-8".to_owned(),
                ),
                ("Cache-Control".to_owned(), "no-store".to_owned()),
            ]),
        }
    }
}

#[derive(Debug)]
struct ParsedHttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: String,
}

/// Blocking loopback HTTP/1.1 server. Each connection gets a short-lived host
/// thread, while inference remains serialized through the bounded session worker.
/// v0.23 also exposes a programmatic graceful-shutdown token for embedders.
pub struct HttpServer {
    config: ServerConfig,
    app: ServerApp,
}

impl HttpServer {
    pub fn open(config: ServerConfig) -> Result<Self, ServerError> {
        config.validate()?;
        let app = ServerApp::open(&config)?;
        Ok(Self { config, app })
    }

    /// Serve until the process is externally terminated. Embedders that need a
    /// graceful drain should use `serve_until` and retain the cloned token.
    pub fn serve(self) -> Result<(), ServerError> {
        self.serve_until(ShutdownToken::new())
    }

    /// Serve until `shutdown.request()` is called. New accepts stop immediately,
    /// readiness flips to false, and already-admitted work receives a bounded
    /// grace period before the host returns.
    pub fn serve_until(self, shutdown: ShutdownToken) -> Result<(), ServerError> {
        let address = self.config.validate()?;
        let listener = TcpListener::bind(address).map_err(ServerError::Io)?;
        listener.set_nonblocking(true).map_err(ServerError::Io)?;
        let active = Arc::new(AtomicUsize::new(0));
        let config = Arc::new(self.config);
        let app = Arc::new(self.app);

        while !shutdown.is_requested() {
            match listener.accept() {
                Ok((mut stream, _peer)) => {
                    if !try_acquire(&active, config.transport.max_connections) {
                        let response = response_with_context(
                            HttpResponse::json(
                                503,
                                stable_server_error(
                                    "SERVER_BUSY",
                                    "maximum concurrent connections reached",
                                ),
                            ),
                            &next_request_id(),
                            config.deployment.mode,
                        );
                        let _ = write_http_response(&mut stream, &response);
                        continue;
                    }
                    let active_for_thread = Arc::clone(&active);
                    let config_for_thread = Arc::clone(&config);
                    let app_for_thread = Arc::clone(&app);
                    thread::Builder::new()
                        .name("turbojev-http".to_owned())
                        .spawn(move || {
                            let _guard = ConnectionGuard(active_for_thread);
                            let _ =
                                handle_connection(&mut stream, &config_for_thread, &app_for_thread);
                        })
                        .map_err(ServerError::Spawn)?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(ACCEPT_POLL_MS));
                }
                Err(error) => return Err(ServerError::Io(error)),
            }
        }

        app.begin_drain();
        let drain_deadline = Instant::now()
            .checked_add(Duration::from_millis(config.transport.drain_timeout_ms))
            .ok_or_else(|| ServerError::InvalidConfig("drain timeout is too large".to_owned()))?;
        while active.load(Ordering::Acquire) != 0 || app.broker.inflight_jobs() != 0 {
            if Instant::now() >= drain_deadline {
                app.lifecycle.mark_stopped();
                return Err(ServerError::DrainTimeout);
            }
            thread::sleep(Duration::from_millis(ACCEPT_POLL_MS));
        }
        app.lifecycle.mark_stopped();
        Ok(())
    }
}

struct ConnectionGuard(Arc<AtomicUsize>);
impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

fn try_acquire(active: &AtomicUsize, maximum: usize) -> bool {
    let mut current = active.load(Ordering::Acquire);
    loop {
        if current >= maximum {
            return false;
        }
        match active.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(actual) => current = actual,
        }
    }
}

fn next_request_id() -> String {
    let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("tjreq-{sequence:016x}")
}

fn request_id_from_headers(
    headers: &BTreeMap<String, String>,
) -> Result<Option<String>, HttpParseError> {
    let value = headers
        .get(TURBOJEV_REQUEST_ID_HEADER)
        .or_else(|| headers.get(GENERIC_REQUEST_ID_HEADER));
    let Some(value) = value else { return Ok(None) };
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        })
    {
        return Err(HttpParseError::InvalidRequestId);
    }
    Ok(Some(value.clone()))
}

fn validate_deployment_headers(
    request: &ParsedHttpRequest,
    deployment: &DeploymentConfig,
) -> Result<(), HttpParseError> {
    if deployment.mode != DeploymentMode::ReverseProxy || !deployment.require_forwarded_https {
        return Ok(());
    }
    if matches!(request.path.split('?').next(), Some("/healthz" | "/readyz")) {
        return Ok(());
    }
    let forwarded = request
        .headers
        .get(FORWARDED_PROTO_HEADER)
        .and_then(|value| value.split(',').next())
        .map(str::trim);
    if !matches!(forwarded, Some(value) if value.eq_ignore_ascii_case("https")) {
        return Err(HttpParseError::ForwardedHttpsRequired);
    }
    Ok(())
}

fn response_with_context(
    mut response: HttpResponse,
    request_id: &str,
    deployment_mode: DeploymentMode,
) -> HttpResponse {
    response
        .headers
        .insert("X-TurboJev-Request-Id".to_owned(), request_id.to_owned());
    response.headers.insert(
        "X-TurboJev-Deployment-Mode".to_owned(),
        deployment_mode.as_str().to_owned(),
    );
    response
}

fn handle_connection(
    stream: &mut TcpStream,
    config: &ServerConfig,
    app: &ServerApp,
) -> Result<(), ServerError> {
    let timeout = Some(Duration::from_millis(config.transport.io_timeout_ms));
    stream.set_read_timeout(timeout).map_err(ServerError::Io)?;
    stream.set_write_timeout(timeout).map_err(ServerError::Io)?;
    let fallback_request_id = next_request_id();
    let request = match read_http_request(
        stream,
        config.transport.max_header_bytes,
        config.limits.max_body_bytes,
    ) {
        Ok(request) => request,
        Err(error) => {
            let response = response_with_context(
                HttpResponse::json(
                    error.status(),
                    stable_server_error(error.code(), &error.to_string()),
                ),
                &fallback_request_id,
                config.deployment.mode,
            );
            return write_http_response(stream, &response).map_err(ServerError::Io);
        }
    };
    let request_id = match request_id_from_headers(&request.headers) {
        Ok(Some(request_id)) => request_id,
        Ok(None) => fallback_request_id,
        Err(error) => {
            let response = response_with_context(
                HttpResponse::json(
                    error.status(),
                    stable_server_error(error.code(), &error.to_string()),
                ),
                &fallback_request_id,
                config.deployment.mode,
            );
            return write_http_response(stream, &response).map_err(ServerError::Io);
        }
    };
    if let Err(error) = validate_deployment_headers(&request, &config.deployment) {
        let response = response_with_context(
            HttpResponse::json(
                error.status(),
                stable_server_error(error.code(), &error.to_string()),
            ),
            &request_id,
            config.deployment.mode,
        );
        return write_http_response(stream, &response).map_err(ServerError::Io);
    }
    let request_timeout =
        match request_timeout_from_headers(&request.headers, config.transport.session_timeout_ms) {
            Ok(timeout) => timeout,
            Err(error) => {
                let response = response_with_context(
                    HttpResponse::json(
                        error.status(),
                        stable_server_error(error.code(), &error.to_string()),
                    ),
                    &request_id,
                    config.deployment.mode,
                );
                return write_http_response(stream, &response).map_err(ServerError::Io);
            }
        };
    let response = response_with_context(
        app.handle_with_timeout(
            &request.method,
            &request.path,
            &request.body,
            request_timeout,
        ),
        &request_id,
        config.deployment.mode,
    );
    write_http_response(stream, &response).map_err(ServerError::Io)
}

fn request_timeout_from_headers(
    headers: &BTreeMap<String, String>,
    server_max_ms: u64,
) -> Result<Option<Duration>, HttpParseError> {
    let Some(raw) = headers.get(REQUEST_TIMEOUT_HEADER) else {
        return Ok(None);
    };
    let requested = raw
        .parse::<u64>()
        .map_err(|_| HttpParseError::InvalidRequestTimeout)?;
    if requested == 0 {
        return Err(HttpParseError::InvalidRequestTimeout);
    }
    Ok(Some(Duration::from_millis(requested.min(server_max_ms))))
}

fn read_http_request<R: Read>(
    reader: &mut R,
    max_header: usize,
    max_body: usize,
) -> Result<ParsedHttpRequest, HttpParseError> {
    let mut buffer = Vec::new();
    let header_end = loop {
        if buffer.len() > max_header {
            return Err(HttpParseError::HeadersTooLarge);
        }
        if let Some(position) = find_header_end(&buffer) {
            break position;
        }
        let mut chunk = [0_u8; 2048];
        let read = reader.read(&mut chunk).map_err(HttpParseError::Io)?;
        if read == 0 {
            return Err(HttpParseError::UnexpectedEof);
        }
        buffer.extend_from_slice(&chunk[..read]);
    };

    let header_text =
        std::str::from_utf8(&buffer[..header_end]).map_err(|_| HttpParseError::InvalidHeaders)?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().ok_or(HttpParseError::InvalidRequestLine)?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or(HttpParseError::InvalidRequestLine)?;
    let path = request_parts
        .next()
        .ok_or(HttpParseError::InvalidRequestLine)?;
    let version = request_parts
        .next()
        .ok_or(HttpParseError::InvalidRequestLine)?;
    if request_parts.next().is_some() || !matches!(version, "HTTP/1.1" | "HTTP/1.0") {
        return Err(HttpParseError::InvalidRequestLine);
    }
    if !path.starts_with('/') {
        return Err(HttpParseError::InvalidRequestLine);
    }

    let mut content_length = None;
    let mut headers = BTreeMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line.split_once(':').ok_or(HttpParseError::InvalidHeaders)?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "transfer-encoding" && !value.eq_ignore_ascii_case("identity") {
            return Err(HttpParseError::UnsupportedTransferEncoding);
        }
        if name == "content-length" {
            if content_length.is_some() {
                return Err(HttpParseError::DuplicateContentLength);
            }
            content_length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| HttpParseError::InvalidContentLength)?,
            );
        }
        headers.insert(name, value.to_owned());
    }
    let content_length = content_length.unwrap_or(0);
    if content_length > max_body {
        return Err(HttpParseError::BodyTooLarge);
    }
    let method = method.to_owned();
    let path = path.to_owned();
    let body_start = header_end + 4;
    while buffer.len() < body_start + content_length {
        let remaining = body_start + content_length - buffer.len();
        let mut chunk = vec![0_u8; remaining.min(8192)];
        let read = reader.read(&mut chunk).map_err(HttpParseError::Io)?;
        if read == 0 {
            return Err(HttpParseError::UnexpectedEof);
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let body = std::str::from_utf8(&buffer[body_start..body_start + content_length])
        .map_err(|_| HttpParseError::InvalidBodyEncoding)?
        .to_owned();
    Ok(ParsedHttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn write_http_response<W: Write>(writer: &mut W, response: &HttpResponse) -> std::io::Result<()> {
    let reason = reason_phrase(response.status);
    write!(writer, "HTTP/1.1 {} {}\r\n", response.status, reason)?;
    for (name, value) in &response.headers {
        write!(writer, "{}: {}\r\n", name, value)?;
    }
    write!(
        writer,
        "Content-Length: {}\r\nConnection: close\r\n\r\n",
        response.body.len()
    )?;
    writer.write_all(response.body.as_bytes())?;
    writer.flush()
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        422 => "Unprocessable Content",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    }
}

fn protocol_error_response(error: &ProtocolError) -> HttpResponse {
    HttpResponse::json(error.status(), encode_error(error))
}

fn server_error_response(error: &ServerError) -> HttpResponse {
    HttpResponse::json(
        error.status(),
        stable_server_error(error.code(), &error.to_string()),
    )
}

fn stable_server_error(code: &str, message: &str) -> String {
    json!({
        "error": {"code": code, "message": message},
        "_turbojev": {"contract": SERVER_ERROR_CONTRACT}
    })
    .to_string()
}

#[derive(Debug, Error)]
enum HttpParseError {
    #[error("HTTP headers exceed configured limit")]
    HeadersTooLarge,
    #[error("unexpected end of HTTP request")]
    UnexpectedEof,
    #[error("HTTP headers are not valid UTF-8")]
    InvalidHeaders,
    #[error("invalid HTTP request line")]
    InvalidRequestLine,
    #[error("transfer encoding is unsupported; send Content-Length")]
    UnsupportedTransferEncoding,
    #[error("duplicate Content-Length header")]
    DuplicateContentLength,
    #[error("invalid Content-Length header")]
    InvalidContentLength,
    #[error("X-TurboJev-Timeout-Ms must be a positive integer")]
    InvalidRequestTimeout,
    #[error("request id must be 1..128 safe ASCII token characters")]
    InvalidRequestId,
    #[error("reverse-proxy mode requires X-Forwarded-Proto: https for non-probe routes")]
    ForwardedHttpsRequired,
    #[error("HTTP body exceeds configured limit")]
    BodyTooLarge,
    #[error("HTTP JSON body is not valid UTF-8")]
    InvalidBodyEncoding,
    #[error("I/O error while reading HTTP request: {0}")]
    Io(#[source] std::io::Error),
}

impl HttpParseError {
    const fn status(&self) -> u16 {
        match self {
            Self::HeadersTooLarge | Self::BodyTooLarge => 413,
            Self::ForwardedHttpsRequired => 403,
            Self::UnsupportedTransferEncoding => 400,
            Self::UnexpectedEof
            | Self::InvalidHeaders
            | Self::InvalidRequestLine
            | Self::DuplicateContentLength
            | Self::InvalidContentLength
            | Self::InvalidRequestTimeout
            | Self::InvalidRequestId
            | Self::InvalidBodyEncoding
            | Self::Io(_) => 400,
        }
    }
    const fn code(&self) -> &'static str {
        match self {
            Self::HeadersTooLarge => "HEADERS_TOO_LARGE",
            Self::UnexpectedEof => "UNEXPECTED_EOF",
            Self::InvalidHeaders => "INVALID_HEADERS",
            Self::InvalidRequestLine => "INVALID_REQUEST_LINE",
            Self::UnsupportedTransferEncoding => "UNSUPPORTED_TRANSFER_ENCODING",
            Self::DuplicateContentLength => "DUPLICATE_CONTENT_LENGTH",
            Self::InvalidContentLength => "INVALID_CONTENT_LENGTH",
            Self::InvalidRequestTimeout => "INVALID_REQUEST_TIMEOUT",
            Self::InvalidRequestId => "INVALID_REQUEST_ID",
            Self::ForwardedHttpsRequired => "FORWARDED_HTTPS_REQUIRED",
            Self::BodyTooLarge => "BODY_TOO_LARGE",
            Self::InvalidBodyEncoding => "INVALID_BODY_ENCODING",
            Self::Io(_) => "HTTP_IO_ERROR",
        }
    }
}

/// Server startup, queue, session, or transport failure.
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("invalid server configuration: {0}")]
    InvalidConfig(String),
    #[error("server protocol error: {0}")]
    Protocol(#[source] ProtocolError),
    #[error("failed to spawn server thread: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("server I/O error: {0}")]
    Io(#[source] std::io::Error),
    #[error("session worker failed ({code}): {message}")]
    Session { code: String, message: String },
    #[error("session worker is unavailable")]
    SessionUnavailable,
    #[error("session worker timed out")]
    SessionTimeout,
    #[error("server request queue is full")]
    Busy,
    #[error("server is draining and is not accepting new inference work")]
    Draining,
    #[error("graceful drain exceeded the configured timeout")]
    DrainTimeout,
    #[error("request deadline exceeded")]
    RequestDeadline,
    #[error("request was cancelled before execution completed")]
    RequestCancelled,
    #[error("session worker returned no response")]
    EmptyWorkerResponse,
}

impl ServerError {
    pub const fn status(&self) -> u16 {
        match self {
            Self::InvalidConfig(_) | Self::Protocol(_) => 400,
            Self::Busy | Self::Draining | Self::SessionUnavailable => 503,
            Self::SessionTimeout
            | Self::DrainTimeout
            | Self::RequestDeadline
            | Self::RequestCancelled => 504,
            Self::Spawn(_) | Self::Io(_) | Self::Session { .. } | Self::EmptyWorkerResponse => 500,
        }
    }
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidConfig(_) => "INVALID_SERVER_CONFIG",
            Self::Protocol(_) => "SERVER_PROTOCOL_ERROR",
            Self::Spawn(_) => "THREAD_SPAWN_ERROR",
            Self::Io(_) => "SERVER_IO_ERROR",
            Self::Session { .. } => "SESSION_ERROR",
            Self::SessionUnavailable => "SESSION_UNAVAILABLE",
            Self::SessionTimeout => "SESSION_TIMEOUT",
            Self::Busy => "SERVER_BUSY",
            Self::Draining => "SERVER_DRAINING",
            Self::DrainTimeout => "DRAIN_TIMEOUT",
            Self::RequestDeadline => "REQUEST_DEADLINE_EXCEEDED",
            Self::RequestCancelled => "REQUEST_CANCELLED",
            Self::EmptyWorkerResponse => "EMPTY_WORKER_RESPONSE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture_config() -> ServerConfig {
        ServerConfig {
            bind: DEFAULT_BIND.to_owned(),
            load: NativeLoadOptions {
                worker: None,
                model: "fixture://server".to_owned(),
                n_ctx: 64,
                n_batch: 64,
                n_seq_max: 2,
                n_gpu_layers: 0,
                n_threads: 0,
                n_threads_batch: 0,
                execution: turbojev_host_native::HostExecutionOptions {
                    order_robustness: turbojev_host_native::HostOrderRobustness::None,
                    ..Default::default()
                },
                prompt: Default::default(),
                fixture_logits: BTreeMap::from([("route".to_owned(), vec![0.0, 4.0])]),
            },
            limits: ServerLimits::default(),
            transport: TransportConfig {
                fairness_quantum: 4,
                max_fused_items: 8,
                fused_memory_budget_bytes: Some(16 * 1024 * 1024),
                ..TransportConfig::default()
            },
            deployment: DeploymentConfig::default(),
        }
    }

    #[test]
    fn parser_reads_content_length_body() {
        let raw = b"POST /v1/evaluate HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
        let mut cursor = Cursor::new(raw.as_slice());
        let request = read_http_request(&mut cursor, 1024, 1024).unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/evaluate");
        assert_eq!(request.body, "{}");
    }

    #[test]
    fn fixture_server_app_runs_through_native_session_worker() {
        let app = ServerApp::open(&fixture_config()).unwrap();
        let body = r#"{"state":"lost","questions":{"route":{"type":"choice","instructions":"route","criteria":{"billing":"billing","shipping":"shipping"}}}}"#;
        let response = app.handle("POST", "/v1/evaluate", body);
        assert_eq!(response.status, 200);
        let value: Value = serde_json::from_str(&response.body).unwrap();
        assert_eq!(value["answers"]["route"]["choice"], "shipping");
    }

    #[test]
    fn timeout_header_is_parsed_and_clamped() {
        let headers = BTreeMap::from([(REQUEST_TIMEOUT_HEADER.to_owned(), "5000".to_owned())]);
        assert_eq!(
            request_timeout_from_headers(&headers, 1000).unwrap(),
            Some(Duration::from_millis(1000)),
        );
    }

    #[test]
    fn metrics_endpoint_reports_fixture_fused_capabilities() {
        let app = ServerApp::open(&fixture_config()).unwrap();
        let response = app.handle("GET", "/v1/server/metrics", "");
        assert_eq!(response.status, 200);
        let value: Value = serde_json::from_str(&response.body).unwrap();
        assert_eq!(value["capabilities"]["fusedBatch"], true);
        assert_eq!(value["capabilities"]["maxFusedItems"], 64);
        assert_eq!(value["capabilities"]["hardDeadlines"], false);
    }

    #[test]
    fn fixture_batch_uses_true_fused_backend_path() {
        let app = ServerApp::open(&fixture_config()).unwrap();
        let body = r#"{"requests":[{"state":"one","questions":{"route":{"type":"choice","instructions":"route","criteria":{"billing":"billing","shipping":"shipping"}}}},{"state":"two","questions":{"route":{"type":"choice","instructions":"route","criteria":{"billing":"billing","shipping":"shipping"}}}}]}"#;
        let response = app.handle("POST", "/v1/batch", body);
        assert_eq!(response.status, 200);
        let value: Value = serde_json::from_str(&response.body).unwrap();
        assert_eq!(value["_turbojev"]["crossRequestExecution"], "fused-batch");
    }

    #[test]
    fn readiness_flips_before_liveness_during_drain() {
        let app = ServerApp::open(&fixture_config()).unwrap();
        assert_eq!(app.handle("GET", "/readyz", "").status, 200);
        app.begin_drain();
        assert_eq!(app.handle("GET", "/readyz", "").status, 503);
        assert_eq!(app.handle("GET", "/healthz", "").status, 200);
        let body = r#"{"state":"lost","questions":{"route":{"type":"choice","instructions":"route","criteria":{"billing":"billing","shipping":"shipping"}}}}"#;
        let response = app.handle("POST", "/v1/evaluate", body);
        assert_eq!(response.status, 503);
    }

    #[test]
    fn request_id_validation_rejects_header_injection() {
        let headers = BTreeMap::from([(
            TURBOJEV_REQUEST_ID_HEADER.to_owned(),
            "bad\r\nid".to_owned(),
        )]);
        assert!(request_id_from_headers(&headers).is_err());
        let safe = BTreeMap::from([(
            GENERIC_REQUEST_ID_HEADER.to_owned(),
            "edge-req:abc_123".to_owned(),
        )]);
        assert_eq!(
            request_id_from_headers(&safe).unwrap().as_deref(),
            Some("edge-req:abc_123")
        );
    }

    #[test]
    fn reverse_proxy_mode_requires_forwarded_https_for_inference() {
        let request = ParsedHttpRequest {
            method: "POST".to_owned(),
            path: "/v1/evaluate".to_owned(),
            headers: BTreeMap::new(),
            body: "{}".to_owned(),
        };
        let deployment = DeploymentConfig {
            mode: DeploymentMode::ReverseProxy,
            require_forwarded_https: true,
        };
        assert!(matches!(
            validate_deployment_headers(&request, &deployment),
            Err(HttpParseError::ForwardedHttpsRequired)
        ));
        let mut secure_headers = BTreeMap::new();
        secure_headers.insert(FORWARDED_PROTO_HEADER.to_owned(), "HTTPS".to_owned());
        let secure = ParsedHttpRequest {
            method: request.method.clone(),
            path: request.path.clone(),
            headers: secure_headers,
            body: request.body.clone(),
        };
        assert!(validate_deployment_headers(&secure, &deployment).is_ok());
        let probe = ParsedHttpRequest {
            path: "/readyz".to_owned(),
            ..request
        };
        assert!(validate_deployment_headers(&probe, &deployment).is_ok());
    }

    #[test]
    fn non_loopback_bind_is_rejected() {
        let mut config = fixture_config();
        config.bind = "0.0.0.0:3217".to_owned();
        assert!(config.validate().is_err());
    }
}
