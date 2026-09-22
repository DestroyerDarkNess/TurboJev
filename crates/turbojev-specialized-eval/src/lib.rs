#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable evaluation contracts for TurboJev decision-specialized models.
//!
//! This crate accepts framework/runtime-neutral evaluation runs, verifies that
//! generic and specialized models were measured on the same semantic benchmark,
//! computes quality/calibration/performance/robustness metrics, and emits a
//! deterministic comparison report. It performs no model loading or training.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use turbojev_benchmark::{summarize, TimingSummary};
use turbojev_calibration::{evaluate_calibration, CalibrationExample, CalibrationMetrics};
use turbojev_specialized::{ModelPair, SpecializationEvaluationPlan};

pub const SPECIALIZATION_RUN_CONTRACT: &str = "turbojev.specialization-run.v1";
pub const SPECIALIZATION_REPORT_CONTRACT: &str = "turbojev.specialization-report.v1";
pub const SPECIALIZATION_BUNDLE_CONTRACT: &str = "turbojev.specialization-bundle.v1";

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VariantKind {
    Baseline,
    CandidateOrder,
    Paraphrase,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationCalibration {
    pub method: String,
    pub temperature: f64,
    pub fitted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_id: Option<String>,
    pub ece_bins: usize,
}

impl EvaluationCalibration {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        if self.method != "temperature-softmax" {
            return Err(EvaluationError::UnsupportedCalibrationMethod(
                self.method.clone(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(EvaluationError::InvalidTemperature(self.temperature));
        }
        if self.ece_bins == 0 {
            return Err(EvaluationError::ZeroEceBins);
        }
        if self.fitted {
            let profile = self
                .profile_id
                .as_deref()
                .ok_or(EvaluationError::MissingCalibrationProfile)?;
            require_text("calibration.profile_id", profile)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationModelIdentity {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_revision: Option<String>,
    pub prompt_ir: String,
    pub renderer_profile: String,
    pub calibration: EvaluationCalibration,
}

impl EvaluationModelIdentity {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        require_text("model.id", &self.id)?;
        require_text("model.prompt_ir", &self.prompt_ir)?;
        require_text("model.renderer_profile", &self.renderer_profile)?;
        if let Some(revision) = &self.revision {
            require_text("model.revision", revision)?;
        }
        if let Some(revision) = &self.tokenizer_revision {
            require_text("model.tokenizer_revision", revision)?;
        }
        self.calibration.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunEnvironment {
    pub backend: String,
    pub host: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner_version: Option<String>,
}

impl RunEnvironment {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        require_text("environment.backend", &self.backend)?;
        require_text("environment.host", &self.host)?;
        for (field, value) in [
            ("environment.quantization", self.quantization.as_deref()),
            ("environment.execution_path", self.execution_path.as_deref()),
            ("environment.runner_version", self.runner_version.as_deref()),
        ] {
            if let Some(value) = value {
                require_text(field, value)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationObservation {
    /// Stable benchmark observation id. The same id must be present in both
    /// generic and specialized runs for a valid pair comparison.
    pub id: String,
    /// Semantic family joining baseline and controlled variants.
    pub family_id: String,
    pub case_id: String,
    pub question_id: String,
    pub variant: VariantKind,
    /// Semantic candidate keys in the exact order corresponding to `logits`.
    pub candidate_keys: Vec<String>,
    pub logits: Vec<f32>,
    pub target_key: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// End-to-end decision latency measured by the external runner.
    pub latency_seconds: f64,
    /// Runner-reported peak resident/device memory for this observation.
    pub peak_memory_bytes: u64,
}

impl EvaluationObservation {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        for (field, value) in [
            ("observation.id", self.id.as_str()),
            ("observation.family_id", self.family_id.as_str()),
            ("observation.case_id", self.case_id.as_str()),
            ("observation.question_id", self.question_id.as_str()),
            ("observation.target_key", self.target_key.as_str()),
        ] {
            require_text(field, value)?;
        }
        if self.candidate_keys.len() < 2 || self.candidate_keys.len() != self.logits.len() {
            return Err(EvaluationError::CandidateShape);
        }
        let mut keys = HashSet::with_capacity(self.candidate_keys.len());
        for key in &self.candidate_keys {
            require_text("observation.candidate_keys", key)?;
            if !keys.insert(key.as_str()) {
                return Err(EvaluationError::DuplicateCandidateKey);
            }
        }
        if !keys.contains(self.target_key.as_str()) {
            return Err(EvaluationError::MissingTargetKey(self.target_key.clone()));
        }
        if self.logits.iter().any(|value| !value.is_finite()) {
            return Err(EvaluationError::NonFiniteLogit);
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(EvaluationError::InvalidWeight(self.weight));
        }
        if !self.latency_seconds.is_finite() || self.latency_seconds <= 0.0 {
            return Err(EvaluationError::InvalidLatency(self.latency_seconds));
        }
        if self.peak_memory_bytes == 0 {
            return Err(EvaluationError::ZeroPeakMemory);
        }
        Ok(())
    }

    fn target_index(&self) -> usize {
        self.candidate_keys
            .iter()
            .position(|key| key == &self.target_key)
            .expect("validated observation target must exist")
    }

    fn predicted_key(&self) -> &str {
        let mut best = 0usize;
        for index in 1..self.logits.len() {
            if self.logits[index] > self.logits[best] {
                best = index;
            }
        }
        &self.candidate_keys[best]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecializationRun {
    pub contract: String,
    pub id: String,
    pub plan_id: String,
    pub benchmark_suite: String,
    pub model: EvaluationModelIdentity,
    pub environment: RunEnvironment,
    pub observations: Vec<EvaluationObservation>,
}

impl SpecializationRun {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        if self.contract != SPECIALIZATION_RUN_CONTRACT {
            return Err(EvaluationError::UnsupportedContract(self.contract.clone()));
        }
        require_text("run.id", &self.id)?;
        require_text("run.plan_id", &self.plan_id)?;
        require_text("run.benchmark_suite", &self.benchmark_suite)?;
        self.model.validate()?;
        self.environment.validate()?;
        if self.observations.is_empty() {
            return Err(EvaluationError::EmptyRun);
        }

        let mut ids = HashSet::with_capacity(self.observations.len());
        let mut families: BTreeMap<&str, Vec<&EvaluationObservation>> = BTreeMap::new();
        let mut candidate_order_count = 0usize;
        let mut paraphrase_count = 0usize;
        for observation in &self.observations {
            observation.validate()?;
            if !ids.insert(observation.id.as_str()) {
                return Err(EvaluationError::DuplicateObservation(
                    observation.id.clone(),
                ));
            }
            match observation.variant {
                VariantKind::CandidateOrder => candidate_order_count += 1,
                VariantKind::Paraphrase => paraphrase_count += 1,
                VariantKind::Baseline => {}
            }
            families
                .entry(observation.family_id.as_str())
                .or_default()
                .push(observation);
        }
        if candidate_order_count == 0 {
            return Err(EvaluationError::MissingVariant("candidate-order"));
        }
        if paraphrase_count == 0 {
            return Err(EvaluationError::MissingVariant("paraphrase"));
        }

        for (family_id, family) in families {
            let baselines: Vec<_> = family
                .iter()
                .copied()
                .filter(|observation| observation.variant == VariantKind::Baseline)
                .collect();
            if baselines.len() != 1 {
                return Err(EvaluationError::BaselineCount {
                    family_id: family_id.to_owned(),
                    count: baselines.len(),
                });
            }
            let baseline = baselines[0];
            let baseline_keys: HashSet<_> = baseline.candidate_keys.iter().collect();
            for observation in family {
                if observation.case_id != baseline.case_id
                    || observation.question_id != baseline.question_id
                    || observation.target_key != baseline.target_key
                {
                    return Err(EvaluationError::FamilySemanticDrift(family_id.to_owned()));
                }
                let keys: HashSet<_> = observation.candidate_keys.iter().collect();
                if keys != baseline_keys {
                    return Err(EvaluationError::FamilyCandidateDrift(family_id.to_owned()));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustnessMetrics {
    pub count: usize,
    pub total_weight: f64,
    /// Fraction whose semantic prediction is invariant to the controlled variant.
    pub invariance: f64,
    /// Fraction of variants that remain correct against the reference target.
    pub accuracy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetrics {
    /// Accuracy/NLL/Brier/ECE are computed on baseline observations only.
    pub baseline_quality: CalibrationMetrics,
    /// Timing summarizes all controlled observations in the external run.
    pub timing: TimingSummary,
    pub decisions_per_second: f64,
    pub peak_memory_bytes: u64,
    pub candidate_order: RobustnessMetrics,
    pub paraphrase: RobustnessMetrics,
    pub baseline_observations: usize,
    pub variant_observations: usize,
}

pub fn evaluate_run(run: &SpecializationRun) -> Result<RunMetrics, EvaluationError> {
    run.validate()?;

    let baselines: Vec<&EvaluationObservation> = run
        .observations
        .iter()
        .filter(|observation| observation.variant == VariantKind::Baseline)
        .collect();
    let examples: Vec<CalibrationExample> = baselines
        .iter()
        .map(|observation| CalibrationExample {
            logits: observation.logits.clone(),
            target: observation.target_index(),
            weight: observation.weight,
        })
        .collect();
    let baseline_quality = evaluate_calibration(
        &examples,
        run.model.calibration.temperature,
        run.model.calibration.ece_bins,
    )
    .map_err(|error| EvaluationError::Calibration(error.to_string()))?;

    let timing = summarize(
        run.observations
            .iter()
            .map(|observation| observation.latency_seconds),
    )
    .map_err(|error| EvaluationError::Timing(error.to_string()))?;
    let total_seconds: f64 = run
        .observations
        .iter()
        .map(|observation| observation.latency_seconds)
        .sum();
    if !total_seconds.is_finite() || total_seconds <= 0.0 {
        return Err(EvaluationError::InvalidAggregateLatency);
    }
    let decisions_per_second = run.observations.len() as f64 / total_seconds;
    let peak_memory_bytes = run
        .observations
        .iter()
        .map(|observation| observation.peak_memory_bytes)
        .max()
        .ok_or(EvaluationError::EmptyRun)?;

    let candidate_order = robustness(run, VariantKind::CandidateOrder)?;
    let paraphrase = robustness(run, VariantKind::Paraphrase)?;
    let variant_observations = run.observations.len() - baselines.len();

    Ok(RunMetrics {
        baseline_quality,
        timing,
        decisions_per_second,
        peak_memory_bytes,
        candidate_order,
        paraphrase,
        baseline_observations: baselines.len(),
        variant_observations,
    })
}

fn robustness(
    run: &SpecializationRun,
    kind: VariantKind,
) -> Result<RobustnessMetrics, EvaluationError> {
    let baselines: HashMap<&str, &EvaluationObservation> = run
        .observations
        .iter()
        .filter(|observation| observation.variant == VariantKind::Baseline)
        .map(|observation| (observation.family_id.as_str(), observation))
        .collect();

    let mut count = 0usize;
    let mut total_weight = 0.0;
    let mut invariant_weight = 0.0;
    let mut correct_weight = 0.0;
    for observation in run
        .observations
        .iter()
        .filter(|observation| observation.variant == kind)
    {
        let baseline = baselines
            .get(observation.family_id.as_str())
            .copied()
            .ok_or_else(|| EvaluationError::MissingBaseline(observation.family_id.clone()))?;
        count += 1;
        total_weight += observation.weight;
        if observation.predicted_key() == baseline.predicted_key() {
            invariant_weight += observation.weight;
        }
        if observation.predicted_key() == observation.target_key.as_str() {
            correct_weight += observation.weight;
        }
    }
    if count == 0 || total_weight <= 0.0 {
        return Err(EvaluationError::MissingVariant(match kind {
            VariantKind::CandidateOrder => "candidate-order",
            VariantKind::Paraphrase => "paraphrase",
            VariantKind::Baseline => "baseline",
        }));
    }
    Ok(RobustnessMetrics {
        count,
        total_weight,
        invariance: invariant_weight / total_weight,
        accuracy: correct_weight / total_weight,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricDelta {
    /// Every field is `specialized - generic`; no implicit "winner" is encoded.
    pub accuracy: f64,
    pub nll: f64,
    pub brier: f64,
    pub ece: f64,
    pub median_latency_seconds: f64,
    pub p95_latency_seconds: f64,
    pub decisions_per_second: f64,
    pub peak_memory_bytes: f64,
    pub candidate_order_invariance: f64,
    pub candidate_order_accuracy: f64,
    pub paraphrase_invariance: f64,
    pub paraphrase_accuracy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairComparison {
    pub pair: ModelPair,
    pub generic_run: String,
    pub specialized_run: String,
    pub generic: RunMetrics,
    pub specialized: RunMetrics,
    pub delta: MetricDelta,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecializationReport {
    pub contract: String,
    pub id: String,
    pub plan_id: String,
    pub benchmark_suite: String,
    pub comparisons: Vec<PairComparison>,
}

pub fn evaluate_plan(
    report_id: impl Into<String>,
    plan: &SpecializationEvaluationPlan,
    runs: &[SpecializationRun],
) -> Result<SpecializationReport, EvaluationError> {
    plan.validate()
        .map_err(|error| EvaluationError::Plan(error.to_string()))?;
    if runs.is_empty() {
        return Err(EvaluationError::NoRuns);
    }

    let mut by_model: HashMap<&str, &SpecializationRun> = HashMap::with_capacity(runs.len());
    for run in runs {
        run.validate()?;
        if run.plan_id != plan.id {
            return Err(EvaluationError::PlanIdMismatch {
                run: run.id.clone(),
                expected: plan.id.clone(),
                actual: run.plan_id.clone(),
            });
        }
        if run.benchmark_suite != plan.benchmark_suite {
            return Err(EvaluationError::BenchmarkMismatch {
                run: run.id.clone(),
                expected: plan.benchmark_suite.clone(),
                actual: run.benchmark_suite.clone(),
            });
        }
        if by_model.insert(run.model.id.as_str(), run).is_some() {
            return Err(EvaluationError::DuplicateModelRun(run.model.id.clone()));
        }
    }

    let mut comparisons = Vec::with_capacity(plan.pairs.len());
    for pair in &plan.pairs {
        let generic = by_model
            .get(pair.generic_model.as_str())
            .copied()
            .ok_or_else(|| EvaluationError::MissingModelRun(pair.generic_model.clone()))?;
        let specialized = by_model
            .get(pair.specialized_model.as_str())
            .copied()
            .ok_or_else(|| EvaluationError::MissingModelRun(pair.specialized_model.clone()))?;
        verify_pair_alignment(generic, specialized)?;
        let generic_metrics = evaluate_run(generic)?;
        let specialized_metrics = evaluate_run(specialized)?;
        let delta = metric_delta(&generic_metrics, &specialized_metrics);
        comparisons.push(PairComparison {
            pair: pair.clone(),
            generic_run: generic.id.clone(),
            specialized_run: specialized.id.clone(),
            generic: generic_metrics,
            specialized: specialized_metrics,
            delta,
        });
    }

    let report = SpecializationReport {
        contract: SPECIALIZATION_REPORT_CONTRACT.to_owned(),
        id: report_id.into(),
        plan_id: plan.id.clone(),
        benchmark_suite: plan.benchmark_suite.clone(),
        comparisons,
    };
    report.validate()?;
    Ok(report)
}

fn verify_pair_alignment(
    generic: &SpecializationRun,
    specialized: &SpecializationRun,
) -> Result<(), EvaluationError> {
    if generic.model.prompt_ir != specialized.model.prompt_ir
        || generic.model.renderer_profile != specialized.model.renderer_profile
        || generic.model.tokenizer_revision != specialized.model.tokenizer_revision
    {
        return Err(EvaluationError::ModelEvaluationContractMismatch);
    }
    if generic.environment.backend != specialized.environment.backend
        || generic.environment.host != specialized.environment.host
        || generic.environment.quantization != specialized.environment.quantization
        || generic.environment.execution_path != specialized.environment.execution_path
    {
        return Err(EvaluationError::EnvironmentMismatch);
    }
    if generic.observations.len() != specialized.observations.len() {
        return Err(EvaluationError::ObservationSetMismatch);
    }
    let specialized_by_id: HashMap<_, _> = specialized
        .observations
        .iter()
        .map(|observation| (observation.id.as_str(), observation))
        .collect();
    for left in &generic.observations {
        let right = specialized_by_id
            .get(left.id.as_str())
            .copied()
            .ok_or(EvaluationError::ObservationSetMismatch)?;
        if left.family_id != right.family_id
            || left.case_id != right.case_id
            || left.question_id != right.question_id
            || left.variant != right.variant
            || left.candidate_keys != right.candidate_keys
            || left.target_key != right.target_key
            || (left.weight - right.weight).abs() > 1e-12
        {
            return Err(EvaluationError::ObservationSemanticMismatch(
                left.id.clone(),
            ));
        }
    }
    Ok(())
}

fn metric_delta(generic: &RunMetrics, specialized: &RunMetrics) -> MetricDelta {
    MetricDelta {
        accuracy: specialized.baseline_quality.accuracy - generic.baseline_quality.accuracy,
        nll: specialized.baseline_quality.nll - generic.baseline_quality.nll,
        brier: specialized.baseline_quality.brier - generic.baseline_quality.brier,
        ece: specialized.baseline_quality.ece - generic.baseline_quality.ece,
        median_latency_seconds: specialized.timing.median_seconds - generic.timing.median_seconds,
        p95_latency_seconds: specialized.timing.p95_seconds - generic.timing.p95_seconds,
        decisions_per_second: specialized.decisions_per_second - generic.decisions_per_second,
        peak_memory_bytes: specialized.peak_memory_bytes as f64 - generic.peak_memory_bytes as f64,
        candidate_order_invariance: specialized.candidate_order.invariance
            - generic.candidate_order.invariance,
        candidate_order_accuracy: specialized.candidate_order.accuracy
            - generic.candidate_order.accuracy,
        paraphrase_invariance: specialized.paraphrase.invariance - generic.paraphrase.invariance,
        paraphrase_accuracy: specialized.paraphrase.accuracy - generic.paraphrase.accuracy,
    }
}

impl SpecializationReport {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        if self.contract != SPECIALIZATION_REPORT_CONTRACT {
            return Err(EvaluationError::UnsupportedContract(self.contract.clone()));
        }
        require_text("report.id", &self.id)?;
        require_text("report.plan_id", &self.plan_id)?;
        require_text("report.benchmark_suite", &self.benchmark_suite)?;
        if self.comparisons.is_empty() {
            return Err(EvaluationError::EmptyReport);
        }
        for comparison in &self.comparisons {
            require_text("report.generic_run", &comparison.generic_run)?;
            require_text("report.specialized_run", &comparison.specialized_run)?;
            if comparison.pair.generic_model == comparison.pair.specialized_model {
                return Err(EvaluationError::IdenticalModelPair);
            }
            validate_metrics(&comparison.generic)?;
            validate_metrics(&comparison.specialized)?;
            for value in [
                comparison.delta.accuracy,
                comparison.delta.nll,
                comparison.delta.brier,
                comparison.delta.ece,
                comparison.delta.median_latency_seconds,
                comparison.delta.p95_latency_seconds,
                comparison.delta.decisions_per_second,
                comparison.delta.peak_memory_bytes,
                comparison.delta.candidate_order_invariance,
                comparison.delta.candidate_order_accuracy,
                comparison.delta.paraphrase_invariance,
                comparison.delta.paraphrase_accuracy,
            ] {
                if !value.is_finite() {
                    return Err(EvaluationError::NonFiniteReportMetric);
                }
            }
        }
        Ok(())
    }
}

fn validate_metrics(metrics: &RunMetrics) -> Result<(), EvaluationError> {
    if metrics.baseline_observations == 0 || metrics.variant_observations == 0 {
        return Err(EvaluationError::InvalidReportCounts);
    }
    if !metrics.decisions_per_second.is_finite() || metrics.decisions_per_second <= 0.0 {
        return Err(EvaluationError::InvalidReportThroughput);
    }
    if metrics.peak_memory_bytes == 0 {
        return Err(EvaluationError::ZeroPeakMemory);
    }
    for robustness in [&metrics.candidate_order, &metrics.paraphrase] {
        if robustness.count == 0
            || !robustness.total_weight.is_finite()
            || robustness.total_weight <= 0.0
            || !valid_probability(robustness.invariance)
            || !valid_probability(robustness.accuracy)
        {
            return Err(EvaluationError::InvalidRobustnessMetric);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactDigest {
    pub contract: String,
    pub id: String,
    /// Lowercase or uppercase hexadecimal SHA-256, without a prefix.
    pub sha256: String,
}

impl ArtifactDigest {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        require_text("artifact.contract", &self.contract)?;
        require_text("artifact.id", &self.id)?;
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(EvaluationError::InvalidSha256(self.id.clone()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecializationExperimentBundle {
    pub contract: String,
    pub id: String,
    pub plan: ArtifactDigest,
    pub benchmark: ArtifactDigest,
    pub runs: Vec<ArtifactDigest>,
    pub report: ArtifactDigest,
    #[serde(default)]
    pub recipes: Vec<ArtifactDigest>,
    #[serde(default)]
    pub model_manifests: Vec<ArtifactDigest>,
    #[serde(default)]
    pub calibration_profiles: Vec<ArtifactDigest>,
}

impl SpecializationExperimentBundle {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        if self.contract != SPECIALIZATION_BUNDLE_CONTRACT {
            return Err(EvaluationError::UnsupportedContract(self.contract.clone()));
        }
        require_text("bundle.id", &self.id)?;
        self.plan.validate()?;
        self.benchmark.validate()?;
        self.report.validate()?;
        if self.runs.len() < 2 {
            return Err(EvaluationError::TooFewBundleRuns);
        }
        let mut identities = HashSet::new();
        for artifact in self
            .runs
            .iter()
            .chain(self.recipes.iter())
            .chain(self.model_manifests.iter())
            .chain(self.calibration_profiles.iter())
        {
            artifact.validate()?;
            let identity = (artifact.contract.as_str(), artifact.id.as_str());
            if !identities.insert(identity) {
                return Err(EvaluationError::DuplicateBundleArtifact(
                    artifact.id.clone(),
                ));
            }
        }
        Ok(())
    }
}

fn valid_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn require_text(field: &'static str, value: &str) -> Result<(), EvaluationError> {
    if value.trim().is_empty() {
        Err(EvaluationError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum EvaluationError {
    #[error("unsupported specialization evaluation contract {0:?}")]
    UnsupportedContract(String),
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("unsupported calibration method {0:?}")]
    UnsupportedCalibrationMethod(String),
    #[error("calibration temperature must be finite and > 0, got {0}")]
    InvalidTemperature(f64),
    #[error("ECE bin count must be greater than zero")]
    ZeroEceBins,
    #[error("fitted calibration requires a profile id")]
    MissingCalibrationProfile,
    #[error("candidate keys/logits must have the same length and at least two candidates")]
    CandidateShape,
    #[error("candidate keys must be unique")]
    DuplicateCandidateKey,
    #[error("target key {0:?} is not present in candidate keys")]
    MissingTargetKey(String),
    #[error("candidate logits must be finite")]
    NonFiniteLogit,
    #[error("observation weight must be finite and > 0, got {0}")]
    InvalidWeight(f64),
    #[error("latency must be finite and > 0, got {0}")]
    InvalidLatency(f64),
    #[error("peak memory must be greater than zero")]
    ZeroPeakMemory,
    #[error("specialization run contains no observations")]
    EmptyRun,
    #[error("duplicate evaluation observation {0:?}")]
    DuplicateObservation(String),
    #[error("run is missing controlled variant kind {0}")]
    MissingVariant(&'static str),
    #[error("family {family_id:?} must contain exactly one baseline, got {count}")]
    BaselineCount { family_id: String, count: usize },
    #[error("evaluation family {0:?} changes case/question/target semantics")]
    FamilySemanticDrift(String),
    #[error("evaluation family {0:?} changes its semantic candidate set")]
    FamilyCandidateDrift(String),
    #[error("missing baseline for family {0:?}")]
    MissingBaseline(String),
    #[error("calibration metric evaluation failed: {0}")]
    Calibration(String),
    #[error("timing summary failed: {0}")]
    Timing(String),
    #[error("aggregate latency is invalid")]
    InvalidAggregateLatency,
    #[error("specialization plan is invalid: {0}")]
    Plan(String),
    #[error("at least one external run is required")]
    NoRuns,
    #[error("run {run:?} targets plan {actual:?}, expected {expected:?}")]
    PlanIdMismatch {
        run: String,
        expected: String,
        actual: String,
    },
    #[error("run {run:?} targets benchmark {actual:?}, expected {expected:?}")]
    BenchmarkMismatch {
        run: String,
        expected: String,
        actual: String,
    },
    #[error("multiple runs were supplied for model {0:?}")]
    DuplicateModelRun(String),
    #[error("missing external run for model {0:?}")]
    MissingModelRun(String),
    #[error("generic and specialized model runs do not share the same prompt/tokenizer evaluation contract")]
    ModelEvaluationContractMismatch,
    #[error("generic and specialized model runs were measured in different backend/host/quantization/execution environments")]
    EnvironmentMismatch,
    #[error("generic and specialized runs do not contain the same observation ids")]
    ObservationSetMismatch,
    #[error("observation {0:?} changes semantic benchmark metadata across model runs")]
    ObservationSemanticMismatch(String),
    #[error("specialization report contains no comparisons")]
    EmptyReport,
    #[error("generic and specialized model ids must differ")]
    IdenticalModelPair,
    #[error("specialization report contains a non-finite metric")]
    NonFiniteReportMetric,
    #[error("specialization report has invalid observation counts")]
    InvalidReportCounts,
    #[error("specialization report has invalid throughput")]
    InvalidReportThroughput,
    #[error("specialization report contains an invalid robustness metric")]
    InvalidRobustnessMetric,
    #[error("artifact {0:?} does not contain a 64-character hexadecimal SHA-256")]
    InvalidSha256(String),
    #[error("specialization bundle requires at least two external runs")]
    TooFewBundleRuns,
    #[error("specialization bundle repeats artifact {0:?}")]
    DuplicateBundleArtifact(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbojev_specialized::{ModelPair, SPECIALIZATION_EVALUATION_CONTRACT};

    fn observation(
        id: &str,
        family_id: &str,
        variant: VariantKind,
        candidate_keys: &[&str],
        logits: &[f32],
        target_key: &str,
        latency_seconds: f64,
    ) -> EvaluationObservation {
        EvaluationObservation {
            id: id.into(),
            family_id: family_id.into(),
            case_id: family_id.into(),
            question_id: "route".into(),
            variant,
            candidate_keys: candidate_keys.iter().map(|value| (*value).into()).collect(),
            logits: logits.to_vec(),
            target_key: target_key.into(),
            weight: 1.0,
            latency_seconds,
            peak_memory_bytes: 1024,
        }
    }

    fn run(model: &str, specialized: bool) -> SpecializationRun {
        let good = if specialized { 4.0 } else { 1.0 };
        SpecializationRun {
            contract: SPECIALIZATION_RUN_CONTRACT.into(),
            id: format!("run-{model}"),
            plan_id: "eval-v1".into(),
            benchmark_suite: "bench-v1".into(),
            model: EvaluationModelIdentity {
                id: model.into(),
                revision: Some("rev".into()),
                tokenizer_revision: Some("tok".into()),
                prompt_ir: "turbojev.prompt-ir.v2".into(),
                renderer_profile: "renderer-v1".into(),
                calibration: EvaluationCalibration {
                    method: "temperature-softmax".into(),
                    temperature: 1.0,
                    fitted: false,
                    profile_id: None,
                    ece_bins: 5,
                },
            },
            environment: RunEnvironment {
                backend: "fixture-runtime".into(),
                host: "fixture-cpu".into(),
                quantization: Some("fixture-q".into()),
                execution_path: Some("reference".into()),
                runner_version: Some("0.20.0".into()),
            },
            observations: vec![
                observation(
                    "a-base",
                    "a",
                    VariantKind::Baseline,
                    &["x", "y"],
                    &[0.0, good],
                    "y",
                    0.02,
                ),
                observation(
                    "a-order",
                    "a",
                    VariantKind::CandidateOrder,
                    &["y", "x"],
                    &[good, 0.0],
                    "y",
                    0.02,
                ),
                observation(
                    "a-para",
                    "a",
                    VariantKind::Paraphrase,
                    &["x", "y"],
                    &[0.0, good],
                    "y",
                    0.02,
                ),
            ],
        }
    }

    #[test]
    fn run_metrics_are_semantic_and_finite() {
        let metrics = evaluate_run(&run("generic", false)).unwrap();
        assert_eq!(metrics.baseline_quality.accuracy, 1.0);
        assert_eq!(metrics.candidate_order.invariance, 1.0);
        assert_eq!(metrics.paraphrase.accuracy, 1.0);
        assert!(metrics.decisions_per_second > 0.0);
    }

    #[test]
    fn same_benchmark_pair_produces_deltas_without_a_winner_label() {
        let plan = SpecializationEvaluationPlan {
            contract: SPECIALIZATION_EVALUATION_CONTRACT.into(),
            id: "eval-v1".into(),
            benchmark_suite: "bench-v1".into(),
            pairs: vec![ModelPair {
                generic_model: "generic".into(),
                specialized_model: "specialized".into(),
            }],
            metrics: vec![
                "accuracy",
                "nll",
                "brier",
                "ece",
                "latency",
                "peak-memory",
                "decisions-per-second",
                "candidate-order-robustness",
                "paraphrase-robustness",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        };
        let report = evaluate_plan(
            "report-v1",
            &plan,
            &[run("generic", false), run("specialized", true)],
        )
        .unwrap();
        assert_eq!(report.comparisons.len(), 1);
        assert!(report.comparisons[0].delta.nll < 0.0);
    }

    #[test]
    fn semantic_drift_across_model_runs_is_rejected() {
        let generic = run("generic", false);
        let mut specialized = run("specialized", true);
        specialized.observations[0].target_key = "x".into();
        assert!(matches!(
            verify_pair_alignment(&generic, &specialized),
            Err(EvaluationError::ObservationSemanticMismatch(_))
        ));
    }

    #[test]
    fn bundle_requires_real_sha256_digests() {
        let digest = ArtifactDigest {
            contract: "x".into(),
            id: "x".into(),
            sha256: "a".repeat(64),
        };
        digest.validate().unwrap();
        let mut bad = digest;
        bad.sha256 = "xyz".into();
        assert!(matches!(
            bad.validate(),
            Err(EvaluationError::InvalidSha256(_))
        ));
    }
}
