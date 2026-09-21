#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable mathematical primitives for candidate-only output projection.
//!
//! This crate deliberately does not know how a model stores or executes its
//! output head. Device/runtime adapters remain responsible for obtaining the
//! final hidden state and the requested output rows. The functions here provide
//! a deterministic CPU reference used by conformance tests and runtimes that
//! already expose dense rows in `f32`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Dense candidate-row matrix in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseCandidateRows {
    hidden_size: usize,
    candidate_count: usize,
    values: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl DenseCandidateRows {
    /// Construct a dense row matrix. `values` must contain exactly
    /// `candidate_count * hidden_size` entries in candidate order.
    pub fn new(
        hidden_size: usize,
        candidate_count: usize,
        values: Vec<f32>,
    ) -> Result<Self, ProjectionError> {
        Self::with_bias(hidden_size, candidate_count, values, None)
    }

    /// Construct a dense row matrix with an optional per-candidate bias.
    pub fn with_bias(
        hidden_size: usize,
        candidate_count: usize,
        values: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, ProjectionError> {
        if hidden_size == 0 {
            return Err(ProjectionError::InvalidShape(
                "hidden_size must be greater than zero".to_owned(),
            ));
        }
        if candidate_count < 2 {
            return Err(ProjectionError::InvalidShape(
                "candidate_count must be at least two".to_owned(),
            ));
        }
        let expected = hidden_size
            .checked_mul(candidate_count)
            .ok_or_else(|| ProjectionError::InvalidShape("row matrix size overflow".to_owned()))?;
        if values.len() != expected {
            return Err(ProjectionError::InvalidShape(format!(
                "row matrix has {} values, expected {expected}",
                values.len()
            )));
        }
        if let Some(values) = &bias {
            if values.len() != candidate_count {
                return Err(ProjectionError::InvalidShape(format!(
                    "bias has {} values, expected {candidate_count}",
                    values.len()
                )));
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ProjectionError::NonFiniteInput("candidate rows"));
        }
        if bias
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(ProjectionError::NonFiniteInput("candidate bias"));
        }
        Ok(Self {
            hidden_size,
            candidate_count,
            values,
            bias,
        })
    }

    /// Width of each candidate row.
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of candidate rows.
    pub const fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Row-major dense values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Optional per-row bias.
    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Borrow one candidate row.
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        if index >= self.candidate_count {
            return None;
        }
        let start = index * self.hidden_size;
        Some(&self.values[start..start + self.hidden_size])
    }
}

/// Candidate-only projection configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectionConfig {
    /// Scalar applied after each row dot product and before the optional bias.
    /// Ordinary untied/tied language-model heads use `1.0`.
    pub output_scale: f32,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self { output_scale: 1.0 }
    }
}

/// Project a last-token hidden state onto only the supplied candidate rows.
///
/// The operation is mathematically equivalent to computing the full dense
/// vocabulary matrix multiplication and gathering these rows, provided the
/// runtime has supplied the exact same post-normalization hidden state, output
/// weights, optional bias and scale used by its ordinary LM head.
pub fn project_candidates_f32(
    hidden: &[f32],
    rows: &DenseCandidateRows,
    config: ProjectionConfig,
) -> Result<Vec<f32>, ProjectionError> {
    if hidden.len() != rows.hidden_size {
        return Err(ProjectionError::InvalidShape(format!(
            "hidden state has {} values, expected {}",
            hidden.len(),
            rows.hidden_size
        )));
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(ProjectionError::NonFiniteInput("hidden state"));
    }
    if !config.output_scale.is_finite() {
        return Err(ProjectionError::NonFiniteInput("output scale"));
    }

    let mut logits = Vec::with_capacity(rows.candidate_count);
    for row_index in 0..rows.candidate_count {
        let row = rows.row(row_index).expect("validated row index");
        let dot = dot_f32(hidden, row)?;
        let bias = rows
            .bias
            .as_ref()
            .map_or(0.0, |values| values[row_index]);
        let logit = dot * config.output_scale + bias;
        if !logit.is_finite() {
            return Err(ProjectionError::NonFiniteOutput { row: row_index });
        }
        logits.push(logit);
    }
    Ok(logits)
}

/// Full-vocabulary dense projection reference used only by conformance tests.
/// The matrix is row-major with shape `vocab_size x hidden_size`.
pub fn project_full_vocab_f32(
    hidden: &[f32],
    vocab_size: usize,
    matrix: &[f32],
    bias: Option<&[f32]>,
    config: ProjectionConfig,
) -> Result<Vec<f32>, ProjectionError> {
    if hidden.is_empty() || vocab_size == 0 {
        return Err(ProjectionError::InvalidShape(
            "hidden state and vocabulary must be non-empty".to_owned(),
        ));
    }
    let expected = hidden
        .len()
        .checked_mul(vocab_size)
        .ok_or_else(|| ProjectionError::InvalidShape("vocabulary matrix overflow".to_owned()))?;
    if matrix.len() != expected {
        return Err(ProjectionError::InvalidShape(format!(
            "vocabulary matrix has {} values, expected {expected}",
            matrix.len()
        )));
    }
    if let Some(values) = bias {
        if values.len() != vocab_size {
            return Err(ProjectionError::InvalidShape(format!(
                "vocabulary bias has {} values, expected {vocab_size}",
                values.len()
            )));
        }
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(ProjectionError::NonFiniteInput("vocabulary matrix"));
    }

    let rows = DenseCandidateRows::with_bias(
        hidden.len(),
        vocab_size,
        matrix.to_vec(),
        bias.map(ToOwned::to_owned),
    )?;
    project_candidates_f32(hidden, &rows, config)
}

/// Gather candidate rows from a row-major full-vocabulary matrix without
/// changing their requested order. Duplicate candidate IDs are legal and remain
/// duplicated in the output, matching ordinary gather semantics.
pub fn gather_candidate_rows_f32(
    vocab_size: usize,
    hidden_size: usize,
    matrix: &[f32],
    bias: Option<&[f32]>,
    candidate_ids: &[u32],
) -> Result<DenseCandidateRows, ProjectionError> {
    if hidden_size == 0 || vocab_size == 0 {
        return Err(ProjectionError::InvalidShape(
            "hidden_size and vocab_size must be non-zero".to_owned(),
        ));
    }
    if candidate_ids.len() < 2 {
        return Err(ProjectionError::InvalidShape(
            "at least two candidate IDs are required".to_owned(),
        ));
    }
    let expected = hidden_size
        .checked_mul(vocab_size)
        .ok_or_else(|| ProjectionError::InvalidShape("vocabulary matrix overflow".to_owned()))?;
    if matrix.len() != expected {
        return Err(ProjectionError::InvalidShape(format!(
            "vocabulary matrix has {} values, expected {expected}",
            matrix.len()
        )));
    }
    if let Some(values) = bias {
        if values.len() != vocab_size {
            return Err(ProjectionError::InvalidShape(format!(
                "vocabulary bias has {} values, expected {vocab_size}",
                values.len()
            )));
        }
    }

    let mut gathered = Vec::with_capacity(candidate_ids.len() * hidden_size);
    let mut gathered_bias = bias.map(|_| Vec::with_capacity(candidate_ids.len()));
    for &token_id in candidate_ids {
        let index = usize::try_from(token_id).map_err(|_| ProjectionError::TokenOutOfRange {
            token_id,
            vocab_size,
        })?;
        if index >= vocab_size {
            return Err(ProjectionError::TokenOutOfRange { token_id, vocab_size });
        }
        let start = index * hidden_size;
        gathered.extend_from_slice(&matrix[start..start + hidden_size]);
        if let (Some(source), Some(target)) = (bias, gathered_bias.as_mut()) {
            target.push(source[index]);
        }
    }
    DenseCandidateRows::with_bias(hidden_size, candidate_ids.len(), gathered, gathered_bias)
}

/// Exact/approximate parity result between full-vocabulary gather and selective
/// projection output.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectionParity {
    /// Maximum absolute logit error across candidates.
    pub max_abs_delta: f32,
    /// Candidate index where the maximum error occurred.
    pub worst_candidate: usize,
    /// Whether every candidate stayed within the supplied tolerance.
    pub passed: bool,
}

/// Compare two candidate-logit vectors with an absolute tolerance.
pub fn compare_logits(
    reference: &[f32],
    candidate: &[f32],
    tolerance: f32,
) -> Result<ProjectionParity, ProjectionError> {
    if reference.len() != candidate.len() || reference.len() < 2 {
        return Err(ProjectionError::InvalidShape(format!(
            "logit vectors have incompatible lengths: {} vs {}",
            reference.len(),
            candidate.len()
        )));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ProjectionError::InvalidTolerance(tolerance));
    }
    if reference.iter().chain(candidate).any(|value| !value.is_finite()) {
        return Err(ProjectionError::NonFiniteInput("logit vector"));
    }

    let mut max_abs_delta = 0.0f32;
    let mut worst_candidate = 0usize;
    for (index, (&left, &right)) in reference.iter().zip(candidate).enumerate() {
        let delta = (left - right).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            worst_candidate = index;
        }
    }
    Ok(ProjectionParity {
        max_abs_delta,
        worst_candidate,
        passed: max_abs_delta <= tolerance,
    })
}

fn dot_f32(left: &[f32], right: &[f32]) -> Result<f32, ProjectionError> {
    if left.len() != right.len() {
        return Err(ProjectionError::InvalidShape(format!(
            "dot operands have different lengths: {} vs {}",
            left.len(),
            right.len()
        )));
    }
    // Accumulate in f64 to make the portable reference less sensitive to host
    // vectorization/order. Runtime implementations may use device-native f32 or
    // mixed precision and validate against a non-zero tolerance.
    let value = left
        .iter()
        .zip(right)
        .fold(0.0f64, |sum, (&a, &b)| sum + f64::from(a) * f64::from(b));
    Ok(value as f32)
}

/// Projection/conformance error.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ProjectionError {
    /// Matrix/vector shape is invalid.
    #[error("invalid projection shape: {0}")]
    InvalidShape(String),
    /// A required numeric input contains NaN/Infinity.
    #[error("non-finite values in {0}")]
    NonFiniteInput(&'static str),
    /// A projected logit became non-finite.
    #[error("non-finite projected output at candidate row {row}")]
    NonFiniteOutput { row: usize },
    /// Candidate token lies outside the vocabulary.
    #[error("candidate token {token_id} is outside vocabulary size {vocab_size}")]
    TokenOutOfRange { token_id: u32, vocab_size: usize },
    /// Tolerance is negative or non-finite.
    #[error("invalid projection tolerance {0}")]
    InvalidTolerance(f32),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_matrix() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hidden = vec![0.5, -1.25, 2.0, 0.75];
        let matrix = vec![
            0.1, 0.2, 0.3, 0.4,
            -0.5, 0.6, -0.7, 0.8,
            1.0, 0.0, 0.25, -0.5,
            0.33, -0.2, 0.75, 1.1,
            -1.0, 0.5, 0.5, -0.25,
        ];
        let bias = vec![0.0, 0.1, -0.2, 0.3, -0.4];
        (hidden, matrix, bias)
    }

    #[test]
    fn candidate_rows_match_full_vocabulary_gather_exactly() {
        let (hidden, matrix, bias) = fixture_matrix();
        let full = project_full_vocab_f32(
            &hidden,
            5,
            &matrix,
            Some(&bias),
            ProjectionConfig::default(),
        )
        .unwrap();
        let ids = [4, 1, 3];
        let rows = gather_candidate_rows_f32(5, 4, &matrix, Some(&bias), &ids).unwrap();
        let selected = project_candidates_f32(&hidden, &rows, ProjectionConfig::default()).unwrap();
        let expected: Vec<f32> = ids.iter().map(|id| full[*id as usize]).collect();
        let parity = compare_logits(&expected, &selected, 0.0).unwrap();
        assert!(parity.passed);
    }

    #[test]
    fn candidate_order_and_duplicates_are_preserved() {
        let (hidden, matrix, bias) = fixture_matrix();
        let rows = gather_candidate_rows_f32(5, 4, &matrix, Some(&bias), &[2, 2, 0]).unwrap();
        let logits = project_candidates_f32(&hidden, &rows, ProjectionConfig::default()).unwrap();
        assert_eq!(logits[0], logits[1]);
        assert_ne!(logits[0], logits[2]);
    }

    #[test]
    fn output_scale_is_applied_before_bias() {
        let rows = DenseCandidateRows::with_bias(
            2,
            2,
            vec![1.0, 2.0, 3.0, 4.0],
            Some(vec![5.0, 7.0]),
        )
        .unwrap();
        let logits = project_candidates_f32(
            &[2.0, 1.0],
            &rows,
            ProjectionConfig { output_scale: 0.5 },
        )
        .unwrap();
        assert_eq!(logits, vec![7.0, 12.0]);
    }

    #[test]
    fn parity_reports_worst_candidate() {
        let report = compare_logits(&[1.0, 2.0, 3.0], &[1.0, 2.2, 2.95], 0.1).unwrap();
        assert_eq!(report.worst_candidate, 1);
        assert!((report.max_abs_delta - 0.2).abs() < 1e-6);
        assert!(!report.passed);
    }
}
