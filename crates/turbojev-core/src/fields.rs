//! Shared workflow state machine. Hosts only perform the requested inference.

use crate::{AsyncScoringBackend, Calibrator, ScoringBackend, TurboJevAsyncEngine, TurboJevEngine};
use indexmap::IndexMap;
use serde_json::{json, Value};
use turbojev_backend::{BackendError, FieldOutput, FieldTask};
use turbojev_schema::fields::{
    dependency_layers, permutation_count, Field, FieldError, GenerateRequest, GenerateResponse,
    Thinking,
};

/// The same validation, skip propagation and output typing serves every host.
pub struct Generation {
    request: GenerateRequest,
    fields: Vec<Field>,
    order: Vec<usize>,
    position: usize,
    orders: Vec<Vec<usize>>,
    rotation: usize,
    sums: Vec<f64>,
    resolved_thinking: Option<Thinking>,
    asking_effort: bool,
    result: IndexMap<String, Value>,
    response: GenerateResponse,
    input_known: bool,
}

impl Generation {
    pub fn new(request: GenerateRequest) -> Result<Self, FieldError> {
        let fields = request.compile()?;
        let order = dependency_layers(&fields)?.into_iter().flatten().collect();
        Ok(Self {
            request,
            fields,
            order,
            position: 0,
            orders: Vec::new(),
            rotation: 0,
            sums: Vec::new(),
            resolved_thinking: None,
            asking_effort: false,
            result: IndexMap::new(),
            input_known: false,
            response: GenerateResponse {
                result: IndexMap::new(),
                thinking: IndexMap::new(),
                thinking_effort: IndexMap::new(),
                skipped: Vec::new(),
                usage: Default::default(),
            },
        })
    }

    pub fn next_task(&mut self) -> Result<Option<FieldTask>, FieldError> {
        while self.position < self.order.len() {
            let field = &self.fields[self.order[self.position]];
            if field
                .depends_on
                .iter()
                .any(|d| self.response.skipped.contains(d))
                || !field.conditions_met(&self.result)
            {
                self.response.skipped.push(field.name.clone());
                self.position += 1;
                continue;
            }
            let mut state = self.request.context.clone();
            if !field.depends_on.is_empty() {
                let answers: IndexMap<_, _> = field
                    .depends_on
                    .iter()
                    .map(|n| (n.clone(), self.result[n].clone()))
                    .collect();
                // Keep multimodal evidence at the envelope's root; dependency text
                // is appended to its existing textual context rather than losing media.
                if state.get("contract").and_then(Value::as_str) == Some("turbojev.media.v1") {
                    let text = state.get("text").and_then(Value::as_str).unwrap_or("");
                    state["text"] = json!(format!("{text}\nPrevious answers: {}", json!(answers)));
                } else {
                    state = json!({"context": state, "previous_answers": answers});
                }
            }
            let policy = self.resolved_thinking.unwrap_or(field.thinking);
            if policy == Thinking::Auto {
                self.asking_effort = true;
                return Ok(Some(FieldTask {
                    name: field.name.clone(), state,
                    instructions: format!("How much reasoning does this question require? none: directly stated or obvious; low: one simple inference; medium: several steps; high: difficult multi-step reasoning. Question: {}. Output type: {}", field.instructions, field.output_schema()),
                    output_schema: json!({"type": "string", "enum": ["none", "low", "medium", "high"]}),
                    values: Some(vec![json!("none"), json!("low"), json!("medium"), json!("high")]),
                    thinking: false, thinking_budget: 0, max_tokens: field.max_tokens,
                }));
            }
            if self.orders.is_empty() {
                self.orders = if let Some(values) = &field.values {
                    orderings(values, &field.permutations, self.request.seed)?
                } else {
                    vec![Vec::new()]
                };
                self.sums = vec![0.0; field.values.as_ref().map_or(0, Vec::len)];
            }
            return Ok(Some(FieldTask {
                name: field.name.clone(),
                state,
                instructions: if field.nullable && field.values.is_none() {
                    format!("{} Return literal JSON null, without quotes, when the value is absent; do not use the string \"null\" to indicate absence.", field.instructions)
                } else {
                    field.instructions.clone()
                },
                output_schema: field.output_schema(),
                values: field.values.as_ref().map(|values| {
                    self.orders[self.rotation]
                        .iter()
                        .map(|&i| values[i].clone())
                        .collect()
                }),
                thinking: matches!(policy, Thinking::Enabled(_)),
                thinking_budget: match policy {
                    Thinking::Enabled(n) => n,
                    _ => 0,
                },
                max_tokens: field.max_tokens,
            }));
        }
        Ok(None)
    }

    pub fn accept<C: Calibrator>(
        &mut self,
        output: FieldOutput,
        calibrator: &C,
    ) -> Result<(), BackendError> {
        let field = &self.fields[self.order[self.position]];
        self.response.usage.generated_tokens += output.generated_tokens;
        if let Some(n) = output.input_tokens {
            self.input_known = true;
            *self.response.usage.input_tokens.get_or_insert(0) += n;
        }
        if !output.thinking.is_empty() {
            let text = self
                .response
                .thinking
                .entry(field.name.clone())
                .or_default();
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&output.thinking);
        }
        if self.asking_effort {
            let probabilities = calibrator
                .probabilities(&output.logits)
                .map_err(|e| BackendError::Inference(e.to_string()))?;
            if probabilities.len() != 4 || output.value.is_some() {
                return Err(BackendError::Inference(
                    "invalid thinking effort scores".into(),
                ));
            }
            let winner = argmax(&probabilities);
            self.resolved_thinking = Some(
                [
                    Thinking::Disabled,
                    Thinking::Enabled(512),
                    Thinking::Enabled(2048),
                    Thinking::Enabled(4096),
                ][winner],
            );
            self.response.thinking_effort.insert(
                field.name.clone(),
                ["none", "low", "medium", "high"][winner].to_owned(),
            );
            self.asking_effort = false;
            return Ok(());
        }
        let value = if let Some(values) = &field.values {
            if output.value.is_some() || output.logits.len() != values.len() {
                return Err(BackendError::Inference(
                    "finite field returned an invalid score shape".into(),
                ));
            }
            let probabilities = if values.len() == 1 {
                vec![1.0]
            } else {
                calibrator
                    .probabilities(&output.logits)
                    .map_err(|e| BackendError::Inference(e.to_string()))?
            };
            for (position, &original) in self.orders[self.rotation].iter().enumerate() {
                self.sums[original] += probabilities[position];
            }
            self.rotation += 1;
            if self.rotation < self.orders.len() {
                return Ok(());
            }
            for probability in &mut self.sums {
                *probability /= self.orders.len() as f64;
            }
            let winner = argmax(&self.sums);
            let value = values[winner].clone();
            self.result.insert(field.name.clone(), value.clone());
            if field.return_probabilities {
                let distribution: IndexMap<_, _> = values
                    .iter()
                    .zip(&self.sums)
                    .map(|(v, p)| {
                        (
                            if let Some(s) = v.as_str() {
                                if field.nullable && s == "null" {
                                    v.to_string()
                                } else {
                                    s.to_owned()
                                }
                            } else {
                                v.to_string()
                            },
                            *p,
                        )
                    })
                    .collect();
                let typed_distribution: Vec<_> = values
                    .iter()
                    .zip(&self.sums)
                    .map(|(value, probability)| json!({"value":value,"probability":probability}))
                    .collect();
                json!({"value": value, "confidence": self.sums[winner], "probabilities": distribution, "distribution":typed_distribution})
            } else {
                value
            }
        } else {
            let value = output
                .value
                .ok_or_else(|| BackendError::Inference("generator returned no value".into()))?;
            if !output.logits.is_empty() || !field.accepts(&value) {
                return Err(BackendError::Inference(
                    "generated value violates field schema".into(),
                ));
            }
            self.result.insert(field.name.clone(), value.clone());
            value
        };
        self.response.result.insert(field.name.clone(), value);
        self.response.usage.decisions += 1;
        self.position += 1;
        self.rotation = 0;
        self.orders.clear();
        self.sums.clear();
        self.resolved_thinking = None;
        Ok(())
    }

    pub fn finish(mut self) -> GenerateResponse {
        // Restore caller order rather than exposing topological scheduling order.
        self.response.result = self
            .fields
            .iter()
            .filter_map(|f| {
                self.response
                    .result
                    .shift_remove(&f.name)
                    .map(|v| (f.name.clone(), v))
            })
            .collect();
        self.response.skipped = self
            .fields
            .iter()
            .filter(|f| self.response.skipped.contains(&f.name))
            .map(|f| f.name.clone())
            .collect();
        if !self.input_known {
            self.response.usage.input_tokens = None;
        }
        self.response
    }
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (i, &p)| {
            if p > best.1 {
                (i, p)
            } else {
                best
            }
        })
        .0
}

/// Williams balanced design; even counts balance positions and neighbours,
/// odd counts additionally include reversed rows. Auto starts in canonical order.
pub fn orderings(
    values: &[Value],
    budget: &Value,
    seed: u64,
) -> Result<Vec<Vec<usize>>, FieldError> {
    let n = values.len();
    let count = permutation_count(budget, n)?;
    if budget == "auto" {
        let mut canonical: Vec<_> = (0..n).collect();
        canonical.sort_by_key(|&i| values[i].to_string());
        let first: Vec<_> = (0..n)
            .map(|i| if i % 2 == 0 { i / 2 } else { n - (i + 1) / 2 })
            .collect();
        let mut rows = Vec::new();
        for offset in 0..n {
            let row: Vec<_> = first.iter().map(|i| canonical[(i + offset) % n]).collect();
            rows.push(row.clone());
            if n % 2 != 0 && n > 1 {
                rows.push(row.into_iter().rev().collect());
            }
        }
        return Ok(rows);
    }
    let total = (1..=n).fold(1usize, |a, b| a.saturating_mul(b));
    if count == total {
        let mut rows = Vec::new();
        let mut row: Vec<_> = (0..n).collect();
        rows.push(row.clone());
        while next_permutation(&mut row) {
            rows.push(row.clone());
        }
        return Ok(rows);
    }
    let mut rng = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut rows = vec![(0..n).collect::<Vec<_>>()];
    while rows.len() < count {
        let mut row: Vec<_> = (0..n).collect();
        for i in (1..n).rev() {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let j = (rng.wrapping_mul(0x2545f4914f6cdd1d) % (i as u64 + 1)) as usize;
            row.swap(i, j);
        }
        if !rows.contains(&row) {
            rows.push(row);
        }
    }
    Ok(rows)
}

fn next_permutation(row: &mut [usize]) -> bool {
    let Some(i) = (0..row.len().saturating_sub(1))
        .rev()
        .find(|&i| row[i] < row[i + 1])
    else {
        return false;
    };
    let j = (i + 1..row.len()).rev().find(|&j| row[j] > row[i]).unwrap();
    row.swap(i, j);
    row[i + 1..].reverse();
    true
}

impl<B: ScoringBackend, C: Calibrator> TurboJevEngine<B, C> {
    /// Generate JSON-typed fields without changing the legacy evaluate contract.
    pub fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, BackendError> {
        let mut generation =
            Generation::new(request).map_err(|e| BackendError::InvalidInput(e.to_string()))?;
        while let Some(task) = generation
            .next_task()
            .map_err(|e| BackendError::InvalidInput(e.to_string()))?
        {
            let output = if task.values.as_ref().is_some_and(|values| values.len() == 1) {
                FieldOutput {
                    value: None,
                    logits: vec![0.0],
                    thinking: String::new(),
                    input_tokens: Some(0),
                    generated_tokens: 0,
                }
            } else {
                self.backend.infer_field(&task)?
            };
            generation.accept(output, &self.calibrator)?;
        }
        Ok(generation.finish())
    }
}

impl<B: AsyncScoringBackend, C: Calibrator> TurboJevAsyncEngine<B, C> {
    pub async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, BackendError> {
        let mut generation =
            Generation::new(request).map_err(|e| BackendError::InvalidInput(e.to_string()))?;
        while let Some(task) = generation
            .next_task()
            .map_err(|e| BackendError::InvalidInput(e.to_string()))?
        {
            let output = if task.values.as_ref().is_some_and(|values| values.len() == 1) {
                FieldOutput {
                    value: None,
                    logits: vec![0.0],
                    thinking: String::new(),
                    input_tokens: Some(0),
                    generated_tokens: 0,
                }
            } else {
                self.backend.infer_field_async(&task).await?
            };
            generation.accept(output, &self.calibrator)?;
        }
        Ok(generation.finish())
    }
}
