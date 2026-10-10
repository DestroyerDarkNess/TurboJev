//! Shared workflow state machine. Hosts only perform the requested inference.

use crate::{AsyncScoringBackend, Calibrator, ScoringBackend, TurboJevAsyncEngine, TurboJevEngine};
#[path = "fields/numeric.rs"]
mod numeric;

use indexmap::IndexMap;
use serde_json::{json, Value};
use turbojev_backend::{BackendError, FieldOutput, FieldTask};
use turbojev_schema::fields::{
    dependency_layers, permutation_count, ArrayField, Field, FieldError, GenerateRequest,
    GenerateResponse, Thinking, MAX_ARRAY_ITEMS, MAX_CONTROL_VALUES,
};

struct ArrayProgress {
    specification: ArrayField,
    items: Vec<Value>,
    limit: usize,
    child: Option<Box<Generation>>,
    asking_continue: bool,
    approved: bool,
    nullable_end: bool,
    duplicate_retries: usize,
}

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
    confidences: IndexMap<String, f64>,
    rng: u64,
    array: Option<ArrayProgress>,
}

impl Generation {
    pub fn new(mut request: GenerateRequest) -> Result<Self, FieldError> {
        let (mode, temperature) = request.decoding()?;
        request.mode = Some(mode);
        request.temperature = Some(temperature);
        request.context = request.effective_context()?;
        request.images.clear();
        let fields = request.compile()?;
        Self::from_fields(request, fields)
    }

    fn from_fields(request: GenerateRequest, fields: Vec<Field>) -> Result<Self, FieldError> {
        let order = dependency_layers(&fields)?.into_iter().flatten().collect();
        let rng = request.seed.wrapping_add(0x9e3779b97f4a7c15).max(1);
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
            confidences: IndexMap::new(),
            rng,
            array: None,
            response: GenerateResponse {
                result: IndexMap::new(),
                thinking: IndexMap::new(),
                thinking_effort: IndexMap::new(),
                skipped: Vec::new(),
                usage: Default::default(),
                prompts: Vec::new(),
            },
        })
    }

    pub fn next_task(&mut self) -> Result<Option<FieldTask>, FieldError> {
        while self.position < self.order.len() {
            let field = &self.fields[self.order[self.position]];
            if field.dependency_groups.iter().any(|group| {
                group
                    .iter()
                    .all(|name| self.response.skipped.contains(name))
            }) || !field.conditions_with_confidence(&self.result, &self.confidences)
            {
                self.response.skipped.push(field.name.clone());
                self.position += 1;
                continue;
            }
            let state = dependency_state(&self.request.context, field, &self.fields, &self.result);
            if field.array.is_some() {
                if let Some(task) = self.next_array_task(state)? {
                    return Ok(Some(task));
                }
                continue;
            }
            let policy = self.resolved_thinking.unwrap_or(field.thinking);
            if policy == Thinking::Auto {
                self.asking_effort = true;
                return Ok(Some(FieldTask {
                    name: field.name.clone(), state,
                    instructions: format!("How much reasoning does this question require? none: directly stated or obvious; low: one simple inference; medium: several steps; high: difficult multi-step reasoning. Question: {}. Output type: {}", field.instructions, field.output_schema()),
                    output_schema: json!({"type": "string", "enum": ["none", "low", "medium", "high"]}),
                    grammar: None,
                    capture_prompt: self.request.print_final_prompt,
                    values: Some(vec![json!("none"), json!("low"), json!("medium"), json!("high")]),
                    thinking: false, thinking_budget: 0, max_tokens: field.max_tokens,
                    temperature: self.request.sampling_temperature(), seed: self.request.seed,
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
                    described_question(field)
                },
                output_schema: field.output_schema(),
                capture_prompt: self.request.print_final_prompt,
                grammar: numeric::grammar(&field.output_schema(), self.request.numeric_max_digits),
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
                temperature: self.request.sampling_temperature(),
                seed: self.request.seed,
            }));
        }
        Ok(None)
    }

    pub fn accept<C: Calibrator>(
        &mut self,
        output: FieldOutput,
        calibrator: &C,
    ) -> Result<(), BackendError> {
        if self.array.is_some() {
            return self.accept_array(output, calibrator);
        }
        let field = &self.fields[self.order[self.position]];
        if let Some(prompt) = &output.prompt {
            self.response.prompts.push(prompt.clone());
        }
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
            let probabilities =
                tempered_probabilities(&probabilities, self.request.temperature.unwrap_or(0.0));
            let winner = if self.request.samples() {
                sample(&probabilities, &mut self.rng)
            } else {
                argmax(&probabilities)
            };
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
            let probabilities =
                tempered_probabilities(&probabilities, self.request.temperature.unwrap_or(0.0));
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
            let tie_keys: Vec<_> = values.iter().map(Value::to_string).collect();
            let best = (0..values.len())
                .reduce(|best, i| {
                    if self.sums[i] > self.sums[best]
                        || (self.sums[i] == self.sums[best] && tie_keys[i] < tie_keys[best])
                    {
                        i
                    } else {
                        best
                    }
                })
                .expect("validated finite fields are nonempty");
            let winner = if self.request.samples() && field.levels.is_empty() {
                sample(&self.sums, &mut self.rng)
            } else {
                best
            };
            let confidence = decision_confidence(&self.sums, &field.levels);
            self.confidences.insert(field.name.clone(), confidence);
            let value = if field.levels.is_empty() {
                values[winner].clone()
            } else {
                json!(self
                    .sums
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| index as f64 * probability)
                    .sum::<f64>())
            };
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
                json!({"value": value, "confidence": confidence, "probability": self.sums[winner], "probabilities": distribution, "distribution":typed_distribution})
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
            numeric::validate(
                &value,
                &field.output_schema(),
                self.request.numeric_max_digits,
            )?;
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

    pub fn usage(&self) -> turbojev_schema::Usage {
        let mut usage = self.response.usage.clone();
        if let Some(child) = self.array.as_ref().and_then(|a| a.child.as_ref()) {
            let child_usage = child.usage();
            usage.generated_tokens += child_usage.generated_tokens;
            if let Some(count) = child_usage.input_tokens {
                *usage.input_tokens.get_or_insert(0) += count;
            }
        }
        usage
    }

    pub fn finish(mut self) -> GenerateResponse {
        // Restore caller order rather than exposing topological scheduling order.
        let mut nested = json!({});
        for field in &self.fields {
            if let Some(value) = self.response.result.shift_remove(&field.name) {
                put_value(&mut nested, &field.path, value);
            }
        }
        self.response.result = serde_json::from_value(nested).expect("result is an object");
        let mut skipped = Vec::new();
        for field in &self.fields {
            if self.response.skipped.contains(&field.name) {
                let owner = &field.path[0];
                let name = if !self.response.result.contains_key(owner) {
                    owner.clone()
                } else {
                    field.name.clone()
                };
                if !skipped.contains(&name) {
                    skipped.push(name);
                }
            }
        }
        self.response.skipped = skipped;
        if !self.input_known {
            self.response.usage.input_tokens = None;
        }
        self.response
    }

    fn next_array_task(&mut self, state: Value) -> Result<Option<FieldTask>, FieldError> {
        let field = self.fields[self.order[self.position]].clone();
        if self.array.is_none() {
            let specification = field.array.clone().expect("array field");
            let items = specification.continue_from.clone();
            let limit = specification
                .max_items
                .unwrap_or(items.len() + MAX_ARRAY_ITEMS);
            self.array = Some(ArrayProgress {
                specification,
                items,
                limit,
                child: None,
                asking_continue: false,
                approved: false,
                nullable_end: false,
                duplicate_retries: 0,
            });
        }
        loop {
            let progress = self.array.as_mut().expect("active array");
            if let Some(child) = &mut progress.child {
                if let Some(task) = child.next_task()? {
                    return Ok(Some(task));
                }
                let response = progress.child.take().unwrap().finish();
                self.response.prompts.extend(response.prompts);
                let value = response
                    .result
                    .get("item")
                    .cloned()
                    .ok_or_else(|| FieldError("array item was skipped".into()))?;
                let label = format!("{}[{}]", field.name, progress.items.len());
                for text in response.thinking.into_values() {
                    self.response.thinking.insert(label.clone(), text);
                }
                for effort in response.thinking_effort.into_values() {
                    self.response.thinking_effort.insert(label.clone(), effort);
                }
                self.response.usage.generated_tokens += response.usage.generated_tokens;
                if let Some(tokens) = response.usage.input_tokens {
                    self.input_known = true;
                    *self.response.usage.input_tokens.get_or_insert(0) += tokens;
                }
                numeric::validate(
                    &value,
                    &progress.specification.item_schema(),
                    self.request.numeric_max_digits,
                )
                .map_err(|e| FieldError(e.to_string()))?;
                self.commit_array_item(value)
                    .map_err(|error| FieldError(error.to_string()))?;
                if self.array.is_none() {
                    return Ok(None);
                }
                continue;
            }
            if progress.items.len() >= progress.limit {
                self.finish_array();
                return Ok(None);
            }
            let may_end = progress.items.len() >= progress.specification.min_items;
            let scalar = &progress.specification.items[0];
            progress.nullable_end = !progress.specification.item_object
                && !scalar.nullable
                && scalar
                    .values
                    .as_ref()
                    .is_none_or(|values| values.len() < MAX_CONTROL_VALUES);
            let array_state = append_evidence(
                &state,
                &format!(
                    "Array: {}\nInstructions: {}\nItem schema: {}\nCurrent array (JSON): {}",
                    json!(field.name),
                    field.instructions,
                    progress.specification.item_schema(),
                    json!(progress.items),
                ),
            );
            if may_end && !progress.nullable_end && !progress.approved {
                progress.asking_continue = true;
                return Ok(Some(FieldTask {
                    name: format!("{}.append_item", field.name),
                    state: array_state,
                    instructions: "Should a new item be appended to the current array?".into(),
                    output_schema: json!({"type": "boolean"}),
                    grammar: None,
                    capture_prompt: self.request.print_final_prompt,
                    values: Some(vec![json!(true), json!(false)]),
                    thinking: false,
                    thinking_budget: 0,
                    max_tokens: self.request.numeric_max_tokens,
                    temperature: self.request.sampling_temperature(),
                    seed: self.request.seed,
                }));
            }
            if progress.specification.item_object {
                let instructions: Vec<_> = progress
                    .specification
                    .items
                    .iter()
                    .map(|item| format!("{}: {}", item.path.join("."), described_question(item)))
                    .collect();
                return Ok(Some(FieldTask {
                    name: format!("{}[{}]", field.name, progress.items.len()),
                    state: array_state,
                    instructions: format!(
                        "Give the next distinct object of the array above. {}",
                        instructions.join("\n")
                    ),
                    output_schema: progress.specification.item_schema(),
                    capture_prompt: self.request.print_final_prompt,
                    grammar: numeric::grammar(
                        &progress.specification.item_schema(),
                        self.request.numeric_max_digits,
                    ),
                    values: None,
                    thinking: false,
                    thinking_budget: 0,
                    max_tokens: self.request.text_max_tokens,
                    temperature: self.request.sampling_temperature(),
                    seed: self
                        .request
                        .seed
                        .wrapping_add(progress.items.len() as u64)
                        .wrapping_add(progress.duplicate_retries as u64),
                }));
            }
            let mut item = scalar.clone();
            item.name = "item".into();
            item.path = vec!["item".into()];
            item.instructions = format!(
                "Give the next distinct item of the array above. {} {}",
                item.instructions,
                if may_end && progress.nullable_end {
                    "Answer null if the array is complete."
                } else {
                    ""
                },
            );
            if may_end && progress.nullable_end {
                item.nullable = true;
                if let Some(values) = &mut item.values {
                    values.push(Value::Null);
                }
            }
            let mut request = self.request.clone();
            request.context = array_state;
            request.seed = self
                .request
                .seed
                .wrapping_add(progress.items.len() as u64)
                .wrapping_add(progress.duplicate_retries as u64);
            progress.child = Some(Box::new(Self::from_fields(request, vec![item])?));
        }
    }

    fn accept_array<C: Calibrator>(
        &mut self,
        output: FieldOutput,
        calibrator: &C,
    ) -> Result<(), BackendError> {
        let progress = self.array.as_mut().expect("active array");
        if let Some(child) = &mut progress.child {
            return child.accept(output, calibrator);
        }
        if let Some(prompt) = &output.prompt {
            self.response.prompts.push(prompt.clone());
        }
        self.response.usage.generated_tokens += output.generated_tokens;
        if let Some(tokens) = output.input_tokens {
            self.input_known = true;
            *self.response.usage.input_tokens.get_or_insert(0) += tokens;
        }
        if progress.asking_continue {
            if output.value.is_some() || output.logits.len() != 2 {
                return Err(BackendError::Inference(
                    "invalid array continuation scores".into(),
                ));
            }
            let probabilities = calibrator
                .probabilities(&output.logits)
                .map_err(|error| BackendError::Inference(error.to_string()))?;
            let selected = if self.request.samples() {
                sample(
                    &tempered_probabilities(
                        &probabilities,
                        self.request.temperature.unwrap_or(0.0),
                    ),
                    &mut self.rng,
                )
            } else {
                argmax(&probabilities)
            };
            progress.asking_continue = false;
            if selected == 1 {
                self.finish_array();
            } else {
                progress.approved = true;
            }
            return Ok(());
        }
        if !output.logits.is_empty() || !output.thinking.is_empty() {
            return Err(BackendError::Inference(
                "invalid structured item output".into(),
            ));
        }
        let value = output
            .value
            .ok_or_else(|| BackendError::Inference("missing array item".into()))?;
        numeric::validate(
            &value,
            &progress.specification.item_schema(),
            self.request.numeric_max_digits,
        )?;
        self.commit_array_item(value)
    }

    fn commit_array_item(&mut self, value: Value) -> Result<(), BackendError> {
        let progress = self.array.as_mut().expect("active array");
        let ends = progress.nullable_end
            && value.is_null()
            && progress.items.len() >= progress.specification.min_items;
        let duplicate = progress
            .items
            .iter()
            .any(|item| turbojev_schema::fields::same_value(item, &value));
        if duplicate && self.request.samples() && progress.duplicate_retries < 2 {
            progress.duplicate_retries += 1;
            progress.approved = false;
            return Ok(());
        }
        if ends || duplicate {
            if progress.items.len() < progress.specification.min_items {
                return Err(BackendError::Inference(
                    "repeated item before minItems was reached".into(),
                ));
            }
            self.finish_array();
            return Ok(());
        }
        let valid = progress.specification.accepts_item(&value)
            && (!progress.specification.item_object
                || progress
                    .specification
                    .items
                    .iter()
                    .all(|field| get_value(&value, &field.path).is_some()));
        if !valid {
            return Err(BackendError::Inference(
                "array item violates its schema".into(),
            ));
        }
        progress.items.push(value);
        progress.approved = false;
        Ok(())
    }

    fn finish_array(&mut self) {
        let progress = self.array.take().expect("active array");
        let name = self.fields[self.order[self.position]].name.clone();
        let value = json!(progress.items);
        self.result.insert(name.clone(), value.clone());
        self.response.result.insert(name, value);
        self.response.usage.decisions += 1;
        self.position += 1;
    }
}

fn described_question(field: &Field) -> String {
    if field.descriptions.is_empty() {
        return field.instructions.clone();
    }
    let descriptions: Vec<_> = field
        .descriptions
        .iter()
        .map(|(value, description)| format!("{value}: {description}"))
        .collect();
    format!(
        "{}\nOption descriptions:\n{}",
        field.instructions,
        descriptions.join("\n")
    )
}

fn append_evidence(state: &Value, text: &str) -> Value {
    if state.get("contract").and_then(Value::as_str) == Some("turbojev.media.v1") {
        let mut state = state.clone();
        state["text"] = json!(format!(
            "{}\n{text}",
            state.get("text").and_then(Value::as_str).unwrap_or("")
        ));
        state
    } else if let Some(context) = state.as_str() {
        json!(format!("{context}\n\n{text}"))
    } else {
        json!({"context": state, "workflow": text})
    }
}

fn dependency_state(
    context: &Value,
    field: &Field,
    fields: &[Field],
    results: &IndexMap<String, Value>,
) -> Value {
    if field.depends_on.is_empty() {
        return context.clone();
    }
    let mut answers = json!({});
    let mut visible = std::collections::HashSet::new();
    let mut pending: Vec<_> = field.depends_on.iter().map(String::as_str).collect();
    while let Some(name) = pending.pop() {
        if visible.insert(name) {
            let parent = fields
                .iter()
                .find(|field| field.name == name)
                .expect("checked dependency");
            pending.extend(parent.depends_on.iter().map(String::as_str));
        }
    }
    // Preserve schema order and expose ancestors, without leaking unrelated branches.
    for parent in fields
        .iter()
        .filter(|parent| visible.contains(parent.name.as_str()))
    {
        if let Some(value) = results.get(&parent.name) {
            put_value(&mut answers, &parent.path, value.clone());
        }
    }
    if context.get("contract").and_then(Value::as_str) == Some("turbojev.media.v1") {
        append_evidence(context, &format!("Previous answers: {answers}"))
    } else {
        json!({"context": context, "previous_answers": answers})
    }
}

fn put_value(object: &mut Value, path: &[String], value: Value) {
    if path.len() == 1 {
        object[&path[0]] = value;
    } else {
        if object.get(&path[0]).is_none() {
            object[&path[0]] = json!({});
        }
        put_value(&mut object[&path[0]], &path[1..], value);
    }
}

fn get_value<'a>(object: &'a Value, path: &[String]) -> Option<&'a Value> {
    path.iter().try_fold(object, |at, name| at.get(name))
}

/// A dispersion measure, not empirical calibration or a success guarantee.
fn decision_confidence(probabilities: &[f64], levels: &[String]) -> f64 {
    let count = probabilities.len();
    if count == 1 {
        return 1.0;
    }
    let top = argmax(probabilities);
    if levels.is_empty() {
        ((probabilities[top] - 1.0 / count as f64) / (1.0 - 1.0 / count as f64)).clamp(0.0, 1.0)
    } else {
        let spread: f64 = probabilities
            .iter()
            .enumerate()
            .map(|(index, probability)| probability * index.abs_diff(top) as f64)
            .sum();
        let middle = (count - 1) as f64 / 2.0;
        let even: f64 = (0..count)
            .map(|index| (index as f64 - middle).abs())
            .sum::<f64>()
            / count as f64;
        (1.0 - spread / even).clamp(0.0, 1.0)
    }
}

fn tempered_probabilities(probabilities: &[f64], temperature: f64) -> Vec<f64> {
    if temperature == 0.0 || temperature == 1.0 {
        return probabilities.to_vec();
    }
    let pivot = probabilities.iter().copied().fold(0.0, f64::max).ln();
    let mut weights: Vec<_> = probabilities
        .iter()
        .map(|probability| ((probability.ln() - pivot) / temperature).exp())
        .collect();
    let mass: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= mass;
    }
    weights
}

fn sample(probabilities: &[f64], rng: &mut u64) -> usize {
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    let mut target = (rng.wrapping_mul(0x2545f4914f6cdd1d) >> 11) as f64 / ((1u64 << 53) as f64);
    for (index, probability) in probabilities.iter().enumerate() {
        target -= probability;
        if target < 0.0 {
            return index;
        }
    }
    probabilities.len() - 1
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
    let mut rng = seed.wrapping_add(0x9e3779b97f4a7c15).max(1);
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
        self.generate_controlled(request, || Ok(()))
    }

    /// Hosts own clocks and cancellation. Checks run before and after each model operation.
    pub fn generate_controlled(
        &self,
        request: GenerateRequest,
        mut check: impl FnMut() -> Result<(), BackendError>,
    ) -> Result<GenerateResponse, BackendError> {
        check()?;
        let mut generation =
            Generation::new(request).map_err(|e| BackendError::InvalidInput(e.to_string()))?;
        let outcome = (|| -> Result<(), BackendError> {
            while let Some(task) = generation
                .next_task()
                .map_err(|e| BackendError::InvalidInput(e.to_string()))?
            {
                check()?;
                let output = if task.values.as_ref().is_some_and(|values| values.len() == 1) {
                    FieldOutput {
                        prompt: None,
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
                check()?;
            }
            check()?;
            Ok(())
        })();
        outcome.map_err(|error| error.with_usage(generation.usage()))?;
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
        let outcome = async {
            while let Some(task) = generation
                .next_task()
                .map_err(|e| BackendError::InvalidInput(e.to_string()))?
            {
                let output = if task.values.as_ref().is_some_and(|values| values.len() == 1) {
                    FieldOutput {
                        prompt: None,
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
            Ok::<(), BackendError>(())
        }
        .await;
        outcome.map_err(|error| error.with_usage(generation.usage()))?;
        Ok(generation.finish())
    }
}
