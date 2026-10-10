//! Portable JSON Schema subset for dependency-aware typed outputs.
//!
//! Inspired by TypeLLM's Apache-2.0 schema and workflow contracts. No runtime,
//! tokenizer, filesystem, or model inference is performed by this module.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;

/// Maximum enum and permutation workload accepted before inference.
pub const MAX_ENUM_VALUES: usize = 512;
/// Single-token control labels; larger sets require sequence/value scoring.
pub const MAX_CONTROL_VALUES: usize = 26;

#[path = "structured.rs"]
mod structured;
pub use structured::{ArrayField, MAX_ARRAY_ITEMS};
/// Hard cap avoids accidentally requesting factorial model work.
pub const MAX_PERMUTATIONS: usize = 720;

/// Request accepted by the additive `generate` API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateRequest {
    /// Text, structured state, or the existing multimodal envelope.
    #[serde(default, alias = "state")]
    pub context: Value,
    /// Inline base64 or data-URI images. Source loading belongs to the host.
    #[serde(default)]
    pub images: Vec<String>,
    /// Ordered field-name to field-schema map.
    #[serde(default)]
    pub questions: Option<IndexMap<String, Value>>,
    /// Alternative JSON Schema object with properties and required.
    #[serde(default)]
    pub schema: Option<Value>,
    /// Default reasoning selection; false, true, or "auto".
    #[serde(default)]
    pub thinking: Value,
    /// Optional default reasoning effort: none, low, medium, high.
    #[serde(default)]
    pub thinking_effort: Option<String>,
    /// Optional default reasoning token budget.
    #[serde(default)]
    pub thinking_budget: Option<usize>,
    /// Maximum generated tokens for a text field.
    #[serde(default = "default_text_tokens")]
    pub text_max_tokens: usize,
    /// Maximum generated tokens for a numeric field.
    #[serde(default = "default_numeric_tokens")]
    pub numeric_max_tokens: usize,
    /// Reproducible permutation sampling seed.
    #[serde(default)]
    pub seed: u64,
    /// Zero chooses the most likely answer; positive values sample decisions.
    #[serde(default)]
    pub temperature: f64,
}

fn default_text_tokens() -> usize {
    256
}
fn default_numeric_tokens() -> usize {
    64
}

/// JSON scalar type retained in the returned value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    String,
    Integer,
    Number,
    Boolean,
}

/// Bounded reasoning policy; auto is a real model decision, not a host heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking {
    Disabled,
    Enabled(usize),
    Auto,
}

/// Checked field independent of any tokenizer or model.
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub path: Vec<String>,
    pub dependency_groups: Vec<Vec<String>>,
    pub gates: Vec<(String, Value)>,
    pub descriptions: Vec<(Value, String)>,
    pub levels: Vec<String>,
    pub array: Option<ArrayField>,
    pub kind: FieldType,
    pub nullable: bool,
    pub instructions: String,
    pub values: Option<Vec<Value>>,
    pub depends_on: Vec<String>,
    pub when: IndexMap<String, Value>,
    pub permutations: Value,
    pub return_probabilities: bool,
    pub thinking: Thinking,
    pub max_tokens: usize,
}

/// Results retain actual JSON types and distinguish skipped fields from null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub result: IndexMap<String, Value>,
    pub thinking: IndexMap<String, String>,
    pub thinking_effort: IndexMap<String, String>,
    pub skipped: Vec<String>,
    pub usage: super::Usage,
}

/// A schema or workflow error discovered before any model execution.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid field schema: {0}")]
pub struct FieldError(pub String);

fn fail(message: impl Into<String>) -> FieldError {
    FieldError(message.into())
}

/// Numeric equality follows JSON semantics; booleans never equal integers.
pub fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            number_order(a, b) == Some(std::cmp::Ordering::Equal)
        }
        _ => a == b,
    }
}

fn number_order(a: &serde_json::Number, b: &serde_json::Number) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    let integer = |n: &serde_json::Number| {
        n.as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
    };
    match (integer(a), integer(b)) {
        (Some(a), Some(b)) => Some(a.cmp(&b)),
        (Some(a), None) => {
            let b = b.as_f64()?;
            if b <= i128::MIN as f64 {
                return Some(Ordering::Greater);
            }
            if b >= i128::MAX as f64 {
                return Some(Ordering::Less);
            }
            let order = a.cmp(&(b as i128));
            if order == Ordering::Equal {
                0.0_f64.partial_cmp(&b.fract())
            } else {
                Some(order)
            }
        }
        (None, Some(_)) => number_order(b, a).map(Ordering::reverse),
        (None, None) => a.as_f64()?.partial_cmp(&b.as_f64()?),
    }
}

fn effort(value: &str) -> Result<Thinking, FieldError> {
    match value {
        "none" => Ok(Thinking::Disabled),
        "low" => Ok(Thinking::Enabled(512)),
        "medium" => Ok(Thinking::Enabled(2048)),
        "high" => Ok(Thinking::Enabled(4096)),
        _ => Err(fail("thinking_effort must be none, low, medium, or high")),
    }
}

fn thinking(
    value: &Value,
    level: Option<&str>,
    budget: Option<usize>,
) -> Result<Thinking, FieldError> {
    if budget.is_some_and(|n| n == 0 || n > 32768) {
        return Err(fail("thinking_budget must be between 1 and 32768"));
    }
    let selected = match value {
        Value::Null | Value::Bool(false) => Thinking::Disabled,
        Value::Bool(true) => Thinking::Enabled(budget.unwrap_or(512)),
        Value::String(s) if s == "auto" => {
            if level.is_some() || budget.is_some() {
                return Err(fail("auto thinking cannot specify an effort or budget"));
            }
            Thinking::Auto
        }
        _ => return Err(fail("thinking must be a boolean or auto")),
    };
    if let Some(level) = level {
        if budget.is_some() {
            return Err(fail(
                "thinking_effort and thinking_budget cannot both be set",
            ));
        }
        let requested = effort(level)?;
        if !value.is_null() && (selected == Thinking::Disabled) != (requested == Thinking::Disabled)
        {
            return Err(fail("thinking contradicts thinking_effort"));
        }
        return Ok(requested);
    }
    Ok(selected)
}

impl GenerateRequest {
    pub fn effective_context(&self) -> Result<Value, FieldError> {
        if self.images.is_empty() {
            return Ok(self.context.clone());
        }
        if self.context.get("contract").is_some() {
            return Err(fail("images cannot be combined with a media envelope"));
        }
        let text = self.context.as_str().map(str::to_owned).unwrap_or_else(|| {
            if self.context.is_null() {
                String::new()
            } else {
                self.context.to_string()
            }
        });
        let images: Vec<_> = self
            .images
            .iter()
            .map(|image| {
                if image.starts_with("data:image/") {
                    image
                        .split_once(";base64,")
                        .map(|(_, bytes)| bytes.to_owned())
                        .ok_or_else(|| fail("image data URI must use base64"))
                } else if image.contains("://") {
                    Err(fail("image URLs must be loaded and encoded by the host"))
                } else {
                    Ok(image.clone())
                }
            })
            .collect::<Result<_, _>>()?;
        let context = json!({"contract": "turbojev.media.v1", "text": text, "modality": "image", "images": images});
        super::media::MediaEvidence::from_state(&context).map_err(fail)?;
        Ok(context)
    }

    /// Compile and validate the complete request before tokenization/inference.
    pub fn compile(&self) -> Result<Vec<Field>, FieldError> {
        super::validate_state(&self.effective_context()?).map_err(|e| fail(e.to_string()))?;
        if !(1..=32768).contains(&self.text_max_tokens)
            || !(1..=32768).contains(&self.numeric_max_tokens)
        {
            return Err(fail("generation token limits must be between 1 and 32768"));
        }
        let default_thinking = thinking(
            &self.thinking,
            self.thinking_effort.as_deref(),
            self.thinking_budget,
        )?;
        let properties = match (&self.questions, &self.schema) {
            (Some(fields), None) => fields.clone(),
            (None, Some(schema)) => {
                let schema = schema
                    .as_object()
                    .ok_or_else(|| fail("schema must be an object"))?;
                if schema.get("type") != Some(&json!("object")) {
                    return Err(fail("root schema type must be object"));
                }
                for key in schema.keys() {
                    if ![
                        "type",
                        "properties",
                        "required",
                        "additionalProperties",
                        "description",
                        "$schema",
                    ]
                    .contains(&key.as_str())
                    {
                        return Err(fail(format!("unsupported root schema keyword {key}")));
                    }
                }
                if schema
                    .get("additionalProperties")
                    .is_some_and(|v| v != &Value::Bool(false))
                {
                    return Err(fail("additionalProperties must be false when specified"));
                }
                let props: IndexMap<String, Value> = serde_json::from_value(
                    schema
                        .get("properties")
                        .cloned()
                        .ok_or_else(|| fail("properties required"))?,
                )
                .map_err(|_| fail("properties must be an object"))?;
                if let Some(required) = schema.get("required") {
                    let required = required
                        .as_array()
                        .ok_or_else(|| fail("required must be a list"))?;
                    let mut seen = HashSet::new();
                    for name in required {
                        let name = name
                            .as_str()
                            .ok_or_else(|| fail("required names must be strings"))?;
                        if !props.contains_key(name) || !seen.insert(name) {
                            return Err(fail("unknown or duplicate required field"));
                        }
                    }
                }
                props
            }
            _ => return Err(fail("provide exactly one of questions or schema")),
        };
        if properties.is_empty() || properties.len() > super::DEFAULT_MAX_QUESTIONS {
            return Err(fail("field count exceeds the supported range"));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(fail("temperature must be finite and nonnegative"));
        }
        let mut fields = Vec::new();
        for expanded in structured::expand(self, &properties)? {
            let name = expanded.name;
            let raw = expanded.raw;
            if name.trim().is_empty() {
                return Err(fail("field names must not be empty"));
            }
            let raw = raw
                .as_object()
                .ok_or_else(|| fail(format!("field {name} must be an object")))?;
            for key in raw.keys() {
                if ![
                    "type",
                    "enum",
                    "choices",
                    "levels",
                    "instructions",
                    "description",
                    "depends_on",
                    "when",
                    "permutations",
                    "return_probabilities",
                    "thinking",
                    "thinking_effort",
                    "thinking_budget",
                ]
                .contains(&key.as_str())
                {
                    return Err(fail(format!("unsupported keyword {key} on {name}")));
                }
            }
            let declared = raw
                .get("type")
                .ok_or_else(|| fail(format!("type required on {name}")))?;
            let (kind, nullable) = if let Some(kinds) = declared.as_array() {
                if kinds.len() != 2 || kinds.iter().filter(|v| *v == "null").count() != 1 {
                    return Err(fail("nullable type must be [type, null]"));
                }
                (kinds.iter().find(|v| *v != "null").unwrap(), true)
            } else {
                (declared, false)
            };
            let kind = match kind.as_str() {
                Some("string") => FieldType::String,
                Some("integer") => FieldType::Integer,
                Some("number") => FieldType::Number,
                Some("boolean") => FieldType::Boolean,
                _ => return Err(fail(format!("unsupported type on {name}"))),
            };
            for key in ["instructions", "description"] {
                if raw.get(key).is_some_and(|v| !v.is_string()) {
                    return Err(fail(format!("{key} must be a string")));
                }
            }
            let instructions = raw
                .get("instructions")
                .or_else(|| raw.get("description"))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Choose the value for {name:?}."));
            let mut field = Field {
                name,
                path: expanded.path,
                dependency_groups: expanded.groups,
                gates: expanded.gates,
                descriptions: Vec::new(),
                levels: Vec::new(),
                array: expanded.array,
                kind,
                nullable,
                instructions,
                values: None,
                depends_on: Vec::new(),
                when: IndexMap::new(),
                permutations: raw.get("permutations").cloned().unwrap_or(json!(1)),
                return_probabilities: raw
                    .get("return_probabilities")
                    .map(|v| {
                        v.as_bool()
                            .ok_or_else(|| fail("return_probabilities must be boolean"))
                    })
                    .transpose()?
                    .unwrap_or(false),
                thinking: default_thinking,
                max_tokens: if kind == FieldType::String {
                    self.text_max_tokens
                } else {
                    self.numeric_max_tokens
                },
            };
            let described = if let Some(choices) = raw.get("choices") {
                if raw.contains_key("enum") || raw.contains_key("levels") {
                    return Err(fail("use exactly one of enum, choices or levels"));
                }
                Some(parse_choices(choices, false, &mut field.descriptions)?)
            } else if let Some(levels) = raw.get("levels") {
                if kind != FieldType::Number
                    || nullable
                    || raw.contains_key("enum")
                    || raw.contains_key("permutations")
                {
                    return Err(fail(
                        "levels require a nonnullable number without enum or permutations",
                    ));
                }
                let values = parse_choices(levels, true, &mut field.descriptions)?;
                if !(2..=MAX_CONTROL_VALUES).contains(&values.as_array().unwrap().len()) {
                    return Err(fail("levels must contain 2 to 26 ordered labels"));
                }
                field.levels = values
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_str().unwrap().to_owned())
                    .collect();
                Some(json!(values))
            } else {
                None
            };
            if let Some(values) = raw.get("enum").or(described.as_ref()) {
                let values = values
                    .as_array()
                    .ok_or_else(|| fail("enum must be a list"))?;
                if values.is_empty() || values.len() > MAX_ENUM_VALUES {
                    return Err(fail("enum must contain 1 to 512 values"));
                }
                for (i, value) in values.iter().enumerate() {
                    if (field.levels.is_empty() && !field.accepts_type(value))
                        || values[..i].iter().any(|v| same_value(v, value))
                    {
                        return Err(fail("enum has a duplicate or incorrectly typed value"));
                    }
                }
                field.values = Some(values.clone());
            } else if kind == FieldType::Boolean {
                let mut values = vec![json!(true), json!(false)];
                if nullable {
                    values.push(Value::Null);
                }
                field.values = Some(values);
            }
            if raw.contains_key("return_probabilities") && field.values.is_none() {
                return Err(fail("probabilities require a finite enum or boolean"));
            }
            if raw.contains_key("permutations") && field.values.is_none() {
                return Err(fail("permutations require a finite enum or boolean"));
            }
            if field.return_probabilities
                && !raw.contains_key("permutations")
                && field.levels.is_empty()
            {
                field.permutations = json!("auto");
            }
            if let Some(values) = &field.values {
                if values.len() > MAX_CONTROL_VALUES && field.permutations == "auto" {
                    field.permutations = json!(1);
                }
                permutation_count(&field.permutations, values.len())?;
            }
            if let Some(deps) = raw.get("depends_on") {
                field.depends_on = serde_json::from_value(deps.clone())
                    .map_err(|_| fail("depends_on must be a list of strings"))?;
                let mut seen = HashSet::new();
                if field
                    .depends_on
                    .iter()
                    .any(|n| n.trim().is_empty() || !seen.insert(n))
                {
                    return Err(fail("empty or duplicate dependency"));
                }
            }
            if let Some(when) = raw.get("when") {
                field.when = serde_json::from_value(when.clone())
                    .map_err(|_| fail("when must be an object"))?;
                if field.when.is_empty() {
                    return Err(fail("when must not be empty"));
                }
                for dependency in field.when.keys() {
                    if !field.depends_on.contains(dependency) {
                        field.depends_on.push(dependency.clone());
                    }
                }
            }
            if ["thinking", "thinking_effort", "thinking_budget"]
                .iter()
                .any(|key| raw.contains_key(*key))
            {
                let mut budget = raw
                    .get("thinking_budget")
                    .map(|v| {
                        v.as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .ok_or_else(|| fail("thinking_budget must be a positive integer"))
                    })
                    .transpose()?;
                let level = raw
                    .get("thinking_effort")
                    .map(|v| {
                        v.as_str()
                            .ok_or_else(|| fail("thinking_effort must be a string"))
                    })
                    .transpose()?;
                let inherited = match default_thinking {
                    Thinking::Disabled => Value::Bool(false),
                    Thinking::Enabled(_) => Value::Bool(true),
                    Thinking::Auto => json!("auto"),
                };
                let selected =
                    raw.get("thinking")
                        .filter(|v| !v.is_null())
                        .unwrap_or(if level.is_some() {
                            &Value::Null
                        } else {
                            &inherited
                        });
                if selected == &Value::Bool(true) && level.is_none() && budget.is_none() {
                    if let Thinking::Enabled(n) = default_thinking {
                        budget = Some(n);
                    }
                }
                field.thinking = thinking(selected, level, budget)?;
            }
            for group in &field.dependency_groups {
                for dependency in group {
                    if !field.depends_on.contains(dependency) {
                        field.depends_on.push(dependency.clone());
                    }
                }
            }
            field.when = field.gates.iter().cloned().collect();
            fields.push(field);
        }
        for field in &fields {
            for dependency in &field.depends_on {
                if dependency == &field.name || !fields.iter().any(|f| &f.name == dependency) {
                    return Err(fail("unknown or self dependency"));
                }
            }
            for (parent, test) in &field.gates {
                validate_condition(fields.iter().find(|f| &f.name == parent).unwrap(), test)?;
            }
        }
        dependency_layers(&fields)?;
        if fields.len() > super::DEFAULT_MAX_QUESTIONS {
            return Err(fail("expanded field count exceeds 256"));
        }
        Ok(fields)
    }
}

/// Compile ordinary JSON Schema without loading a model or touching a tokenizer.
pub fn compile_json_schema(schema: Value) -> Result<Vec<Field>, FieldError> {
    let request: GenerateRequest = serde_json::from_value(json!({
        "context": "Schema compilation", "schema": schema,
    }))
    .map_err(|error| fail(error.to_string()))?;
    request.compile()
}

/// Language-neutral introspection of the compiled decisions and array workflows.
pub fn compile_schema_json(input: &str) -> Result<String, FieldError> {
    let schema = serde_json::from_str(input).map_err(|error| fail(error.to_string()))?;
    let fields = compile_json_schema(schema)?;
    let descriptions: Vec<_> = fields.iter().map(Field::describe).collect();
    serde_json::to_string(&descriptions).map_err(|error| fail(error.to_string()))
}

impl Field {
    pub fn describe(&self) -> Value {
        let (thinking, budget) = match self.thinking {
            Thinking::Disabled => (json!(false), Value::Null),
            Thinking::Enabled(budget) => (json!(true), json!(budget)),
            Thinking::Auto => (json!("auto"), Value::Null),
        };
        json!({
            "name": self.name, "path": self.path, "question": self.instructions,
            "output_schema": self.output_schema(), "choices": self.values,
            "descriptions": self.descriptions, "levels": self.levels,
            "permutations": self.permutations, "return_probabilities": self.return_probabilities,
            "depends_on": self.depends_on, "dependency_groups": self.dependency_groups,
            "when": self.gates, "thinking": thinking, "thinking_budget": budget,
            "array": self.array.as_ref().map(|array| json!({
                "items": array.items.iter().map(Field::describe).collect::<Vec<_>>(),
                "item_object": array.item_object, "minItems": array.min_items,
                "maxItems": array.max_items, "continue_from": array.continue_from,
            })),
        })
    }
}

impl Field {
    pub fn accepts_type(&self, value: &Value) -> bool {
        if value.is_null() {
            return self.nullable;
        }
        match self.kind {
            FieldType::String => value.is_string(),
            FieldType::Boolean => value.is_boolean(),
            FieldType::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
            FieldType::Number => value.is_number(),
        }
    }
    pub fn accepts(&self, value: &Value) -> bool {
        self.accepts_type(value)
            && self
                .values
                .as_ref()
                .is_none_or(|values| values.iter().any(|v| same_value(v, value)))
    }
    pub fn conditions_met(&self, answers: &IndexMap<String, Value>) -> bool {
        self.conditions_with_confidence(answers, &IndexMap::new())
    }
    pub fn conditions_with_confidence(
        &self,
        answers: &IndexMap<String, Value>,
        confidences: &IndexMap<String, f64>,
    ) -> bool {
        self.gates.iter().all(|(name, test)| {
            let Some(answer) = answers.get(name) else {
                return false;
            };
            if let Some(operators) = test.as_object() {
                operators.iter().all(|(operator, operand)| {
                    if operator == "confidence" {
                        confidences
                            .get(name)
                            .is_some_and(|confidence| condition_passes(&json!(confidence), operand))
                    } else {
                        condition_passes(answer, &json!({operator: operand}))
                    }
                })
            } else {
                condition_passes(answer, test)
            }
        })
    }
    pub fn output_schema(&self) -> Value {
        if let Some(array) = &self.array {
            return array.item_schema();
        }
        let kind = serde_json::to_value(self.kind).expect("scalar field type serializes");
        let mut schema = json!({"type": if self.nullable { json!([kind, "null"]) } else { kind }});
        if let Some(values) = &self.values {
            if !self.levels.is_empty() {
                schema["type"] = json!("string");
            }
            schema["enum"] = json!(values);
        }
        schema
    }
}

fn validate_condition(source: &Field, test: &Value) -> Result<(), FieldError> {
    if source.array.is_some() {
        return Err(fail("conditions may only test scalar fields"));
    }
    if let Some(operators) = test.as_object() {
        if operators.is_empty() {
            return Err(fail("empty condition"));
        }
        for (operator, operand) in operators {
            match operator.as_str() {
                "confidence" => {
                    if !source.return_probabilities {
                        return Err(fail("confidence conditions require return_probabilities"));
                    }
                    let bounds = operand
                        .as_object()
                        .filter(|object| !object.is_empty())
                        .ok_or_else(|| fail("confidence needs comparison bounds"))?;
                    for (comparison, bound) in bounds {
                        if !["gt", "gte", "lt", "lte"].contains(&comparison.as_str())
                            || !bound
                                .as_f64()
                                .is_some_and(|number| (0.0..=1.0).contains(&number))
                        {
                            return Err(fail(
                                "confidence bounds must use numeric comparisons between 0 and 1",
                            ));
                        }
                    }
                }
                "gt" | "gte" | "lt" | "lte" => {
                    if !matches!(source.kind, FieldType::Integer | FieldType::Number)
                        || !operand.is_number()
                    {
                        return Err(fail(
                            "numeric comparisons require a numeric field and operand",
                        ));
                    }
                }
                "in" | "not_in" => {
                    let values = operand
                        .as_array()
                        .ok_or_else(|| fail("membership condition requires a list"))?;
                    if values.is_empty()
                        || values.iter().any(|v| !condition_value_allowed(source, v))
                    {
                        return Err(fail("condition has an impossible value"));
                    }
                }
                "ne" => {
                    if !condition_value_allowed(source, operand) {
                        return Err(fail("condition has an impossible value"));
                    }
                }
                _ => return Err(fail("unknown condition operator")),
            }
        }
    } else if let Some(values) = test.as_array() {
        if values.is_empty() || values.iter().any(|v| !condition_value_allowed(source, v)) {
            return Err(fail("condition has an impossible value"));
        }
    } else if !condition_value_allowed(source, test) {
        return Err(fail("condition has an impossible value"));
    }
    Ok(())
}

fn condition_value_allowed(source: &Field, value: &Value) -> bool {
    source.levels.is_empty()
        && source.accepts(value)
        && (source.values.is_some() || source.kind != FieldType::String || value.is_null())
}

fn condition_passes(answer: &Value, test: &Value) -> bool {
    if let Some(ops) = test.as_object() {
        ops.iter().all(|(op, operand)| match op.as_str() {
            "in" => operand
                .as_array()
                .unwrap()
                .iter()
                .any(|v| same_value(answer, v)),
            "not_in" => !operand
                .as_array()
                .unwrap()
                .iter()
                .any(|v| same_value(answer, v)),
            "ne" => !same_value(answer, operand),
            _ => match (answer.as_number(), operand.as_number()) {
                (Some(a), Some(b)) => matches!(
                    (op.as_str(), number_order(a, b)),
                    ("gt", Some(std::cmp::Ordering::Greater))
                        | ("lt", Some(std::cmp::Ordering::Less))
                        | (
                            "gte",
                            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                        )
                        | (
                            "lte",
                            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                        )
                ),
                _ => false,
            },
        })
    } else if let Some(values) = test.as_array() {
        values.iter().any(|v| same_value(answer, v))
    } else {
        same_value(answer, test)
    }
}

/// Stable layers; forward references are allowed, cycles fail before inference.
pub fn dependency_layers(fields: &[Field]) -> Result<Vec<Vec<usize>>, FieldError> {
    let mut done = HashSet::new();
    let mut layers = Vec::new();
    while done.len() < fields.len() {
        let layer: Vec<_> = fields
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                !done.contains(f.name.as_str())
                    && f.depends_on.iter().all(|n| done.contains(n.as_str()))
            })
            .map(|(i, _)| i)
            .collect();
        if layer.is_empty() {
            return Err(fail("dependency cycle"));
        }
        for &i in &layer {
            done.insert(fields[i].name.as_str());
        }
        layers.push(layer);
    }
    Ok(layers)
}

pub fn permutation_count(budget: &Value, count: usize) -> Result<usize, FieldError> {
    let total = (1..=count).fold(1usize, |a, b| a.saturating_mul(b));
    let requested = match budget {
        Value::String(s) if s == "auto" => {
            if count == 1 || count % 2 == 0 {
                count
            } else {
                count * 2
            }
        }
        Value::String(s) if s == "all" => total,
        Value::Number(n) => n
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| fail("permutations must be auto, all, or a positive integer"))?
            .min(total),
        _ => {
            return Err(fail(
                "permutations must be auto, all, or a positive integer",
            ))
        }
    };
    if requested > MAX_PERMUTATIONS {
        return Err(fail("permutation budget exceeds 720"));
    }
    Ok(requested)
}

fn parse_choices(
    raw: &Value,
    levels: bool,
    descriptions: &mut Vec<(Value, String)>,
) -> Result<Value, FieldError> {
    let entries = raw
        .as_array()
        .filter(|entries| !entries.is_empty())
        .ok_or_else(|| fail("choices or levels must be a nonempty list"))?;
    let value_key = if levels { "label" } else { "value" };
    let mut values = Vec::new();
    for entry in entries {
        let entry = entry
            .as_object()
            .ok_or_else(|| fail("each choice or level must be an object"))?;
        if entry
            .keys()
            .any(|key| key != value_key && key != "description")
        {
            return Err(fail("unknown choice or level keyword"));
        }
        let mut value = entry
            .get(value_key)
            .cloned()
            .ok_or_else(|| fail("choice value or level label is missing"))?;
        if levels {
            value = json!(value
                .as_str()
                .filter(|label| !label.trim().is_empty())
                .ok_or_else(|| fail("level label must be a nonempty string"))?
                .trim());
        }
        if let Some(description) = entry.get("description") {
            let description = description
                .as_str()
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| fail("description must be a nonempty string"))?;
            descriptions.push((value.clone(), description.trim().to_owned()));
        }
        values.push(value);
    }
    Ok(json!(values))
}
