//! Structured field expansion and validation, inspired by TypeLLM 0.6.9.
//! Objects are compiled into scalar decisions; arrays remain bounded workflows.

use super::{fail, Field, FieldError, GenerateRequest};
use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use std::collections::HashSet;

pub const MAX_ARRAY_ITEMS: usize = 50;
const MAX_DEPTH: usize = 16;

/// Validated array specification. The cap counts newly generated items when no
/// explicit maximum is given, so continuation can read longer documents.
#[derive(Debug, Clone)]
pub struct ArrayField {
    pub items: Vec<Field>,
    pub item_object: bool,
    pub min_items: usize,
    pub max_items: Option<usize>,
    pub continue_from: Vec<Value>,
}

impl ArrayField {
    pub fn item_schema(&self) -> Value {
        if !self.item_object {
            return self.items[0].output_schema();
        }
        let mut schema = json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false});
        for field in &self.items {
            put_schema(&mut schema, &field.path, field.output_schema());
        }
        schema
    }

    pub fn accepts_item(&self, value: &Value) -> bool {
        if !self.item_object {
            return self.items[0].accepts(value);
        }
        accepts_object(value, &self.items, &[])
    }
}

fn accepts_object(value: &Value, fields: &[Field], prefix: &[String]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.iter().all(|(key, value)| {
        let mut path = prefix.to_vec();
        path.push(key.clone());
        if let Some(field) = fields.iter().find(|field| field.path == path) {
            field.accepts(value)
        } else if fields.iter().any(|field| field.path.starts_with(&path)) {
            accepts_object(value, fields, &path)
        } else {
            false
        }
    })
}

fn put_schema(schema: &mut Value, path: &[String], value: Value) {
    let key = &path[0];
    let required = schema["required"].as_array_mut().expect("object schema");
    if !required.contains(&json!(key)) {
        required.push(json!(key));
    }
    if path.len() == 1 {
        schema["properties"][key] = value;
    } else {
        if schema["properties"].get(key).is_none() {
            schema["properties"][key] = json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false});
        }
        put_schema(&mut schema["properties"][key], &path[1..], value);
    }
}

#[derive(Clone)]
struct GateScope {
    raw: Map<String, Value>,
    prefix: Vec<String>,
}

pub(super) struct ExpandedField {
    pub name: String,
    pub path: Vec<String>,
    pub raw: Value,
    pub groups: Vec<Vec<String>>,
    pub gates: Vec<(String, Value)>,
    pub array: Option<ArrayField>,
}

struct Entry {
    path: Vec<String>,
    raw: Map<String, Value>,
    scopes: Vec<GateScope>,
    headers: Vec<String>,
}

pub(super) fn expand(
    request: &GenerateRequest,
    properties: &IndexMap<String, Value>,
) -> Result<Vec<ExpandedField>, FieldError> {
    let mut entries = Vec::new();
    collect(properties, &[], &[], &[], &mut entries)?;
    let mut names = HashSet::new();
    for entry in &entries {
        if !names.insert(entry.path.join(".")) {
            return Err(fail("a property path clashes with another field name"));
        }
    }
    let paths: Vec<_> = entries.iter().map(|entry| entry.path.clone()).collect();
    entries
        .into_iter()
        .map(|entry| link(request, entry, &paths))
        .collect()
}

fn collect(
    properties: &IndexMap<String, Value>,
    prefix: &[String],
    scopes: &[GateScope],
    headers: &[String],
    entries: &mut Vec<Entry>,
) -> Result<(), FieldError> {
    if prefix.len() > MAX_DEPTH {
        return Err(fail("object nesting exceeds 16 levels"));
    }
    for (name, value) in properties {
        if name.trim().is_empty() {
            return Err(fail("field names must not be empty"));
        }
        let raw = value
            .as_object()
            .ok_or_else(|| fail("field must be an object"))?;
        let mut path = prefix.to_vec();
        path.push(name.clone());
        let mut own_scopes = scopes.to_vec();
        own_scopes.push(GateScope {
            raw: raw.clone(),
            prefix: prefix.to_vec(),
        });
        if raw.get("type") == Some(&json!("object")) {
            check_container(raw, false)?;
            let children = object_properties(raw)?;
            let mut own_headers = headers.to_vec();
            own_headers.push(format!("Object: {}", json!(path.join("."))));
            if let Some(instructions) = instructions(raw)? {
                own_headers.push(format!("Object instructions: {instructions}"));
            }
            collect(&children, &path, &own_scopes, &own_headers, entries)?;
        } else {
            if raw.get("type") == Some(&json!("array")) && !prefix.is_empty() {
                return Err(fail(
                    "arrays inside objects are not supported; place the array at the root",
                ));
            }
            entries.push(Entry {
                path,
                raw: raw.clone(),
                scopes: own_scopes,
                headers: headers.to_vec(),
            });
        }
    }
    Ok(())
}

fn instructions(raw: &Map<String, Value>) -> Result<Option<&str>, FieldError> {
    for key in ["instructions", "description"] {
        if raw.get(key).is_some_and(|value| !value.is_string()) {
            return Err(fail(format!("{key} must be a string")));
        }
    }
    Ok(raw
        .get("instructions")
        .or_else(|| raw.get("description"))
        .and_then(Value::as_str))
}

fn check_container(raw: &Map<String, Value>, array: bool) -> Result<(), FieldError> {
    let common = ["type", "instructions", "description", "depends_on", "when"];
    let keys = if array {
        &["items", "minItems", "maxItems", "continue_from"][..]
    } else {
        &["properties", "required", "additionalProperties"][..]
    };
    for key in raw.keys() {
        if !common.contains(&key.as_str()) && !keys.contains(&key.as_str()) {
            return Err(fail(format!(
                "unsupported container keyword {key}; use it on a property or item"
            )));
        }
    }
    instructions(raw)?;
    if raw
        .get("additionalProperties")
        .is_some_and(|value| value != &json!(false))
    {
        return Err(fail("additionalProperties must be false"));
    }
    Ok(())
}

fn object_properties(raw: &Map<String, Value>) -> Result<IndexMap<String, Value>, FieldError> {
    let properties: IndexMap<String, Value> = serde_json::from_value(
        raw.get("properties")
            .cloned()
            .ok_or_else(|| fail("object needs properties"))?,
    )
    .map_err(|_| fail("properties must be an object"))?;
    if properties.is_empty() {
        return Err(fail("object properties must not be empty"));
    }
    if let Some(required) = raw.get("required") {
        let required: Vec<String> = serde_json::from_value(required.clone())
            .map_err(|_| fail("required must be a list of names"))?;
        let mut seen = HashSet::new();
        if required
            .iter()
            .any(|name| !properties.contains_key(name) || !seen.insert(name))
        {
            return Err(fail("unknown or duplicate required property"));
        }
    }
    Ok(properties)
}

fn resolve(
    name: &str,
    prefix: &[String],
    paths: &[Vec<String>],
) -> Result<Vec<String>, FieldError> {
    let mut target = prefix.to_vec();
    target.push(name.to_owned());
    let selected: Vec<_> = paths
        .iter()
        .filter(|path| path.starts_with(&target))
        .map(|path| path.join("."))
        .collect();
    if selected.is_empty() {
        return Err(fail(format!("unknown dependency {name}")));
    }
    Ok(selected)
}

fn link(
    request: &GenerateRequest,
    entry: Entry,
    paths: &[Vec<String>],
) -> Result<ExpandedField, FieldError> {
    let mut groups = Vec::new();
    let mut gates = Vec::new();
    for scope in &entry.scopes {
        if let Some(dependencies) = scope.raw.get("depends_on") {
            let names: Vec<String> = serde_json::from_value(dependencies.clone())
                .map_err(|_| fail("depends_on must be a list of names"))?;
            let mut seen = HashSet::new();
            for name in names {
                if name.trim().is_empty() || !seen.insert(name.clone()) {
                    return Err(fail("empty or duplicate dependency"));
                }
                let group = resolve(&name, &scope.prefix, paths)?;
                if !groups.contains(&group) {
                    groups.push(group);
                }
            }
        }
        if let Some(when) = scope.raw.get("when") {
            let when = when
                .as_object()
                .filter(|object| !object.is_empty())
                .ok_or_else(|| fail("when must be a nonempty object"))?;
            for (name, test) in when {
                let target = resolve(name, &scope.prefix, paths)?;
                let scalar = paths.iter().any(|path| {
                    let mut at = scope.prefix.clone();
                    at.push(name.clone());
                    path == &at
                });
                if !scalar || target.len() != 1 {
                    return Err(fail("conditions may only test scalar fields"));
                }
                gates.push((target[0].clone(), test.clone()));
                if !groups.contains(&target) {
                    groups.push(target);
                }
            }
        }
    }
    let array = if entry.raw.get("type") == Some(&json!("array")) {
        Some(compile_array(request, &entry.raw)?)
    } else {
        None
    };
    let mut raw = if array.is_some() {
        json!({"type": "string"}).as_object().unwrap().clone()
    } else {
        entry.raw.clone()
    };
    raw.remove("depends_on");
    raw.remove("when");
    let own_question = instructions(&entry.raw)?
        .map(str::to_owned)
        .unwrap_or_else(|| format!("Choose the value for {:?}.", entry.path.last().unwrap()));
    let mut lines = entry.headers;
    lines.push(own_question);
    raw.insert("instructions".into(), json!(lines.join("\n")));
    Ok(ExpandedField {
        name: entry.path.join("."),
        path: entry.path,
        raw: Value::Object(raw),
        groups,
        gates,
        array,
    })
}

fn compile_array(
    request: &GenerateRequest,
    raw: &Map<String, Value>,
) -> Result<ArrayField, FieldError> {
    check_container(raw, true)?;
    let item = raw.get("items").ok_or_else(|| fail("array needs items"))?;
    let item_object = item.get("type") == Some(&json!("object"));
    reject_item_keywords(item, item_object)?;
    let properties = if item_object {
        object_properties(
            item.as_object()
                .ok_or_else(|| fail("items must be a schema object"))?,
        )?
    } else {
        IndexMap::from([("item".to_owned(), item.clone())])
    };
    let mut item_request = request.clone();
    item_request.questions = Some(properties);
    item_request.schema = None;
    if item_object {
        item_request.thinking = json!(false);
        item_request.thinking_effort = None;
        item_request.thinking_budget = None;
    }
    let mut items = item_request.compile()?;
    if item_object {
        if let Some(question) = instructions(item.as_object().expect("checked items object"))? {
            for field in &mut items {
                field.instructions = format!("{question}\n{}", field.instructions);
            }
        }
    }
    if items.iter().any(|field| {
        field.array.is_some()
            || !field.levels.is_empty()
            || field.return_probabilities
            || field
                .values
                .as_ref()
                .is_some_and(|values| values.len() > super::MAX_CONTROL_VALUES)
    }) {
        return Err(fail(
            "array items do not support nested arrays, levels, probabilities or large enums",
        ));
    }
    let integer = |key: &str, fallback| -> Result<usize, FieldError> {
        raw.get(key).map_or(Ok(fallback), |value| {
            value
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| fail(format!("{key} must be a nonnegative integer")))
        })
    };
    let min_items = integer("minItems", 0)?;
    let max_items = raw
        .get("maxItems")
        .map(|_| integer("maxItems", 0))
        .transpose()?;
    if min_items > MAX_ARRAY_ITEMS
        || max_items.is_some_and(|max| max == 0 || max > MAX_ARRAY_ITEMS || min_items > max)
    {
        return Err(fail("array bounds must be ordered and at most 50 items"));
    }
    let continue_from: Vec<Value> = raw
        .get("continue_from")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| fail("continue_from must be an array"))
        })
        .transpose()?
        .unwrap_or_default();
    let array = ArrayField {
        items,
        item_object,
        min_items,
        max_items,
        continue_from,
    };
    if array
        .max_items
        .is_some_and(|max| array.continue_from.len() > max)
        || array
            .continue_from
            .iter()
            .any(|value| !array.accepts_item(value))
    {
        return Err(fail(
            "continue_from violates the array item schema or maximum",
        ));
    }
    Ok(array)
}

fn reject_item_keywords(item: &Value, object: bool) -> Result<(), FieldError> {
    let raw = item
        .as_object()
        .ok_or_else(|| fail("items must be a schema object"))?;
    let forbidden = if object {
        &[
            "thinking",
            "thinking_effort",
            "thinking_budget",
            "depends_on",
            "when",
            "permutations",
            "return_probabilities",
            "levels",
        ][..]
    } else {
        &["depends_on", "when", "return_probabilities", "levels"][..]
    };
    if forbidden.iter().any(|key| raw.contains_key(*key))
        || raw.get("type") == Some(&json!("array"))
    {
        return Err(fail("unsupported option inside array items"));
    }
    if raw.get("type") == Some(&json!("object")) {
        check_container(raw, false)?;
        for value in object_properties(raw)?.values() {
            reject_item_keywords(value, true)?;
        }
    }
    Ok(())
}
