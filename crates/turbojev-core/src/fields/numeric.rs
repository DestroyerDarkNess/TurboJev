//! Plain-decimal limits shared by generated scalar and object fields.
use serde_json::Value;
use turbojev_backend::BackendError;

fn kind(schema: &Value) -> &str {
    schema["type"]
        .as_str()
        .or_else(|| {
            schema["type"].as_array().and_then(|types| {
                types
                    .iter()
                    .filter_map(Value::as_str)
                    .find(|kind| *kind != "null")
            })
        })
        .unwrap_or("")
}

fn contains_number(schema: &Value) -> bool {
    if schema.get("enum").is_some() {
        return false;
    }
    matches!(kind(schema), "number" | "integer")
        || schema["properties"]
            .as_object()
            .is_some_and(|properties| properties.values().any(contains_number))
}

/// A complete wrapper grammar, compatible with llama.cpp's GBNF grammar engine.
pub fn grammar(schema: &Value, digits: usize) -> Option<String> {
    if !contains_number(schema) {
        return None;
    }
    let mut rules = vec![
        r#"root ::= "{" ws "\"value\"" ws ":" ws field ws "}" ws"#.to_owned(),
        r#"ws ::= [ \t\n\r]*"#.to_owned(),
        r#"string ::= "\"" ([^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"""#
            .to_owned(),
        format!(
            r#"integer ::= "-"? ("0" | [1-9] [0-9]{{0,{}}})"#,
            digits - 1
        ),
    ];
    let mut decimals = Vec::with_capacity(digits);
    for count in 1..=digits {
        let integer = if count == 1 {
            r#"("0" | [1-9])"#.to_owned()
        } else {
            format!("[1-9] [0-9]{{{}}}", count - 1)
        };
        let fraction = if count < digits {
            format!(r#" ("." [0-9]{{1,{}}})?"#, digits - count)
        } else {
            String::new()
        };
        decimals.push(format!("({integer}{fraction})"));
    }
    rules.push(format!(r#"number ::= "-"? ({})"#, decimals.join(" | ")));
    add_rule(schema, "field", &mut rules);
    Some(rules.join("\n"))
}

fn terminal(text: &str) -> String {
    serde_json::to_string(text).expect("grammar terminal serializes")
}

fn add_rule(schema: &Value, name: &str, rules: &mut Vec<String>) {
    let mut body = if let Some(values) = schema["enum"].as_array() {
        values
            .iter()
            .map(|value| terminal(&value.to_string()))
            .collect::<Vec<_>>()
            .join(" | ")
    } else {
        match kind(schema) {
            "integer" => "integer".into(),
            "number" => "number".into(),
            "boolean" => r#""true" | "false""#.into(),
            "string" => "string".into(),
            "object" => {
                let mut fields = Vec::new();
                if let Some(properties) = schema["properties"].as_object() {
                    for (index, (key, child)) in properties.iter().enumerate() {
                        let child_name = format!("{name}-{index}");
                        add_rule(child, &child_name, rules);
                        fields.push(format!(
                            "{} ws \":\" ws {child_name}",
                            terminal(&serde_json::to_string(key).unwrap())
                        ));
                    }
                }
                format!(r#""{{" ws {} ws "}}""#, fields.join(r#" ws "," ws "#))
            }
            _ => r#""null""#.into(),
        }
    };
    if schema["type"]
        .as_array()
        .is_some_and(|types| types.contains(&Value::String("null".into())))
    {
        body = format!(r#"({body}) | "null""#);
    }
    rules.push(format!("{name} ::= {body}"));
}

/// Validate every open numeric leaf even when an adapter cannot constrain decoding.
pub fn validate(value: &Value, schema: &Value, max_digits: usize) -> Result<(), BackendError> {
    if value.is_null() || schema.get("enum").is_some() {
        return Ok(());
    }
    if matches!(kind(schema), "integer" | "number") {
        let text = value.to_string();
        // serde_json may serialize a finite decimal in exponent form after parsing.
        // Count its expanded decimal digits; the native grammar rejects exponent syntax.
        let digits = if let Some((mantissa, exponent)) = text.split_once(['e', 'E']) {
            let exponent: i32 = exponent
                .parse()
                .map_err(|_| BackendError::Inference("invalid numeric exponent".into()))?;
            let integer_digits = mantissa
                .trim_start_matches('-')
                .split('.')
                .next()
                .unwrap_or("")
                .len() as i32;
            let significant_digits = mantissa.chars().filter(char::is_ascii_digit).count() as i32;
            let point = integer_digits + exponent;
            if point <= 0 {
                (1 - point + significant_digits) as usize
            } else {
                point.max(significant_digits) as usize
            }
        } else {
            text.chars().filter(char::is_ascii_digit).count()
        };
        if digits > max_digits {
            return Err(BackendError::Inference(format!(
                "numeric value exceeds the plain-decimal limit of {max_digits} digits"
            )));
        }
    } else if let (Some(value), Some(properties)) =
        (value.as_object(), schema["properties"].as_object())
    {
        for (key, child) in properties {
            if let Some(item) = value.get(key) {
                validate(item, child, max_digits)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn limits_apply_to_open_numeric_leaves_only() {
        assert!(validate(&json!(-12.3), &json!({"type":"number"}), 3).is_ok());
        assert!(validate(&json!(12.34), &json!({"type":"number"}), 3).is_err());
        assert!(validate(&json!(12345), &json!({"type":"integer", "enum":[12345]}), 2).is_ok());
        assert!(validate(
            &json!({"n":123}),
            &json!({"type":"object", "properties":{"n":{"type":"integer"}}}),
            2
        )
        .is_err());
        assert!(grammar(&json!({"type":"string"}), 3).is_none());
        let rules = grammar(&json!({"type":["number","null"]}), 3).unwrap();
        assert!(rules.contains("root ::="));
        assert!(rules.contains("[0-9]{1,2}"));
        assert!(!rules.contains("[eE]"));
    }
}
