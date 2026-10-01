// OpenAI-Chat-Completions-compatible client: OpenAI, OpenRouter, Ollama,
// LiteLLM, LM Studio, vLLM. One wire shape covers all five.

use serde_json::{json, Value};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub fn endpoint(base: &str) -> String {
    format!("{}/chat/completions", base.trim_end_matches('/'))
}

fn body(model: &str, system: &str, messages: &[Value]) -> Value {
    json!({
        "model": model,
        "max_tokens": super::claude::MAX_TOKENS,
        "system": system,
        "messages": messages,
    })
}

/// Reads `choices[0].message.content` as a string or an array of text parts.
fn reply_text(response: &Value) -> Result<String, String> {
    let content = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"));
    let text = match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return Err("Unexpected API response.".into()),
    };
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("No response text.".into());
    }
    Ok(text)
}

fn api_error(status: reqwest::StatusCode, text: &str) -> String {
    let detail = serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| text.chars().take(200).collect());
    format!("OpenAI API {status}: {detail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_joins_without_double_slash() {
        assert_eq!(
            endpoint("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            endpoint("http://localhost:11434/v1"),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn body_carries_model_system_and_messages() {
        let b = body("gpt-5.4-mini", "sys", &[json!({"role": "user"})]);
        assert_eq!(b["model"], "gpt-5.4-mini");
        assert_eq!(b["system"], "sys");
        assert_eq!(b["messages"], json!([{"role": "user"}]));
    }

    #[test]
    fn reply_reads_string_content() {
        let v = json!({"choices": [{"message": {"content": "  hi  "}}]});
        assert_eq!(reply_text(&v).unwrap(), "hi");
    }

    #[test]
    fn reply_reads_part_array_and_rejects_empties() {
        let v = json!({"choices": [{"message": {"content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]}}]});
        assert_eq!(reply_text(&v).unwrap(), "a\nb");
        let empty = json!({"choices": [{"message": {"content": ""}}]});
        assert!(reply_text(&empty).is_err());
        assert!(reply_text(&json!({})).is_err());
    }

    #[test]
    fn error_surfaces_the_api_message() {
        let e = api_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error": {"message": "bad key"}}"#,
        );
        assert!(e.contains("bad key"), "got: {e}");
    }
}
