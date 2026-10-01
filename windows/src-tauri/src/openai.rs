// OpenAI-Chat-Completions-compatible client: OpenAI, OpenRouter, Ollama,
// LiteLLM, LM Studio, vLLM. One wire shape covers all five.

use serde_json::{json, Value};

use crate::claude::MAX_INLINE_TEXT;

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

/// One chat turn, same contract as `claude::send`. History is provider-scoped
/// (see `Chat::ensure_provider`); file context is text-inline + image data
/// URLs — Chat Completions has no document block, so unreadable PDFs are
/// skipped. No tools: web search is Anthropic-only.
pub async fn send(
    chat: &crate::claude::Chat,
    base_url: &str,
    key: &str,
    model: &str,
    query: String,
    context: Option<crate::claude::ChatContext>,
) -> Result<crate::claude::ChatReply, String> {
    use crate::claude::{ChatContext, SYSTEM_PROMPT};

    chat.ensure_provider("openai-compatible");

    let mut parts: Vec<Value> = Vec::new();
    if chat.is_empty() {
        match &context {
            Some(ChatContext::File { name, path }) => {
                parts.extend(file_parts(path));
                parts.push(json!({ "type": "text", "text": format!("File: {name}") }));
            }
            Some(ChatContext::Window { app_name, title, url }) => {
                let mut text = format!("Context — App: {app_name}, Window: {title}");
                if let Some(url) = url {
                    text.push_str(&format!(", URL: {url}"));
                }
                parts.push(json!({ "type": "text", "text": text }));
            }
            None => {}
        }
    }
    parts.push(json!({ "type": "text", "text": query }));
    chat.push(json!({ "role": "user", "content": parts }));

    let body = body(model, SYSTEM_PROMPT, &chat.snapshot());
    let response = match call(base_url, key, &body).await {
        Ok(v) => v,
        Err(err) => {
            chat.pop();
            return Err(err);
        }
    };
    let text = match reply_text(&response) {
        Ok(t) => t,
        Err(err) => {
            chat.pop();
            return Err(err);
        }
    };
    chat.push(json!({ "role": "assistant", "content": text.clone() }));
    Ok(crate::claude::ChatReply { text })
}

async fn call(base_url: &str, key: &str, body: &Value) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build()
        .map_err(|e| e.to_string())?;

    let mut request = client
        .post(endpoint(base_url))
        .header("content-type", "application/json")
        .json(body);
    if !key.is_empty() {
        // Local backends (Ollama) need no key; remote ones do.
        request = request.header("authorization", format!("Bearer {key}"));
    }
    let response = request
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?;

    let status = response.status();
    let text = response.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(api_error(status, &text));
    }
    serde_json::from_str(&text).map_err(|e| format!("Bad API response: {e}"))
}

/// Text/code → inline text part; images → data-URL part; anything else → none.
fn file_parts(path: &str) -> Vec<Value> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let media = match ext.as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    };
    if let Some(media) = media {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return vec![],
        };
        return vec![json!({
            "type": "image_url",
            "image_url": { "url": format!("data:{media};base64,{}", crate::claude::base64(&bytes)) },
        })];
    }

    let len = match std::fs::metadata(path) {
        Ok(m) => m.len(),
        Err(_) => return vec![],
    };
    if len > MAX_INLINE_TEXT {
        return vec![];
    }
    match std::fs::read_to_string(path) {
        Ok(text) => vec![json!({ "type": "text", "text": format!("File contents:\n{text}") })],
        Err(_) => vec![],
    }
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
