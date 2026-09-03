//! `ChatRequest` → the wire's JSON, and back.
//!
//! The port of Go's `openai/convert.go`. The one seat for usage
//! ([`usage`]) is shared by the blocking and streaming paths, so a counter
//! the wire adds is read in one place.

use serde::{Deserialize, Serialize};

use crate::{
    ChatRequest, ChatResponse, Error, Finish, Message, Result, Role, ToolCall, Usage, http,
};

// ---------------------------------------------------------------- the request

/// The body, exactly as it goes out.
#[derive(Debug, Serialize)]
pub(crate) struct Request {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    /// `max_tokens` was deprecated in favour of this on the chat wire, and
    /// reasoning models refuse the old name.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's skip_serializing_if shape
fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    /// Without this there is no usage on a streamed turn at all, and every
    /// run logs zero tokens.
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct WireMessage {
    role: &'static str,
    /// Omitted only on an assistant turn that is nothing but tool calls;
    /// present, possibly empty, everywhere else.
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireToolCallFunction,
}

#[derive(Debug, Serialize)]
struct WireToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct WireTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireFunction,
}

#[derive(Debug, Serialize)]
struct WireFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ToolChoice {
    #[serde(rename = "type")]
    kind: &'static str,
    function: NamedTool,
}

#[derive(Debug, Serialize)]
struct NamedTool {
    name: String,
}

/// The request, ready to serialize.
///
/// `cache_stable_prefix` is read and ignored here on purpose: OpenAI and its
/// compatibles cache a stable prefix server-side with no flag, so honouring it
/// would mean sending something the wire does not have. Anthropic's arm is
/// where it becomes a `cache_control` breakpoint.
pub(crate) fn request(req: ChatRequest, model: &str, stream: bool) -> Request {
    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    // This wire has no system field: the system prompt is the first message.
    if !req.system.is_empty() {
        messages.push(WireMessage {
            role: Role::System.as_str(),
            content: Some(req.system),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }
    for m in req.messages {
        let has_calls = !m.tool_calls.is_empty();
        messages.push(WireMessage {
            role: m.role.as_str(),
            content: if has_calls && m.content.is_empty() {
                None
            } else {
                Some(m.content)
            },
            tool_calls: m
                .tool_calls
                .into_iter()
                .map(|tc| WireToolCall {
                    id: tc.id,
                    kind: "function",
                    function: WireToolCallFunction {
                        name: tc.name,
                        arguments: tc.arguments,
                    },
                })
                .collect(),
            tool_call_id: m.tool_call_id,
        });
    }

    let has_tools = !req.tools.is_empty();
    let tools = req
        .tools
        .into_iter()
        .map(|t| WireTool {
            kind: "function",
            function: WireFunction {
                name: t.name,
                description: t.description,
                parameters: if t.parameters.is_null() {
                    serde_json::json!({"type": "object", "properties": {}})
                } else {
                    t.parameters
                },
            },
        })
        .collect();

    Request {
        model: model.to_string(),
        messages,
        tools,
        // Forcing a tool with no tools declared is not a request the wire
        // accepts, so the choice rides on there being tools — Go's rule.
        tool_choice: req
            .tool_choice
            .filter(|_| has_tools)
            .map(|name| ToolChoice {
                kind: "function",
                function: NamedTool { name },
            }),
        temperature: req.temperature,
        max_completion_tokens: req.max_tokens,
        stream,
        stream_options: stream.then_some(StreamOptions {
            include_usage: true,
        }),
    }
}

// --------------------------------------------------------------- the response

#[derive(Debug, Deserialize)]
struct Response {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    message: ChoiceMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChoiceMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

#[derive(Debug, Deserialize)]
struct ResponseToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    function: ResponseToolCallFunction,
}

#[derive(Debug, Default, Deserialize)]
struct ResponseToolCallFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

/// A completed turn.
///
/// An empty `choices` is an [`Error::Decode`] and not a panic: Go indexed
/// `Choices[0]` unconditionally and a gateway that answers 200 with no choice
/// took the process down with it.
pub(crate) fn response(text: &str) -> Result<ChatResponse> {
    let resp: Response = serde_json::from_str(text)
        .map_err(|e| Error::Decode(format!("{e} in {}", http::clip(text))))?;
    let Some(choice) = resp.choices.into_iter().next() else {
        return Err(Error::Decode("a 200 with no choices in it".into()));
    };

    let tool_calls: Vec<ToolCall> = choice
        .message
        .tool_calls
        .into_iter()
        .map(|tc| ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
        })
        .collect();

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: choice.message.content.unwrap_or_default(),
            tool_calls: tool_calls.clone(),
            tool_call_id: None,
        },
        tool_calls,
        usage: resp.usage.map(usage).unwrap_or_default(),
        finish: choice
            .finish_reason
            .filter(|s| !s.is_empty())
            .map(|s| Finish::parse(&s)),
    })
}

// -------------------------------------------------------------------- usage

#[derive(Debug, Default, Deserialize)]
pub(crate) struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionDetails>,
}

#[derive(Debug, Default, Deserialize)]
struct PromptDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Debug, Default, Deserialize)]
struct CompletionDetails {
    #[serde(default)]
    reasoning_tokens: u32,
}

/// The one seat for this wire's counters, shared by the blocking and the
/// streaming path. `cache_creation` stays 0: OpenAI does not bill a cache
/// write separately, so there is no number to report.
pub(crate) fn usage(u: WireUsage) -> Usage {
    Usage {
        input: u.prompt_tokens,
        output: u.completion_tokens,
        cached: u.prompt_tokens_details.unwrap_or_default().cached_tokens,
        cache_creation: 0,
        reasoning: u
            .completion_tokens_details
            .unwrap_or_default()
            .reasoning_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, Tool};

    fn json(req: Request) -> serde_json::Value {
        serde_json::to_value(req).unwrap()
    }

    #[test]
    fn the_system_prompt_is_the_first_message() {
        let out = json(request(
            ChatRequest::ask("You are terse.", "Say hi"),
            "gpt-4o-mini",
            false,
        ));
        assert_eq!(
            out,
            serde_json::json!({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Say hi"},
                ],
            }),
            "a plain ask sends nothing it was not given"
        );
    }

    #[test]
    fn a_tool_turn_round_trips_through_the_wires_shape() {
        let req = ChatRequest {
            messages: vec![
                Message::user("weather in NYC?"),
                Message::assistant("").with_tool_calls(vec![ToolCall {
                    id: "call_123".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"location":"NYC"}"#.into(),
                }]),
                Message::tool("call_123", "17C"),
            ],
            tools: vec![Tool {
                name: "get_weather".into(),
                description: "the weather".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            tool_choice: Some("get_weather".into()),
            temperature: Some(0.2),
            max_tokens: Some(256),
            system: String::new(),
            cache_stable_prefix: true,
        };
        let out = json(request(req, "gpt-4o", false));
        assert_eq!(
            out,
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "weather in NYC?"},
                    // No content key at all: an assistant turn that is only
                    // tool calls must not send an empty string.
                    {"role": "assistant", "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"},
                    }]},
                    {"role": "tool", "content": "17C", "tool_call_id": "call_123"},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "the weather",
                        "parameters": {"type": "object"},
                    },
                }],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "temperature": 0.2,
                "max_completion_tokens": 256,
            }),
            "cache_stable_prefix leaves the OpenAI request byte-identical"
        );
    }

    #[test]
    fn streaming_asks_for_the_usage_it_would_not_otherwise_get() {
        let out = json(request(ChatRequest::ask("", "hi"), "gpt-4o", true));
        assert_eq!(out["stream"], serde_json::json!(true));
        assert_eq!(
            out["stream_options"],
            serde_json::json!({"include_usage": true})
        );
        // And a blocking turn sends neither.
        let out = json(request(ChatRequest::ask("", "hi"), "gpt-4o", false));
        assert!(out.get("stream").is_none(), "{out}");
        assert!(out.get("stream_options").is_none(), "{out}");
    }

    #[test]
    fn a_tool_choice_without_tools_is_not_sent() {
        let req = ChatRequest {
            messages: vec![Message::user("hi")],
            tool_choice: Some("get_weather".into()),
            ..ChatRequest::default()
        };
        let out = json(request(req, "gpt-4o", false));
        assert!(out.get("tool_choice").is_none(), "{out}");
    }

    #[test]
    fn a_tool_with_no_arguments_still_gets_a_schema() {
        let req = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![Tool {
                name: "now".into(),
                description: "the time".into(),
                parameters: serde_json::Value::Null,
            }],
            ..ChatRequest::default()
        };
        let out = json(request(req, "gpt-4o", false));
        assert_eq!(
            out["tools"][0]["function"]["parameters"],
            serde_json::json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn a_200_with_no_choices_is_an_error_and_not_a_panic() {
        match response(r#"{"id":"x","choices":[]}"#) {
            Err(Error::Decode(why)) => assert!(why.contains("no choices"), "{why}"),
            other => panic!("{other:?}"),
        }
    }
}
