//! `ChatRequest` → the messages body, and back.
//!
//! The port of Go's `anthropic/convert.go`, which was a mapping onto the
//! vendor SDK's types; here it is the JSON itself, because that is what the
//! pins are about.
//!
//! Four things this wire does that the OpenAI one does not, and they are the
//! whole of why it is a second wire rather than a flag:
//!
//! - the **system prompt is its own field**, a list of text blocks, not the
//!   first message;
//! - **`max_tokens` is required**, so there is a default here and nowhere else;
//! - a tool call is a **`tool_use` content block** whose input is a JSON
//!   object, and its answer a `tool_result` block on a *user* turn;
//! - **`cache_stable_prefix` is acted on**: it becomes a `cache_control`
//!   breakpoint, and this is the only wire where the flag changes a byte.

use serde::{Deserialize, Serialize};

use crate::{
    ChatRequest, ChatResponse, Error, Finish, Message, Result, Role, ToolCall, Usage, http,
};

use super::DEFAULT_MAX_TOKENS;

// ---------------------------------------------------------------- the request

/// The body, exactly as it goes out.
#[derive(Debug, Serialize)]
pub(crate) struct Request {
    model: String,
    /// Required by this wire, always sent.
    max_tokens: u32,
    messages: Vec<WireMessage>,
    /// A list of text blocks, not a string: a `cache_control` breakpoint has
    /// to land on a block, and the last one is where it lands.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<TextBlock>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's skip_serializing_if shape
fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Serialize)]
struct TextBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

impl TextBlock {
    fn new(text: String) -> TextBlock {
        TextBlock {
            kind: "text",
            text,
            cache_control: None,
        }
    }
}

/// The one prompt-cache breakpoint shape this wire has.
#[derive(Debug, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

impl CacheControl {
    fn ephemeral() -> CacheControl {
        CacheControl { kind: "ephemeral" }
    }
}

#[derive(Debug, Serialize)]
struct WireMessage {
    role: &'static str,
    content: Vec<Block>,
}

/// A content block on the way out. One enum rather than three structs so a
/// turn's blocks keep their order in one `Vec`.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum Block {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        /// A JSON **object**, not the argument text: this wire takes the
        /// parsed input, which is why [`request`] can fail.
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize)]
struct WireTool {
    name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    description: String,
    /// The caller's JSON Schema, verbatim. Go sent only `properties` and
    /// `required` and dropped everything else in the schema — `$defs`, nested
    /// object types, `additionalProperties` — which silently widened every
    /// tool this wire served.
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Serialize)]
struct ToolChoice {
    #[serde(rename = "type")]
    kind: &'static str,
    name: String,
}

/// The request, ready to serialize.
///
/// Fails only on a tool call whose arguments are not JSON — see
/// [`tool_use_input`].
pub(crate) fn request(req: ChatRequest, model: &str, stream: bool) -> Result<Request> {
    // The system prompt, plus any system message in the transcript. Go dropped
    // those messages on the floor; folding them in is the same content the
    // OpenAI wire would have sent, hoisted to where this wire keeps it.
    let mut system: Vec<TextBlock> = Vec::new();
    if !req.system.is_empty() {
        system.push(TextBlock::new(req.system));
    }

    let mut messages: Vec<WireMessage> = Vec::new();
    for m in req.messages {
        match m.role {
            Role::System => {
                if !m.content.is_empty() {
                    system.push(TextBlock::new(m.content));
                }
            }
            Role::User => {
                // Every turn in the transcript goes out, empty content and
                // all — the OpenAI wire sends it, and one `ChatRequest` must
                // not become a different conversation depending on which wire
                // takes it. This wire refuses an empty text block, and that
                // refusal is the provider's to make: dropping the turn here
                // would answer a question nobody asked it, silently.
                messages.push(WireMessage {
                    role: "user",
                    content: vec![Block::Text { text: m.content }],
                });
            }
            Role::Assistant => {
                let has_calls = !m.tool_calls.is_empty();
                let mut content = Vec::with_capacity(m.tool_calls.len() + 1);
                // The same sentence, and its one exception: an assistant turn
                // that is ONLY tool calls has no text block at all, because an
                // empty one beside them is what this wire rejects. With no
                // calls beside it, an empty turn goes out like the user's.
                if !m.content.is_empty() || !has_calls {
                    content.push(Block::Text { text: m.content });
                }
                for tc in m.tool_calls {
                    content.push(Block::ToolUse {
                        input: tool_use_input(&tc)?,
                        id: tc.id,
                        name: tc.name,
                    });
                }
                messages.push(WireMessage {
                    role: "assistant",
                    content,
                });
            }
            Role::Tool => {
                // A tool result is a user turn on this wire, and parallel
                // results belong in ONE turn: splitting them across
                // consecutive user messages is what teaches a model to stop
                // making parallel calls. The transcript spells one message per
                // result, so consecutive ones fold together here.
                let block = Block::ToolResult {
                    tool_use_id: m.tool_call_id.unwrap_or_default(),
                    content: m.content,
                };
                match messages.last_mut() {
                    Some(last)
                        if last.role == "user"
                            && last
                                .content
                                .iter()
                                .all(|b| matches!(b, Block::ToolResult { .. })) =>
                    {
                        last.content.push(block);
                    }
                    _ => messages.push(WireMessage {
                        role: "user",
                        content: vec![block],
                    }),
                }
            }
        }
    }

    let has_tools = !req.tools.is_empty();
    let mut tools: Vec<WireTool> = req
        .tools
        .into_iter()
        .map(|t| WireTool {
            name: t.name,
            description: t.description,
            input_schema: if t.parameters.is_null() {
                serde_json::json!({"type": "object", "properties": {}})
            } else {
                t.parameters
            },
            cache_control: None,
        })
        .collect();

    // The prompt cache breakpoint, and the whole of it. This wire caches in
    // the order tools → system → messages, so a breakpoint on the LAST system
    // block caches the tools and the system prompt together; the last tool is
    // marked as well so a request with tools and no system prompt still caches
    // its definitions. The per-turn messages after it stay uncached, which is
    // the point — they are what changes.
    if req.cache_stable_prefix {
        if let Some(last) = system.last_mut() {
            last.cache_control = Some(CacheControl::ephemeral());
        }
        if let Some(last) = tools.last_mut() {
            last.cache_control = Some(CacheControl::ephemeral());
        }
    }

    Ok(Request {
        model: model.to_string(),
        max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages,
        system,
        tools,
        // Forcing a tool with no tools declared is not a request the wire
        // accepts, so the choice rides on there being tools — the OpenAI
        // wire's rule, and Go's.
        tool_choice: req
            .tool_choice
            .filter(|_| has_tools)
            .map(|name| ToolChoice { kind: "tool", name }),
        temperature: req.temperature,
        stream,
    })
}

/// A replayed tool call's arguments as this wire wants them: an object.
///
/// [`ToolCall::arguments`] is the JSON **text** the model produced, because a
/// stream assembles it from fragments and a model that emits invalid JSON is a
/// fact the caller should see. This wire cannot carry text: `input` is a JSON
/// value. So empty arguments become `{}` — a call that takes none — and
/// anything that does not parse is refused by name rather than sent as `null`,
/// which is what Go did and what makes a model answer a call it cannot read.
fn tool_use_input(tc: &ToolCall) -> Result<serde_json::Value> {
    let text = tc.arguments.trim();
    if text.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(text).map_err(|e| {
        Error::Decode(format!(
            "the arguments of tool call {} ({}) are not JSON: {e}",
            tc.name, tc.id
        ))
    })
}

// --------------------------------------------------------------- the response

#[derive(Debug, Deserialize)]
struct Response {
    #[serde(default)]
    content: Vec<ResponseBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct ResponseBlock {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

/// A completed turn.
///
/// There is no `choices` here and so no empty-choices trap: a message with no
/// content blocks is an answer that said nothing, which is a turn the caller
/// can report, not a body that failed to parse.
pub(crate) fn response(text: &str) -> Result<ChatResponse> {
    let resp: Response = serde_json::from_str(text)
        .map_err(|e| Error::Decode(format!("{e} in {}", http::clip(text))))?;

    let mut content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for block in resp.content {
        match block.kind.as_str() {
            "text" => content.push_str(&block.text),
            "tool_use" => tool_calls.push(ToolCall {
                id: block.id,
                name: block.name,
                // Back to the text the rest of the crate carries. `input` is
                // an object here, so this always round-trips.
                arguments: block
                    .input
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "{}".into()),
            }),
            // A thinking block, or a kind nobody has seen yet: not an error.
            // A wire that adds a block type must not end a turn with one.
            _ => {}
        }
    }

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content,
            tool_calls: tool_calls.clone(),
            tool_call_id: None,
        },
        tool_calls,
        usage: resp.usage.map(usage).unwrap_or_default(),
        finish: resp
            .stop_reason
            .filter(|s| !s.is_empty())
            .map(|s| finish(&s)),
    })
}

/// This wire's spellings for why generation stopped.
///
/// Everything it does not name falls through to [`Finish::parse`], so a reason
/// nobody has seen yet — `pause_turn`, whatever comes next — stays readable
/// rather than ending a run with a parse failure.
pub(crate) fn finish(reason: &str) -> Finish {
    match reason {
        // A stop sequence is a stop: the caller asked for the turn to end
        // there, and it did.
        "end_turn" | "stop_sequence" => Finish::Stop,
        "tool_use" => Finish::ToolCalls,
        // The classifiers declined. It is not the model's own `end_turn`, and
        // a caller that re-prompts on `Length` must not re-prompt on this.
        "refusal" => Finish::ContentFilter,
        other => Finish::parse(other),
    }
}

// -------------------------------------------------------------------- usage

#[derive(Debug, Default, Clone, Copy, Deserialize)]
pub(crate) struct WireUsage {
    #[serde(default)]
    pub(crate) input_tokens: u32,
    #[serde(default)]
    pub(crate) output_tokens: u32,
    /// A cache HIT, billed at a discount.
    #[serde(default)]
    pub(crate) cache_read_input_tokens: u32,
    /// A cache WRITE — the first call's price for a cached prefix, billed
    /// above the input rate. This is the wire that has one.
    #[serde(default)]
    pub(crate) cache_creation_input_tokens: u32,
}

/// The one seat for this wire's counters, shared by the blocking and the
/// streaming path.
///
/// The asymmetry to know, because it is not a bug: here `input` **excludes**
/// the cached tokens and `cached` is added on top, where the OpenAI wire's
/// `input` includes them. Each call is billed as reported either way.
/// `reasoning` stays 0: this wire bundles thinking tokens into
/// `output_tokens` and does not separate them, and a character-count estimate
/// would be a number nobody measured.
pub(crate) fn usage(u: WireUsage) -> Usage {
    Usage {
        input: u.input_tokens,
        output: u.output_tokens,
        cached: u.cache_read_input_tokens,
        cache_creation: u.cache_creation_input_tokens,
        reasoning: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, Tool};

    fn json(req: Request) -> serde_json::Value {
        serde_json::to_value(req).unwrap()
    }

    fn build(req: ChatRequest) -> serde_json::Value {
        json(request(req, "claude-sonnet-4", false).unwrap())
    }

    #[test]
    fn the_system_prompt_is_its_own_field_and_max_tokens_is_always_sent() {
        let out = build(ChatRequest::ask("You are terse.", "Say hi"));
        assert_eq!(
            out,
            serde_json::json!({
                "model": "claude-sonnet-4",
                // Required by this wire: a request without it is a 400.
                "max_tokens": DEFAULT_MAX_TOKENS,
                "system": [{"type": "text", "text": "You are terse."}],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Say hi"}]},
                ],
            }),
            "a plain ask sends nothing it was not given"
        );
    }

    #[test]
    fn a_system_message_in_the_transcript_is_hoisted_and_not_dropped() {
        // Go skipped `system` messages in the loop and sent only the field,
        // so a caller that put its prompt in the transcript — which is what
        // the OpenAI wire takes — got a turn with no instructions at all.
        let req = ChatRequest {
            messages: vec![Message::system("Answer in French."), Message::user("hi")],
            system: "You are terse.".into(),
            ..ChatRequest::default()
        };
        let out = build(req);
        assert_eq!(
            out["system"],
            serde_json::json!([
                {"type": "text", "text": "You are terse."},
                {"type": "text", "text": "Answer in French."},
            ])
        );
        assert_eq!(out["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn a_tool_turn_becomes_tool_use_and_tool_result_blocks() {
        let req = ChatRequest {
            messages: vec![
                Message::user("weather in NYC?"),
                Message::assistant("Looking.").with_tool_calls(vec![ToolCall {
                    id: "toolu_1".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"location":{"city":"NYC"}}"#.into(),
                }]),
                Message::tool("toolu_1", "17C"),
            ],
            // Not a flat schema: a `$defs` entry, a property that `$ref`s it,
            // and `additionalProperties: false` on both objects. Go sent only
            // `properties` and `required`, so every one of these was dropped
            // and the tool the model saw accepted anything — which is what the
            // assertion below is actually about.
            tools: vec![Tool {
                name: "get_weather".into(),
                description: "the weather".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {"location": {"$ref": "#/$defs/place"}},
                    "required": ["location"],
                    "$defs": {
                        "place": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"type": "string", "enum": ["C", "F"]},
                            },
                            "required": ["city"],
                        },
                    },
                }),
            }],
            tool_choice: Some("get_weather".into()),
            temperature: Some(0.2),
            max_tokens: Some(256),
            ..ChatRequest::default()
        };
        let out = build(req);
        assert_eq!(
            out,
            serde_json::json!({
                "model": "claude-sonnet-4",
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "Looking."},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                         // The arguments arrive as text and go out as an object.
                         "input": {"location": {"city": "NYC"}}},
                    ]},
                    // A tool result is a USER turn on this wire.
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "17C"},
                    ]},
                ],
                "tools": [{
                    "name": "get_weather",
                    "description": "the weather",
                    // Verbatim, to the last key: the `$defs`, the `$ref` into
                    // it, and both `additionalProperties: false`. Reaching for
                    // `properties` and `required` again — it does look like
                    // less JSON — fails here.
                    "input_schema": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {"location": {"$ref": "#/$defs/place"}},
                        "required": ["location"],
                        "$defs": {
                            "place": {
                                "type": "object",
                                "additionalProperties": false,
                                "properties": {
                                    "city": {"type": "string"},
                                    "unit": {"type": "string", "enum": ["C", "F"]},
                                },
                                "required": ["city"],
                            },
                        },
                    },
                }],
                "tool_choice": {"type": "tool", "name": "get_weather"},
                "temperature": 0.2,
            })
        );
    }

    #[test]
    fn parallel_tool_results_ride_in_one_user_turn() {
        // Two results, one message. Go sent a user message each, and splitting
        // them is what teaches a model to stop making parallel calls.
        let req = ChatRequest {
            messages: vec![
                Message::assistant("").with_tool_calls(vec![
                    ToolCall {
                        id: "toolu_a".into(),
                        name: "a".into(),
                        arguments: String::new(),
                    },
                    ToolCall {
                        id: "toolu_b".into(),
                        name: "b".into(),
                        arguments: r#"{"x":1}"#.into(),
                    },
                ]),
                Message::tool("toolu_a", "one"),
                Message::tool("toolu_b", "two"),
                Message::user("and now?"),
            ],
            ..ChatRequest::default()
        };
        let out = build(req);
        assert_eq!(
            out["messages"],
            serde_json::json!([
                {"role": "assistant", "content": [
                    // No text block at all: an assistant turn that is only
                    // tool calls must not send an empty one, which this wire
                    // rejects.
                    {"type": "tool_use", "id": "toolu_a", "name": "a", "input": {}},
                    {"type": "tool_use", "id": "toolu_b", "name": "b", "input": {"x": 1}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": "one"},
                    {"type": "tool_result", "tool_use_id": "toolu_b", "content": "two"},
                ]},
                {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
            ])
        );
    }

    #[test]
    fn the_cache_breakpoint_lands_on_the_last_system_block_and_the_last_tool() {
        let tools = |n: usize| {
            (0..n)
                .map(|i| Tool {
                    name: format!("t{i}"),
                    description: String::new(),
                    parameters: serde_json::json!({"type": "object"}),
                })
                .collect::<Vec<_>>()
        };
        let req = |cache: bool| ChatRequest {
            messages: vec![Message::system("second"), Message::user("hi")],
            tools: tools(2),
            system: "first".into(),
            cache_stable_prefix: cache,
            ..ChatRequest::default()
        };

        // Off: the request is byte-identical to one that never heard of the
        // flag, which is what makes it safe to default off.
        let out = build(req(false));
        assert!(!out.to_string().contains("cache_control"), "{out}");

        let out = build(req(true));
        let breakpoint = serde_json::json!({"type": "ephemeral"});
        // The LAST system block, and only it: this wire caches
        // tools → system → messages, so one breakpoint there covers both.
        assert!(out["system"][0].get("cache_control").is_none(), "{out}");
        assert_eq!(out["system"][1]["cache_control"], breakpoint);
        // And the last tool, so a request with tools and no system prompt
        // still caches its definitions.
        assert!(out["tools"][0].get("cache_control").is_none(), "{out}");
        assert_eq!(out["tools"][1]["cache_control"], breakpoint);
        // Nowhere else — the per-turn messages are what changes.
        assert_eq!(out.to_string().matches("cache_control").count(), 2, "{out}");
    }

    #[test]
    fn an_empty_turn_goes_out_as_it_was_given_and_is_not_dropped() {
        // The OpenAI wire pushes every message in the transcript, so this one
        // does too: one `ChatRequest` must not become a different conversation
        // depending on which wire takes it. `ask(system, "")` coming out as
        // `messages: []` would be a 400 answering a question nobody asked.
        let out = build(ChatRequest::ask("You are terse.", ""));
        assert_eq!(
            out["messages"],
            serde_json::json!([
                {"role": "user", "content": [{"type": "text", "text": ""}]},
            ])
        );

        // And an assistant turn with nothing in it, for the same reason. The
        // one exception is the turn that is only tool calls, where the empty
        // text block would sit beside them — that is the wire's rule, and it
        // has its own pin above.
        let out = build(ChatRequest {
            messages: vec![Message::assistant(""), Message::user("go on")],
            ..ChatRequest::default()
        });
        assert_eq!(
            out["messages"][0],
            serde_json::json!({"role": "assistant", "content": [{"type": "text", "text": ""}]})
        );
    }

    #[test]
    fn a_tool_choice_without_tools_is_not_sent() {
        let req = ChatRequest {
            messages: vec![Message::user("hi")],
            tool_choice: Some("get_weather".into()),
            ..ChatRequest::default()
        };
        assert!(build(req).get("tool_choice").is_none());
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
        assert_eq!(
            build(req)["tools"][0]["input_schema"],
            serde_json::json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn a_replayed_tool_call_whose_arguments_are_not_json_is_refused_by_name() {
        // Go's `_ = json.Unmarshal` sent `input: null` and let the model
        // answer a call it could not read.
        let req = ChatRequest {
            messages: vec![Message::assistant("").with_tool_calls(vec![ToolCall {
                id: "toolu_1".into(),
                name: "search".into(),
                arguments: r#"{"query":"#.into(),
            }])],
            ..ChatRequest::default()
        };
        match request(req, "claude-sonnet-4", false) {
            Err(Error::Decode(why)) => {
                assert!(why.contains("search"), "{why}");
                assert!(why.contains("toolu_1"), "{why}");
            }
            other => panic!("{:?}", other.err()),
        }
    }

    #[test]
    fn a_blocking_reply_gives_up_its_text_tool_calls_and_both_cache_counters() {
        let answer = response(
            r#"{
              "id": "msg_1", "type": "message", "role": "assistant",
              "content": [
                {"type": "text", "text": "Looking it up."},
                {"type": "thinking", "thinking": "…"},
                {"type": "tool_use", "id": "toolu_1", "name": "search",
                 "input": {"query": "acme"}}
              ],
              "stop_reason": "tool_use",
              "usage": {"input_tokens": 100, "output_tokens": 25,
                        "cache_read_input_tokens": 1024,
                        "cache_creation_input_tokens": 512}
            }"#,
        )
        .unwrap();

        assert_eq!(answer.message.content, "Looking it up.");
        assert_eq!(answer.finish, Some(Finish::ToolCalls));
        assert_eq!(
            answer.tool_calls,
            vec![ToolCall {
                id: "toolu_1".into(),
                name: "search".into(),
                arguments: r#"{"query":"acme"}"#.into(),
            }]
        );
        assert_eq!(answer.message.tool_calls, answer.tool_calls);
        // input EXCLUDES cached on this wire — the asymmetry `Usage` names.
        assert_eq!(answer.usage.input, 100);
        assert_eq!(answer.usage.output, 25);
        assert_eq!(answer.usage.cached, 1024);
        assert_eq!(
            answer.usage.cache_creation, 512,
            "the wire that bills a cache write"
        );
        assert_eq!(
            answer.usage.reasoning, 0,
            "thinking tokens are inside output_tokens and are not guessed at"
        );
    }

    #[test]
    fn every_stop_reason_this_wire_sends_lands_on_an_arm_a_caller_branches_on() {
        assert_eq!(finish("end_turn"), Finish::Stop);
        assert_eq!(finish("stop_sequence"), Finish::Stop);
        assert_eq!(finish("max_tokens"), Finish::Length);
        assert_eq!(finish("tool_use"), Finish::ToolCalls);
        assert_eq!(finish("refusal"), Finish::ContentFilter);
        // And one nobody has seen yet stays readable.
        assert_eq!(finish("pause_turn"), Finish::Other("pause_turn".into()));
    }
}
