//! What a caller hands a provider and what it gets back.
//!
//! The port of Go's `domains/llm/providers/provider.go`, with the strings that
//! were an enum in everything but the type made into enums: [`Role`] and
//! [`Finish`]. There is no `model` on a [`ChatRequest`] — the model is part of
//! the [`Wire`](crate::Wire) a client was built from, so one client is one
//! deployment and a caller cannot ask it for a different model by accident.

use crate::{Error, Image, Result};

/// Who a message is from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// The result of a tool call, answering [`Message::tool_call_id`].
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One turn in the transcript.
///
/// Not a wire shape. Nothing here serializes: what goes out is
/// `openai::convert::WireMessage`, which omits `content` entirely on an
/// assistant turn that is only tool calls — a rule this struct could not
/// carry and would misstate if it tried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Images on a user turn, sent before its text — the order both wires'
    /// docs recommend. Only a user turn may carry them; a request with one
    /// elsewhere is refused before the socket, by both wires alike.
    pub images: Vec<Image>,
    /// Set on an assistant turn that asked for tools.
    pub tool_calls: Vec<ToolCall>,
    /// Set on a `tool` turn: which call this answers.
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Message {
        Message::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Message {
        Message::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Message {
        Message::new(Role::Assistant, content)
    }

    /// The answer to one tool call, by the id the model gave it.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Message {
        Message {
            role: Role::Tool,
            content: content.into(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    fn new(role: Role, content: impl Into<String>) -> Message {
        Message {
            role,
            content: content.into(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// An assistant turn that asked for tools; the content may be empty.
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Message {
        self.tool_calls = calls;
        self
    }

    /// A user turn that shows the model these images, before its text.
    pub fn with_images(mut self, images: impl IntoIterator<Item = Image>) -> Message {
        self.images.extend(images);
        self
    }
}

/// An image anywhere but a user turn is a request neither chat wire sends:
/// OpenAI takes image parts only from the user, and Anthropic refuses them
/// from the assistant. Refused here, by the turn's index, so the answer is
/// the same whichever wire the card names.
pub(crate) fn images_on_user_turns_only(wire: &'static str, req: &ChatRequest) -> Result<()> {
    match req
        .messages
        .iter()
        .position(|m| !m.images.is_empty() && m.role != Role::User)
    {
        Some(i) => Err(Error::Invalid {
            wire,
            what: format!(
                "message {i} ({}) carries images; only a user turn may",
                req.messages[i].role.as_str()
            ),
        }),
        None => Ok(()),
    }
}

/// A function the model may call.
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema. `Value::Null` means "no arguments" and is sent as an empty
    /// object where the wire requires the field.
    pub parameters: serde_json::Value,
}

/// The model's request to call one tool.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCall {
    /// The id to echo back on the answering [`Message::tool_call_id`].
    pub id: String,
    pub name: String,
    /// The arguments, as the JSON text the model produced. Kept as text and
    /// not a `Value` on purpose: a stream assembles it from fragments, and a
    /// model that emits invalid JSON is a fact the caller should see rather
    /// than an error thrown three layers below where it can be reported.
    pub arguments: String,
}

/// A chat turn to run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[allow(clippy::struct_field_names)]
    pub tools: Vec<Tool>,
    /// Force one named tool. `None` lets the model choose, which is the
    /// default; a name not in `tools` is a caller error the wire will report.
    pub tool_choice: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    /// The system prompt. Wires that have no system role put it first as a
    /// system message; wires that do (Anthropic) put it in its own field.
    ///
    /// This is the **stable** part: what stays the same bytes from one
    /// request to the next, which is what a prompt cache can reuse. Anything
    /// that changes per request goes in [`ChatRequest::system_volatile`].
    pub system: String,
    /// **The part of the system prompt that changes per request** — a clock,
    /// the record being processed — sent after [`ChatRequest::system`].
    ///
    /// Every provider caches by prefix, so volatile text inside the cached
    /// part costs the cache everything behind it. On Anthropic it is a system
    /// block of its own after the `cache_control` breakpoint, which
    /// `cache_stable_prefix` stamps on the last *stable* block; on OpenAI it
    /// is appended to the system message, one blank line after the stable
    /// text, so the stable text still leads. Empty sends nothing, and the
    /// request is byte-identical to one that never had the field.
    ///
    /// **It is still ahead of `messages`**, on both wires, so the transcript
    /// behind it is not part of any reusable prefix: on OpenAI it ends the
    /// server-side match before the first message. Per-request text that has
    /// a user turn to ride belongs on that turn; this field is for the text
    /// that has to stay in the system prompt.
    ///
    /// **A `Role::System` message in the transcript lands on different sides
    /// of it.** Anthropic hoists such a message into its system blocks, on
    /// the stable side — stable, hoisted, then volatile — because a breakpoint
    /// behind the volatile block would never hit. OpenAI has no field to
    /// hoist into and sends that message where it stands in the transcript,
    /// after the system message that carries this one. Each wire keeps the
    /// placement it already had for those messages.
    pub system_volatile: String,
    /// Ask the provider to cache the stable prefix (system prompt + tool
    /// definitions). Only Anthropic acts on it — OpenAI and its compatibles
    /// cache prefixes server-side with no flag — so on the OpenAI wire this
    /// is read and deliberately ignored, and the request is byte-identical
    /// either way.
    pub cache_stable_prefix: bool,
}

impl ChatRequest {
    /// The one-shot shape: a system prompt and a question.
    pub fn ask(system: impl Into<String>, question: impl Into<String>) -> ChatRequest {
        ChatRequest {
            messages: vec![Message::user(question)],
            system: system.into(),
            ..ChatRequest::default()
        }
    }
}

/// Why generation stopped. An enum and not a string: the callers that matter
/// branch on it (a `Length` is re-prompted, a `ToolCalls` runs the tools), and
/// a typo in a match arm should be a compile error. [`Finish::Other`] keeps
/// an unknown reason readable instead of losing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other(String),
}

impl Finish {
    /// The wire's spelling. Unknown reasons survive as [`Finish::Other`] —
    /// a provider that invents one must not end a run with a parse error.
    pub fn parse(s: &str) -> Finish {
        match s {
            "stop" => Finish::Stop,
            "length" | "max_tokens" => Finish::Length,
            "tool_calls" | "function_call" => Finish::ToolCalls,
            "content_filter" => Finish::ContentFilter,
            other => Finish::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
            Finish::ContentFilter => "content_filter",
            Finish::Other(s) => s,
        }
    }
}

/// What one call cost.
///
/// The asymmetry to know, because it is not a bug: on the OpenAI wire `input`
/// already **includes** `cached` — the cached part is a discounted subset. On
/// Anthropic's, `input` **excludes** it and `cached` is added on top. Each
/// call is billed as reported, so a sum over calls is right either way; the
/// difference only matters when explaining the numbers to an operator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input: u32,
    pub output: u32,
    /// Prompt cache READ — a hit, billed at a discount.
    pub cached: u32,
    /// Prompt cache WRITE — populating the cache, billed above input rate.
    /// Anthropic only; stays 0 on the OpenAI wire, which does not bill it
    /// separately.
    pub cache_creation: u32,
    /// Reasoning tokens, for the models that report them.
    pub reasoning: u32,
}

impl Usage {
    /// Nothing was reported. Distinguishes "no usage on this frame" from
    /// "a turn that genuinely cost nothing", which does not happen.
    pub fn is_zero(&self) -> bool {
        *self == Usage::default()
    }
}

/// A completed chat turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub message: Message,
    /// The same calls as `message.tool_calls`, hoisted so the common caller —
    /// "did this turn ask for tools?" — does not reach through the message.
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    /// `None` when the provider did not say. Gateways sometimes do not.
    pub finish: Option<Finish>,
}

/// A streaming delta for one tool call's arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Which of the parallel calls this fragment belongs to.
    pub index: u32,
    pub id: String,
    pub name: String,
    /// The fragment, exactly as it arrived. Not JSON on its own.
    pub delta: String,
}

/// One frame of a streamed turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Chunk {
    /// Incremental text.
    pub delta: String,
    /// Complete tool calls, emitted once with the finish frame.
    pub tool_calls: Vec<ToolCall>,
    /// One argument fragment, as it arrives.
    pub tool_call_delta: Option<ToolCallDelta>,
    pub finish: Option<Finish>,
    /// What the turn has cost **so far**: a running total, never an increment,
    /// and a wire may report it more than once.
    ///
    /// **A caller takes the last one it saw and never a sum.** The OpenAI wire
    /// reports once, on a trailing frame; Anthropic's reports the input side
    /// at `message_start` — so a pane has the prompt's price before the answer
    /// starts — and the whole bill again on the finish. Adding those two up
    /// double-counts the prompt on one wire and not on the other, which is the
    /// same "a caller has to know which wire answered" failure the rest of
    /// this type exists to prevent.
    pub usage: Option<Usage>,
}

/// What a client is, for a pane and a log line. Never a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    /// `openai`, `azure`, `anthropic`, `typesafe`.
    pub wire: &'static str,
    /// The model name, or the Azure deployment.
    pub model: String,
    /// The endpoint as configured.
    pub endpoint: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_finish_reason_survives_as_itself() {
        assert_eq!(Finish::parse("stop"), Finish::Stop);
        assert_eq!(Finish::parse("tool_calls"), Finish::ToolCalls);
        // Anthropic's spellings land on the same arms.
        assert_eq!(Finish::parse("max_tokens"), Finish::Length);
        // And a reason nobody has seen yet is readable, not an error.
        assert_eq!(
            Finish::parse("guardrail_intervened"),
            Finish::Other("guardrail_intervened".into())
        );
        assert_eq!(
            Finish::parse("guardrail_intervened").as_str(),
            "guardrail_intervened"
        );
    }
}
