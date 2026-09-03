//! The LLM wires.
//!
//! One trait, one set of types, one switch. A caller names a [`Wire`] and a
//! key and gets back a `Box<dyn Provider>` that can be asked a question:
//!
//! ```no_run
//! # async fn go() -> llm_wires::Result<()> {
//! use llm_wires::{ChatRequest, Wire};
//!
//! let client = llm_wires::build(
//!     Wire::openai("https://api.openai.com/v1", "gpt-4o-mini"),
//!     Some(wire_secret::Secret::from("sk-…")),
//! )?;
//! let answer = client.chat(ChatRequest::ask("You are terse.", "Say hi")).await?;
//! println!("{}", answer.message.content);
//! # Ok(()) }
//! ```
//!
//! ## What this crate is not
//!
//! It knows **wires**, not kinds. There is no notion that `ollama` exists: an
//! ollama deployment is the OpenAI wire at a different endpoint with the
//! constant key `"ollama"`, and knowing that is the caller's job. The fence is
//! a Cargo edge — this crate depends on `wire-secret` and nothing else of its
//! own — because the wire drags HTTP, TLS and SSE in with it, and a caller that
//! only wants to hold a record should not have to link them.
//!
//! ## The stream has no `close`
//!
//! Go's `ChatStream` had `Recv()` (which took no context, so nothing could
//! interrupt it) and `Close()` (which for a long time only flipped a flag,
//! leaking the response body and a pooled connection on every early return).
//! Neither bug is writable here: [`Provider::chat_stream`] hands back a
//! `Stream`, so a caller `select!`s on it against a cancellation the way it
//! would any future, and **dropping the stream drops the response body**,
//! which is what closes the connection. There is nothing to remember to call.

mod anthropic;
mod error;
mod http;
mod openai;
mod sse;
#[cfg(feature = "test-support")]
pub mod testing;
mod tls;
mod types;

use std::collections::BTreeMap;
use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use wire_secret::Secret;

pub use error::{Error, Result};
pub use types::{
    ChatRequest, ChatResponse, Chunk, Finish, Info, Message, Role, Tool, ToolCall, ToolCallDelta,
    Usage,
};

/// A streamed turn: chunks in order, then the end.
///
/// Boxed because [`Provider`] is a trait object — the registry above this
/// crate holds one client per configured model and calls them through `dyn`.
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<Chunk>> + Send>>;

/// One deployment, asked a question.
///
/// A client is one wire at one endpoint for one model: the model is not a
/// request field, so a caller holding a client cannot address a model the
/// operator did not configure.
#[async_trait]
pub trait Provider: Send + Sync {
    /// The whole turn, when it is done.
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse>;

    /// The turn as it arrives. Dropping the returned stream closes the body.
    async fn chat_stream(&self, req: ChatRequest) -> Result<ChunkStream>;

    /// What this client is, for a pane and a log line. Never a key.
    fn info(&self) -> Info;
}

/// Extra headers sent on every request, in a stable order.
///
/// Sorted rather than insertion-ordered so a request is the same bytes twice —
/// which is what makes the request itself pinnable. What they are for: an
/// OpenAI-compatible gateway that needs one header and no new code (Bedrock's
/// `OpenAI-Project`, OpenRouter's `HTTP-Referer` / `X-Title`).
pub type Headers = BTreeMap<String, String>;

/// Which wire, where, and speaking for what.
///
/// The kinds an operator picks in the pane — `openai`, `azure`, `ollama`,
/// `openrouter`, `anthropic` — collapse onto these three; the mapping lives
/// in `domains::llm`, one level up, where the card is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Wire {
    /// `POST {endpoint}/chat/completions` with a bearer key. Serves OpenAI
    /// itself, a local ollama, OpenRouter, and any gateway that speaks the
    /// same shape.
    ///
    /// **The caller supplies the endpoint; there is no default here.** Go
    /// treated an empty endpoint as "the SDK's base URL" and filled
    /// `https://api.openai.com` back in on the way out, which put one kind's
    /// address inside a switch that is supposed to know only wires.
    /// `domains::llm` already has the per-kind answer —
    /// `SubKind::default_endpoint` behind `effective_endpoint()` — so an
    /// empty endpoint reaching this far is a card that was never resolved,
    /// and [`Error::Missing`] says so at `build` rather than at the socket.
    OpenAi {
        endpoint: String,
        model: String,
        headers: Headers,
    },
    /// The same body at Azure's URL: the deployment is in the path, the
    /// api-version in the query, and the key in `Api-Key` rather than
    /// `Authorization`.
    Azure {
        endpoint: String,
        deployment: String,
        api_version: String,
    },
    /// `POST {endpoint}/v1/messages` with the key in `x-api-key`. Anthropic's
    /// own API, and any gateway that speaks Messages at another host — the
    /// operator's `headers` reach every request, `anthropic-version` among
    /// them if the card spells it.
    Anthropic {
        endpoint: String,
        model: String,
        headers: Headers,
    },
}

impl Wire {
    /// The name that reaches [`Info::wire`] and an error message.
    pub fn name(&self) -> &'static str {
        match self {
            Wire::OpenAi { .. } => "openai",
            Wire::Azure { .. } => "azure",
            Wire::Anthropic { .. } => "anthropic",
        }
    }

    /// The common case: an endpoint and a model, no gateway headers.
    pub fn openai(endpoint: impl Into<String>, model: impl Into<String>) -> Wire {
        Wire::OpenAi {
            endpoint: endpoint.into(),
            model: model.into(),
            headers: Headers::new(),
        }
    }

    /// The common case for Anthropic.
    pub fn anthropic(endpoint: impl Into<String>, model: impl Into<String>) -> Wire {
        Wire::Anthropic {
            endpoint: endpoint.into(),
            model: model.into(),
            headers: Headers::new(),
        }
    }
}

/// Azure's version when the card leaves it blank. Go's default, kept: an
/// operator who has not thought about it gets a version that serves tools.
pub const AZURE_DEFAULT_API_VERSION: &str = "2024-08-01-preview";

/// The ONE switch over wires.
///
/// `key` is a [`wire_secret::Secret`] and not a `String` so that a client cannot
/// carry a key anything could print: the body reaches a header value marked
/// sensitive and is never cloned into the struct. `None` is refused for every
/// wire — a local ollama is served by passing the constant `"ollama"`, which
/// is what the server accepts and ignores, so "needs no credential" is a
/// sentence `domains::llm` says about a card and not one this crate has to
/// have an arm for.
pub fn build(wire: Wire, key: Option<Secret>) -> Result<Box<dyn Provider>> {
    let name = wire.name();
    let key = key.ok_or(Error::NoKey(name))?;
    match wire {
        Wire::OpenAi {
            endpoint,
            model,
            headers,
        } => {
            require(name, "endpoint", &endpoint)?;
            require(name, "model", &model)?;
            Ok(Box::new(openai::OpenAi::chat_completions(
                &endpoint, &model, &headers, &key,
            )?))
        }
        Wire::Azure {
            endpoint,
            deployment,
            api_version,
        } => {
            require(name, "endpoint", &endpoint)?;
            require(name, "deployment", &deployment)?;
            let version = if api_version.trim().is_empty() {
                AZURE_DEFAULT_API_VERSION
            } else {
                api_version.trim()
            };
            Ok(Box::new(openai::OpenAi::azure(
                &endpoint,
                &deployment,
                version,
                &key,
            )?))
        }
        Wire::Anthropic {
            endpoint,
            model,
            headers,
        } => {
            require(name, "endpoint", &endpoint)?;
            require(name, "model", &model)?;
            Ok(Box::new(anthropic::Anthropic::messages(
                &endpoint, &model, &headers, &key,
            )?))
        }
    }
}

fn require(wire: &'static str, field: &'static str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::Missing { wire, field });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_wire_needs_a_key() {
        for wire in [
            Wire::openai("https://api.openai.com/v1", "gpt-4o"),
            Wire::Azure {
                endpoint: "https://x.openai.azure.com".into(),
                deployment: "gpt-4o".into(),
                api_version: String::new(),
            },
            Wire::anthropic("https://api.anthropic.com", "claude-sonnet-4"),
        ] {
            let name = wire.name();
            match build(wire, None) {
                Err(Error::NoKey(got)) => assert_eq!(got, name),
                other => panic!("{name}: expected NoKey, got {other:?}", other = other.err()),
            }
        }
    }

    #[test]
    fn ollama_is_the_openai_wire_with_a_constant_key() {
        // The whole of "a local ollama needs no credential", from this crate's
        // side: there is no arm for it.
        let client = build(
            Wire::openai("http://localhost:11434/v1", "llama3.2:3b"),
            Some(Secret::from("ollama")),
        )
        .expect("a local ollama builds");
        assert_eq!(client.info().wire, "openai");
        assert_eq!(client.info().model, "llama3.2:3b");
        assert_eq!(client.info().endpoint, "http://localhost:11434/v1");
    }

    #[test]
    fn each_wire_reports_what_it_was_built_from() {
        let client = build(
            Wire::anthropic("https://api.anthropic.com", "claude-sonnet-4"),
            Some(Secret::from("sk-ant-x")),
        )
        .expect("the anthropic wire is built");
        assert_eq!(client.info().wire, "anthropic");
        assert_eq!(client.info().model, "claude-sonnet-4");
        assert_eq!(client.info().endpoint, "https://api.anthropic.com");
    }

    #[test]
    fn a_wire_missing_a_field_is_refused_before_a_socket_is_opened() {
        let cases: Vec<(Wire, &str)> = vec![
            (Wire::openai("", "gpt-4o"), "endpoint"),
            (Wire::anthropic("", "claude-sonnet-4"), "endpoint"),
            (Wire::anthropic("https://api.anthropic.com", ""), "model"),
            (Wire::openai("https://api.openai.com/v1", "  "), "model"),
            (
                Wire::Azure {
                    // Azure's default endpoint is empty on the card: there is
                    // no host to guess, so it must be asked for.
                    endpoint: String::new(),
                    deployment: "gpt-4o".into(),
                    api_version: String::new(),
                },
                "endpoint",
            ),
            (
                Wire::Azure {
                    endpoint: "https://x.openai.azure.com".into(),
                    deployment: String::new(),
                    api_version: String::new(),
                },
                "deployment",
            ),
        ];
        for (wire, want) in cases {
            match build(wire, Some(Secret::from("k"))) {
                Err(Error::Missing { field, .. }) => assert_eq!(field, want),
                other => panic!("expected Missing {want}, got {:?}", other.err()),
            }
        }
    }
}
