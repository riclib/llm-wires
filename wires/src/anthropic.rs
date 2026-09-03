//! The Anthropic messages wire.
//!
//! `POST {endpoint}/v1/messages`, the key in `x-api-key`, and
//! `anthropic-version` on every request. It serves the one kind the OpenAI
//! shape cannot: Anthropic's own API, and any gateway that speaks Messages at
//! another host.
//!
//! Hand-rolled rather than an SDK, for the reason the OpenAI wire is: the
//! subset we speak is one request and two response shapes, and Go's
//! `anthropic/convert.go` and `anthropic/stream.go` document the mapping
//! exactly. What Go got from the SDK and this file spells out instead is the
//! streaming accumulator — the block table that turns `input_json_delta`
//! fragments into a complete tool call.
//!
//! Not here, and deliberately: embeddings (Anthropic has none) and
//! extended-thinking blocks (a `thinking` block would be a fourth content kind
//! and a `Chunk` field nothing above this crate reads yet).

mod convert;
mod stream;

use async_trait::async_trait;
use reqwest::header::{HeaderName, HeaderValue};
use wire_secret::Secret;

use crate::{ChatRequest, ChatResponse, ChunkStream, Headers, Info, Provider, Result, http};

/// The version this wire speaks, sent on every request.
///
/// Anthropic requires it and has no default: a request without it is a 400.
/// An operator whose gateway pins another version can spell `anthropic-version`
/// in the card's headers and it wins — see [`Anthropic::messages`].
pub(crate) const VERSION: &str = "2023-06-01";

/// `max_tokens` when the request does not say.
///
/// Required by this wire — there is no "as many as it takes" — so a caller
/// that did not think about it still gets a turn rather than a 400. Go's
/// number, kept.
///
/// It **assumes a current-generation model**: 16384 is above the output cap of
/// the older ones, so a card naming one of those turns this default into the
/// 400 it exists to avoid. The fix is a `max_tokens` on the card, because
/// which cap applies is a fact about a *model* and this crate knows wires.
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 16384;

const API_KEY: HeaderName = HeaderName::from_static("x-api-key");
const VERSION_HEADER: HeaderName = HeaderName::from_static("anthropic-version");

/// A client for one model.
///
/// No `Debug` derive and no key field, for the reason the OpenAI client has
/// neither: the credential reaches [`reqwest::Client`]'s default headers as a
/// value marked sensitive and is dropped, and a derived `Debug` would walk
/// those headers.
pub(crate) struct Anthropic {
    http: reqwest::Client,
    /// The full URL, built once at construction.
    url: reqwest::Url,
    model: String,
    /// The endpoint as the operator wrote it, for [`Info`].
    endpoint: String,
}

impl Anthropic {
    /// Anthropic, and any gateway that speaks Messages.
    ///
    /// The operator's headers go on first, so a gateway pinned to a different
    /// `anthropic-version` can say so on the card. The key goes on last and
    /// unconditionally: a header that could displace the credential is a 401
    /// nobody can explain.
    pub(crate) fn messages(
        endpoint: &str,
        model: &str,
        headers: &Headers,
        key: &Secret,
    ) -> Result<Anthropic> {
        let url = http::join(endpoint, "v1/messages", None)?;
        let mut map = http::extra("anthropic", headers)?;
        if !map.contains_key(&VERSION_HEADER) {
            map.insert(VERSION_HEADER, HeaderValue::from_static(VERSION));
        }
        map.insert(API_KEY, http::key_header("anthropic", key, "")?);
        Ok(Anthropic {
            http: http::client(map)?,
            url,
            model: model.to_string(),
            endpoint: endpoint.to_string(),
        })
    }

    /// Send the body. A non-2xx becomes `Error::Api` carrying the **body's**
    /// message, in the one seat both wires share.
    async fn send(&self, body: &convert::Request) -> Result<reqwest::Response> {
        http::post_json(&self.http, &self.url, body).await
    }
}

#[async_trait]
impl Provider for Anthropic {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        let body = convert::request(req, &self.model, false)?;
        let text = self.send(&body).await?.text().await?;
        convert::response(&text)
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<ChunkStream> {
        let body = convert::request(req, &self.model, true)?;
        let resp = http::require_event_stream(self.send(&body).await?).await?;
        Ok(stream::chunks(resp))
    }

    fn info(&self) -> Info {
        Info {
            wire: "anthropic",
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }
}

impl std::fmt::Debug for Anthropic {
    /// Everything a log line may have, and nothing else.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Anthropic")
            .field("wire", &"anthropic")
            .field("model", &self.model)
            .field("url", &self.url.as_str())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_does_not_print() {
        let c = Anthropic::messages(
            "https://api.anthropic.com",
            "claude-sonnet-4",
            &Headers::new(),
            &Secret::from("sk-ant-do-not-log-me"),
        )
        .unwrap();
        let shown = format!("{c:?}");
        assert!(!shown.contains("sk-ant"), "{shown}");
        assert!(shown.contains("claude-sonnet-4"), "{shown}");
        assert!(
            shown.contains("https://api.anthropic.com/v1/messages"),
            "{shown}"
        );
    }
}
