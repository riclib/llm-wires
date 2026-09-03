//! The OpenAI chat-completions wire.
//!
//! `POST {endpoint}/chat/completions`, and the same body at Azure's URL. It
//! serves four of the five kinds an operator can pick — everything but
//! Anthropic — because ollama, OpenRouter and every gateway anyone has asked
//! for speak this shape at a different host.
//!
//! Hand-rolled rather than an SDK: the subset we speak is one request and two
//! response shapes, and the vendor crates are third-party, large, and move
//! their types on a schedule that is not ours. The mapping is Go's
//! `openai/convert.go` and `openai/stream.go`, ported.
//!
//! Not here, and deliberately: `/v1/responses` (a second wire, and the newest
//! models need it — a later ticket), and `/v1/embeddings` (a different
//! interface entirely; an embedding deployment cannot chat).

mod convert;
mod stream;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName};
use wire_secret::Secret;

use crate::{ChatRequest, ChatResponse, ChunkStream, Headers, Info, Provider, Result, http};

/// A client for one deployment.
///
/// No `Debug` derive, and no key field: the credential body reaches
/// [`reqwest::Client`]'s default headers as a value marked sensitive and is
/// dropped. The hand-written `Debug` below prints what a log line may have.
pub(crate) struct OpenAi {
    http: reqwest::Client,
    /// The full URL, query included. Built once at construction so a request
    /// is a `post(url.clone())` and the path rules live in one place.
    url: reqwest::Url,
    /// The model name, or the Azure deployment — the same field on the wire.
    model: String,
    /// The endpoint as the operator wrote it, for [`Info`].
    endpoint: String,
    wire: &'static str,
}

impl OpenAi {
    /// OpenAI and everything that speaks its shape.
    pub(crate) fn chat_completions(
        endpoint: &str,
        model: &str,
        headers: &Headers,
        key: &Secret,
    ) -> Result<OpenAi> {
        let url = http::join(endpoint, "chat/completions", None)?;
        let mut map = http::extra("openai", headers)?;
        map.insert(AUTHORIZATION, http::key_header("openai", key, "Bearer ")?);
        Ok(OpenAi {
            http: http::client(map)?,
            url,
            model: model.to_string(),
            endpoint: endpoint.to_string(),
            wire: "openai",
        })
    }

    /// Azure, whose URL carries what the body carries elsewhere.
    ///
    /// The deployment name is in the path and the api-version in the query,
    /// and the endpoint may have a proxy prefix (`https://gw.corp/openai-chat`),
    /// so the deployment path is **appended** to whatever the operator wrote
    /// rather than replacing its path. Go got this by building the URL by
    /// hand for exactly the same reason: the SDK's Azure middleware matched
    /// hardcoded paths and broke on a prefixed gateway.
    pub(crate) fn azure(
        endpoint: &str,
        deployment: &str,
        api_version: &str,
        key: &Secret,
    ) -> Result<OpenAi> {
        let path = format!("openai/deployments/{deployment}/chat/completions");
        let url = http::join(endpoint, &path, Some(("api-version", api_version)))?;
        let mut map = HeaderMap::new();
        // Azure names the header itself and does not take a bearer.
        map.insert(
            HeaderName::from_static("api-key"),
            http::key_header("azure", key, "")?,
        );
        Ok(OpenAi {
            http: http::client(map)?,
            url,
            model: deployment.to_string(),
            endpoint: endpoint.to_string(),
            wire: "azure",
        })
    }

    /// Send the body. A non-2xx becomes [`Error::Api`] carrying the **body's**
    /// message, in the one seat both wires share.
    async fn send(&self, body: &convert::Request) -> Result<reqwest::Response> {
        http::post_json(&self.http, &self.url, body).await
    }
}

#[async_trait]
impl Provider for OpenAi {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        let body = convert::request(req, &self.model, false);
        let text = self.send(&body).await?.text().await?;
        convert::response(&text)
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<ChunkStream> {
        let body = convert::request(req, &self.model, true);
        let resp = http::require_event_stream(self.send(&body).await?).await?;
        Ok(stream::chunks(resp))
    }

    fn info(&self) -> Info {
        Info {
            wire: self.wire,
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }
}

impl std::fmt::Debug for OpenAi {
    /// Everything a log line may have, and nothing else. Not a derive: a
    /// derive would print `http`, and a `reqwest::Client`'s own `Debug` walks
    /// its default headers.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAi")
            .field("wire", &self.wire)
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
        // Both halves of the guard. First: the header value the key becomes
        // is marked sensitive, so http's own Debug — the one a middleware or
        // a panic message would reach for — prints Sensitive, not the key.
        let v =
            http::key_header("openai", &Secret::from("sk-live-do-not-log-me"), "Bearer ").unwrap();
        assert_eq!(format!("{v:?}"), "Sensitive");
        assert!(v.is_sensitive());
        let v = http::key_header("azure", &Secret::from("azure-live-key"), "").unwrap();
        assert_eq!(format!("{v:?}"), "Sensitive");

        // Second: the client itself. Not a derive, so the reqwest::Client —
        // whose Debug walks its default headers — is not in it at all.
        let c = OpenAi::chat_completions(
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            &Headers::new(),
            &Secret::from("sk-live-do-not-log-me"),
        )
        .unwrap();
        let shown = format!("{c:?}");
        assert!(!shown.contains("sk-live"), "{shown}");
        assert!(shown.contains("gpt-4o-mini"), "{shown}");
    }
}
