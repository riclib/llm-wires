//! The HTTP both wires share: the URL, the client, the headers, and what a
//! non-2xx says.
//!
//! One seat each, because the rules here are the crate's and not a wire's. An
//! operator's endpoint keeps its own path on every wire; a header the card
//! carries is refused rather than dropped on every wire; and an error carries
//! the provider's own words, clipped, and nothing of the request — the rule
//! `providers.md` §5 states once and both wires obey.

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use wire_secret::Secret;

use crate::{Error, Headers, Result, tls};

/// `{endpoint}/{path}`, with the endpoint's own path kept.
///
/// A trailing slash on the endpoint is the operator's, not a second segment;
/// a query is appended, not replaced. The endpoint's path is a **prefix**: a
/// corporate gateway at `https://gw.corp/llm` must come out as `/llm/…`, which
/// is why the URL is built here rather than by an SDK that matches hardcoded
/// paths.
pub(crate) fn join(
    endpoint: &str,
    path: &str,
    query: Option<(&str, &str)>,
) -> Result<reqwest::Url> {
    let base = endpoint.trim().trim_end_matches('/');
    let mut url = reqwest::Url::parse(&format!("{base}/{path}")).map_err(|e| Error::Endpoint {
        endpoint: endpoint.to_string(),
        why: e.to_string(),
    })?;
    if let Some((k, v)) = query {
        url.query_pairs_mut().append_pair(k, v);
    }
    Ok(url)
}

/// The client, with the auth and gateway headers as defaults so that every
/// request — blocking and streaming — carries them without a second seat.
pub(crate) fn client(defaults: HeaderMap) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .use_preconfigured_tls(tls::client_config()?)
        .default_headers(defaults)
        .build()
        .map_err(Error::from)
}

/// The operator's extra headers. A name or value the HTTP layer refuses is a
/// card that needs fixing, so it is refused here rather than dropped: a
/// gateway that needs a header does not work without it, and a silent drop
/// makes that a 401 nobody can explain.
pub(crate) fn extra(wire: &'static str, headers: &Headers) -> Result<HeaderMap> {
    let mut map = HeaderMap::with_capacity(headers.len());
    for (k, v) in headers {
        let name = HeaderName::try_from(k.as_str()).map_err(|e| Error::BadHeader {
            wire,
            what: format!("header name {k:?}: {e}"),
        })?;
        let value = HeaderValue::try_from(v.as_str()).map_err(|e| Error::BadHeader {
            wire,
            what: format!("header {k}: {e}"),
        })?;
        map.insert(name, value);
    }
    Ok(map)
}

/// A key as a header value, with the scheme in front of it where the wire
/// wants one: `""` for Anthropic's `x-api-key` and Azure's `Api-Key`,
/// `"Bearer "` for OpenAI's `Authorization`.
///
/// **One function and not two**, so that a wire is named once and the
/// sensitive flag is a rule rather than a convention: there is no way to get a
/// credential into a header value that does not go through here and mark it,
/// and `http`'s own `Debug` — the one a middleware or a panic message reaches
/// for — therefore prints `Sensitive` for every wire, including the next one.
pub(crate) fn key_header(wire: &'static str, key: &Secret, prefix: &str) -> Result<HeaderValue> {
    let text = key.expose_str().map_err(|_| Error::KeyNotText(wire))?;
    let mut value =
        HeaderValue::try_from(format!("{prefix}{text}")).map_err(|_| Error::KeyNotText(wire))?;
    value.set_sensitive(true);
    Ok(value)
}

/// Send the body, and turn a non-2xx into [`Error::Api`] carrying the
/// **body's** message. The request never appears in the error.
pub(crate) async fn post_json<B: Serialize + ?Sized>(
    http: &reqwest::Client,
    url: &reqwest::Url,
    body: &B,
) -> Result<reqwest::Response> {
    let resp = http.post(url.clone()).json(body).send().await?;
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let text = resp.text().await.unwrap_or_default();
    Err(Error::Api {
        status: status.as_u16(),
        message: api_message(&text, status.as_u16()),
    })
}

/// A 200 that is not an event stream must not become an empty stream.
///
/// Framing a body with no `data:` lines produces no chunks, the stream ends
/// `Ok`, and the caller sees a turn that succeeded and said nothing — the
/// exact outcome a mid-stream `error` arm exists to prevent, arriving through
/// the door that arm does not watch. A gateway that answers 200 with a bare
/// JSON error, an HTML interstitial, or an ordinary completion body because it
/// quietly ignored `"stream": true` all land here.
pub(crate) async fn require_event_stream(resp: reqwest::Response) -> Result<reqwest::Response> {
    let kind = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    if kind.starts_with("text/event-stream") {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    Err(Error::Decode(format!(
        "a 200 that is not an event stream (content-type {kind:?}): {}",
        clip(&body)
    )))
}

// -------------------------------------------------------------- the error body

/// Both wires answer a failure the same way: `{"error": {"message": …}}`.
#[derive(Debug, Deserialize)]
struct ApiError {
    error: ApiErrorBody,
}

#[derive(Debug, Deserialize)]
struct ApiErrorBody {
    #[serde(default)]
    message: String,
}

/// What a non-2xx says, from the **body**. A body that is not the shape the
/// wire promises (an HTML error page from a proxy, say) is clipped and passed
/// through rather than swallowed — an operator debugging a gateway needs the
/// first line of it.
pub(crate) fn api_message(body: &str, status: u16) -> String {
    if let Ok(e) = serde_json::from_str::<ApiError>(body)
        && !e.error.message.is_empty()
    {
        return e.error.message;
    }
    let body = body.trim();
    if body.is_empty() {
        return format!("no body with the {status}");
    }
    clip(body)
}

/// Long bodies do not belong in an error that reaches a pane.
pub(crate) fn clip(s: &str) -> String {
    const MAX: usize = 400;
    if s.len() <= MAX {
        return s.to_string();
    }
    let mut end = MAX;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_endpoints_own_path_survives_the_join() {
        let cases = [
            (
                "https://api.openai.com/v1",
                "chat/completions",
                "https://api.openai.com/v1/chat/completions",
            ),
            (
                "https://api.openai.com/v1/",
                "chat/completions",
                "https://api.openai.com/v1/chat/completions",
            ),
            (
                "http://localhost:11434/v1",
                "chat/completions",
                "http://localhost:11434/v1/chat/completions",
            ),
            // A gateway behind a prefix: the prefix is kept, not replaced.
            (
                "https://gw.corp.com/llm/openai",
                "chat/completions",
                "https://gw.corp.com/llm/openai/chat/completions",
            ),
            // Anthropic's own path is the wire's, not the operator's.
            (
                "https://api.anthropic.com",
                "v1/messages",
                "https://api.anthropic.com/v1/messages",
            ),
            (
                "https://gw.corp.com/claude/",
                "v1/messages",
                "https://gw.corp.com/claude/v1/messages",
            ),
        ];
        for (endpoint, path, want) in cases {
            let got = join(endpoint, path, None).unwrap();
            assert_eq!(got.as_str(), want);
        }

        // Azure's whole URL, including the version in the query.
        let got = join(
            "https://gw.corp.com/openai-chat/",
            "openai/deployments/gpt-4o-mini/chat/completions",
            Some(("api-version", "2024-08-01-preview")),
        )
        .unwrap();
        assert_eq!(
            got.as_str(),
            "https://gw.corp.com/openai-chat/openai/deployments/gpt-4o-mini/chat/completions\
             ?api-version=2024-08-01-preview"
        );
    }

    #[test]
    fn an_endpoint_that_is_not_a_url_is_refused_by_name() {
        match join("not a url", "chat/completions", None) {
            Err(Error::Endpoint { endpoint, .. }) => assert_eq!(endpoint, "not a url"),
            other => panic!("expected Endpoint, got {other:?}"),
        }
    }

    #[test]
    fn an_error_body_gives_up_its_message_and_a_stray_page_is_clipped() {
        assert_eq!(
            api_message(
                r#"{"error":{"message":"Incorrect API key provided: sk-***","type":"invalid_request_error"}}"#,
                401
            ),
            "Incorrect API key provided: sk-***"
        );
        // Anthropic's shape is the same one, with its own envelope around it.
        assert_eq!(
            api_message(
                r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
                401
            ),
            "invalid x-api-key"
        );
        // A proxy's HTML is not JSON; the operator still needs to see it.
        assert_eq!(
            api_message("<html>502 Bad Gateway</html>", 502),
            "<html>502 Bad Gateway</html>"
        );
        assert_eq!(api_message("   ", 500), "no body with the 500");
        let long = "x".repeat(1000);
        assert_eq!(api_message(&long, 500).len(), 400 + "…".len());
    }
}
