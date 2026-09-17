//! The HTTP both wires share: the URL, the client, the headers, and what a
//! non-2xx says.
//!
//! One seat each, because the rules here are the crate's and not a wire's. An
//! operator's endpoint keeps its own path on every wire; a header the card
//! carries is refused rather than dropped on every wire; and an error carries
//! the provider's own words, clipped, and nothing of the request — the rule
//! `providers.md` §5 states once and both wires obey.

use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;
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
/// **body's** message. The request never appears in the error; the
/// provider's request id and a 429's retry hint do, read off the response
/// headers before the body is consumed.
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
    let request_id = request_id(resp.headers());
    let retry_after = retry_after(resp.headers());
    let text = resp.text().await.unwrap_or_default();
    Err(Error::Api {
        status: status.as_u16(),
        message: api_message(&text, status.as_u16()),
        request_id,
        retry_after,
    })
}

/// The provider's id for the exchange. Each wire spells the header its own
/// way; the first one present wins, and none is a request id of `None`.
fn request_id(headers: &HeaderMap) -> Option<String> {
    ["x-typesafe-request-id", "request-id", "x-request-id"]
        .into_iter()
        .find_map(|name| headers.get(name))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// A 429's hint, the way TypeSafe's SDK reads it: `retry-after-ms` first,
/// then `Retry-After` in whole seconds. The HTTP-date form of the latter is
/// not parsed — nothing we speak sends it, and a wrong guess at a clock is
/// worse than no hint.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let text = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    if let Some(ms) = text("retry-after-ms").and_then(|s| s.trim().parse::<u64>().ok()) {
        return Some(Duration::from_millis(ms));
    }
    text("retry-after")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
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

/// What a non-2xx says, from the **body**. A body that is not the shape the
/// wire promises (an HTML error page from a proxy, say) is clipped and passed
/// through rather than swallowed — an operator debugging a gateway needs the
/// first line of it.
pub(crate) fn api_message(body: &str, status: u16) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body)
        && let Some(message) = error_message(&v)
    {
        return message;
    }
    let body = body.trim();
    if body.is_empty() {
        return format!("no body with the {status}");
    }
    clip(body)
}

/// The message inside an error body, across the shapes the wires answer
/// with. OpenAI and Anthropic: `{"error": {"message": …}}`. TypeSafe:
/// `{"detail": …}`, where `detail` is a string, an object with a `message`,
/// or — on a 422 — an array of `{loc, msg}` validation errors, joined the way
/// their SDK joins them so the two read the same in a log line.
fn error_message(v: &serde_json::Value) -> Option<String> {
    let non_empty = |s: &str| (!s.is_empty()).then(|| s.to_string());
    if let Some(s) = v.get("error").and_then(|e| e.as_str()) {
        return non_empty(s);
    }
    if let Some(s) = v.pointer("/error/message").and_then(|m| m.as_str()) {
        return non_empty(s);
    }
    if let Some(s) = v.get("message").and_then(|m| m.as_str()) {
        return non_empty(s);
    }
    let detail = v.get("detail")?;
    if let Some(s) = detail.as_str() {
        return non_empty(s);
    }
    if let Some(s) = detail.get("message").and_then(|m| m.as_str()) {
        return non_empty(s);
    }
    let errors = detail.as_array()?;
    let parts: Vec<String> = errors
        .iter()
        .filter_map(|e| {
            let msg = e.get("msg")?.as_str()?;
            let loc: Vec<&str> = e
                .get("loc")
                .and_then(|l| l.as_array())
                .map(|l| {
                    l.iter()
                        .filter_map(|x| x.as_str())
                        .filter(|x| *x != "body")
                        .collect()
                })
                .unwrap_or_default();
            Some(if loc.is_empty() {
                msg.to_string()
            } else {
                format!("{}: {msg}", loc.join("."))
            })
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join("; "))
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

    #[test]
    fn typesafes_detail_gives_up_its_message_in_each_of_its_shapes() {
        // A plain string.
        assert_eq!(
            api_message(r#"{"detail":"Invalid API key"}"#, 401),
            "Invalid API key"
        );
        // An object with a message.
        assert_eq!(
            api_message(r#"{"detail":{"message":"Overloaded","code":"busy"}}"#, 529),
            "Overloaded"
        );
        // A 422's validation array, joined the way their SDK joins it, with
        // the `body` root dropped from each path.
        assert_eq!(
            api_message(
                r#"{"detail":[
                    {"loc":["body","questions","tone","criteria"],"msg":"at least two levels","type":"value_error"},
                    {"loc":["body","model"],"msg":"field required","type":"missing"},
                    {"not":"a validation error"}
                ]}"#,
                422
            ),
            "questions.tone.criteria: at least two levels; model: field required"
        );
        // A body that is JSON but says nothing is passed through whole.
        assert_eq!(api_message(r#"{"detail":[]}"#, 422), r#"{"detail":[]}"#);
        assert_eq!(
            api_message(r#"{"error":{"message":""}}"#, 500),
            r#"{"error":{"message":""}}"#
        );
    }

    #[test]
    fn a_request_id_is_read_from_whichever_header_the_wire_uses() {
        let mut h = HeaderMap::new();
        assert_eq!(request_id(&h), None);
        h.insert("x-request-id", HeaderValue::from_static("req_openai"));
        assert_eq!(request_id(&h).as_deref(), Some("req_openai"));
        h.insert("request-id", HeaderValue::from_static("req_anthropic"));
        assert_eq!(request_id(&h).as_deref(), Some("req_anthropic"));
        h.insert("x-typesafe-request-id", HeaderValue::from_static("req_ts"));
        assert_eq!(request_id(&h).as_deref(), Some("req_ts"));
    }

    #[test]
    fn retry_after_prefers_milliseconds_and_ignores_a_date() {
        let mut h = HeaderMap::new();
        assert_eq!(retry_after(&h), None);
        h.insert(
            "retry-after",
            HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(retry_after(&h), None);
        h.insert("retry-after", HeaderValue::from_static("3"));
        assert_eq!(retry_after(&h), Some(Duration::from_secs(3)));
        h.insert("retry-after-ms", HeaderValue::from_static("250"));
        assert_eq!(retry_after(&h), Some(Duration::from_millis(250)));
    }
}
