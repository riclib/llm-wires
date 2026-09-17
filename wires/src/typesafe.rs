//! The TypeSafe System One wire.
//!
//! `POST {endpoint}/v1/systemone` with the key in `Authorization: Bearer`.
//! One `state` and a map of typed questions go out; one typed answer per
//! question comes back, each a distribution with a confidence. It is a
//! [`Judge`](crate::Judge) and not a [`Provider`](crate::Provider): there is
//! no message list, no tool, no stream and no generated text, and a
//! deployment that judges cannot chat.
//!
//! Hand-rolled from the published SDK's declarations rather than from the
//! docs' prose, because the SDK is what their server is tested against and
//! the two disagree in places (`instructions` is optional; a score's
//! criteria is an array, not a map). Where the SDK retries a 429 or a 5xx,
//! this wire does not — the run above it owns retries — and surfaces the
//! `retry-after` on the error instead.
//!
//! Not here, and deliberately: `GET /v1/models`, which is a card's model
//! picker and not a judgement.

mod convert;

use async_trait::async_trait;
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderName, HeaderValue, USER_AGENT};
use wire_secret::Secret;

use crate::{Headers, Info, Judge, Judgement, Result, Verdict, http};

pub(crate) const WIRE: &str = "typesafe";

/// What this crate calls itself to their server, where their SDK sends
/// `typesafe-sdk/<version>`. The same string on `User-Agent` and on
/// `X-TypeSafe-SDK`, as theirs is.
const IDENT: &str = concat!("llm-wires/", env!("CARGO_PKG_VERSION"));

const SDK_HEADER: HeaderName = HeaderName::from_static("x-typesafe-sdk");
const RUNTIME_HEADER: HeaderName = HeaderName::from_static("x-typesafe-runtime");

/// A client for one model.
///
/// No `Debug` derive and no key field, for the reason the chat clients have
/// neither: the credential reaches [`reqwest::Client`]'s default headers as a
/// value marked sensitive and is dropped, and a derived `Debug` would walk
/// those headers.
pub(crate) struct TypeSafe {
    http: reqwest::Client,
    /// The full URL, built once at construction.
    url: reqwest::Url,
    model: String,
    /// The endpoint as the operator wrote it, for [`Info`].
    endpoint: String,
}

impl TypeSafe {
    /// TypeSafe, and any gateway that speaks System One.
    ///
    /// The operator's headers go on first, so a card can name itself to a
    /// gateway or replace the `User-Agent`. The key goes on last and
    /// unconditionally: a header that could displace the credential is a
    /// 401 nobody can explain.
    pub(crate) fn system_one(
        endpoint: &str,
        model: &str,
        headers: &Headers,
        key: &Secret,
    ) -> Result<TypeSafe> {
        let url = http::join(endpoint, "v1/systemone", None)?;
        let mut map = http::extra(WIRE, headers)?;
        for (name, value) in [
            (ACCEPT, "application/json"),
            (USER_AGENT, IDENT),
            (SDK_HEADER, IDENT),
            (RUNTIME_HEADER, "rust"),
        ] {
            if !map.contains_key(&name) {
                map.insert(name, HeaderValue::from_static(value));
            }
        }
        map.insert(AUTHORIZATION, http::key_header(WIRE, key, "Bearer ")?);
        Ok(TypeSafe {
            http: http::client(map)?,
            url,
            model: model.to_string(),
            endpoint: endpoint.to_string(),
        })
    }
}

#[async_trait]
impl Judge for TypeSafe {
    async fn judge(&self, req: Judgement) -> Result<Verdict> {
        let body = convert::request(&req, &self.model)?;
        let text = http::post_json(&self.http, &self.url, &body)
            .await?
            .text()
            .await?;
        convert::response(&text, &req.questions)
    }

    fn info(&self) -> Info {
        Info {
            wire: WIRE,
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }
}

impl std::fmt::Debug for TypeSafe {
    /// Everything a log line may have, and nothing else.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeSafe")
            .field("wire", &WIRE)
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
        let c = TypeSafe::system_one(
            "https://api.typesafe.ai",
            "jev-latest",
            &Headers::new(),
            &Secret::from("ts-do-not-log-me"),
        )
        .unwrap();
        let shown = format!("{c:?}");
        assert!(!shown.contains("do-not-log-me"), "{shown}");
        assert!(shown.contains("jev-latest"), "{shown}");
        assert!(
            shown.contains("https://api.typesafe.ai/v1/systemone"),
            "{shown}"
        );
    }
}
