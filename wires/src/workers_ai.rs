//! The Cloudflare Workers AI wire, speaking Clairic.
//!
//! `POST {endpoint}/accounts/{account}/ai/run/{model}` with the token in
//! `Authorization: Bearer`, where the endpoint an operator types is
//! `https://api.cloudflare.com/client/v4` — the version is Cloudflare's whole
//! API's and not this route's, so it is the operator's, as OpenAI's `/v1` is.
//! A state and a map of typed fields go out; one typed answer per field comes
//! back inside Cloudflare's `{result, success, errors, messages}` envelope,
//! which is unwrapped here. It is an [`Extract`](crate::Extract) and nothing
//! else: there is no message list and no generated text.
//!
//! Hand-rolled from the model's published JSON schemas rather than from the
//! launch post, which leaves names out; see [`crate::Extraction`] for the
//! three it got wrong by omission.
//!
//! Not here, and deliberately: Clef (`@cf/cloudflare/clef`), Cloudflare's
//! decision model on the same route. It would be a [`Judge`](crate::Judge)
//! over this wire — a second arm in `build_judge` and a second converter —
//! and is that variant's job, not this one's.

mod convert;

use async_trait::async_trait;
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderValue, USER_AGENT};
use wire_secret::Secret;

use crate::{Extract, Extracted, Extraction, Headers, Info, Result, http};

pub(crate) const WIRE: &str = "workers_ai";

const IDENT: &str = concat!("llm-wires/", env!("CARGO_PKG_VERSION"));

/// A client for one model in one account.
///
/// No `Debug` derive and no key field, for the reason the other clients have
/// neither.
pub(crate) struct WorkersAi {
    http: reqwest::Client,
    /// The full URL, built once at construction.
    url: reqwest::Url,
    model: String,
    /// The endpoint as the operator wrote it, for [`Info`].
    endpoint: String,
}

impl WorkersAi {
    /// Workers AI's REST route, and any gateway that serves the same path.
    ///
    /// The operator's headers go on first; the token goes on last and
    /// unconditionally, as on every wire.
    pub(crate) fn run(
        endpoint: &str,
        account: &str,
        model: &str,
        headers: &Headers,
        key: &Secret,
    ) -> Result<WorkersAi> {
        // The model's `@cf/…/…` is a path of its own on Cloudflare's side
        // and goes in as written: its slashes are segments, not escapes.
        let url = http::join(
            endpoint,
            &format!("accounts/{}/ai/run/{}", account.trim(), model.trim()),
            None,
        )?;
        let mut map = http::extra(WIRE, headers)?;
        for (name, value) in [(ACCEPT, "application/json"), (USER_AGENT, IDENT)] {
            if !map.contains_key(&name) {
                map.insert(name, HeaderValue::from_static(value));
            }
        }
        map.insert(AUTHORIZATION, http::key_header(WIRE, key, "Bearer ")?);
        Ok(WorkersAi {
            http: http::client(map)?,
            url,
            model: model.to_string(),
            endpoint: endpoint.to_string(),
        })
    }
}

#[async_trait]
impl Extract for WorkersAi {
    async fn extract(&self, req: Extraction) -> Result<Extracted> {
        let body = convert::request(&req, &self.model)?;
        let resp = http::post_json(&self.http, &self.url, &body).await?;
        let status = resp.status().as_u16();
        let text = resp.text().await?;
        convert::response(&text, status, &req)
    }

    fn info(&self) -> Info {
        Info {
            wire: WIRE,
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }
}

impl std::fmt::Debug for WorkersAi {
    /// Everything a log line may have, and nothing else.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkersAi")
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
    fn the_key_does_not_print_and_the_model_is_the_path() {
        let c = WorkersAi::run(
            "https://api.cloudflare.com/client/v4/",
            "acc123",
            "@cf/cloudflare/clairic",
            &Headers::new(),
            &Secret::from("cf-do-not-log-me"),
        )
        .unwrap();
        let shown = format!("{c:?}");
        assert!(!shown.contains("do-not-log-me"), "{shown}");
        assert!(
            shown.contains(
                "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run/@cf/cloudflare/clairic"
            ),
            "{shown}"
        );
    }
}
