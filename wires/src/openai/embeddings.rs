//! The OpenAI embeddings wire.
//!
//! `POST {endpoint}/embeddings`, and the same body at Azure's URL. A separate
//! client from the chat one and not a method on it: an embedding deployment
//! cannot chat and a chat deployment cannot embed, so one interface would
//! force every implementation to carry a method that always errors — the rule
//! [`Judge`](crate::Judge) is the first instance of.
//!
//! It serves everything that speaks OpenAI's shape at another host, as the
//! chat wire does: a local ollama, a gateway, a self-hosted server in front of
//! a small open model. The endpoint is the caller's, there is no default here,
//! and the model is the client's.

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName};
use serde::{Deserialize, Serialize};
use wire_secret::Secret;

use crate::{Embed, EmbedRequest, EmbedResponse, Error, Headers, Info, Result, Usage, http};

/// A client for one embedding deployment.
///
/// No `Debug` derive and no key field, for the reason the chat client has
/// neither: the credential reaches [`reqwest::Client`]'s default headers as a
/// value marked sensitive and is dropped, and a derived `Debug` would walk
/// those headers.
pub(crate) struct Embeddings {
    http: reqwest::Client,
    /// The full URL, query included, built once at construction.
    url: reqwest::Url,
    /// The model name, or the Azure deployment — the same field on the wire.
    model: String,
    /// The endpoint as the operator wrote it, for [`Info`].
    endpoint: String,
    wire: &'static str,
}

impl Embeddings {
    /// OpenAI and everything that speaks its shape.
    pub(crate) fn openai(
        endpoint: &str,
        model: &str,
        headers: &Headers,
        key: &Secret,
    ) -> Result<Embeddings> {
        let url = http::join(endpoint, "embeddings", None)?;
        let mut map = http::extra("openai", headers)?;
        map.insert(AUTHORIZATION, http::key_header("openai", key, "Bearer ")?);
        Ok(Embeddings {
            http: http::client(map)?,
            url,
            model: model.to_string(),
            endpoint: endpoint.to_string(),
            wire: "openai",
        })
    }

    /// Azure, whose URL carries what the body carries elsewhere: the
    /// deployment in the path, the api-version in the query, the key in its
    /// own header. The deployment path is **appended** to the operator's own,
    /// as the chat wire appends it, so a gateway behind a prefix works.
    pub(crate) fn azure(
        endpoint: &str,
        deployment: &str,
        api_version: &str,
        key: &Secret,
    ) -> Result<Embeddings> {
        let path = format!("openai/deployments/{deployment}/embeddings");
        let url = http::join(endpoint, &path, Some(("api-version", api_version)))?;
        let mut map = HeaderMap::new();
        map.insert(
            HeaderName::from_static("api-key"),
            http::key_header("azure", key, "")?,
        );
        Ok(Embeddings {
            http: http::client(map)?,
            url,
            model: deployment.to_string(),
            endpoint: endpoint.to_string(),
            wire: "azure",
        })
    }
}

#[async_trait]
impl Embed for Embeddings {
    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse> {
        let wanted = req.inputs.len();
        self.check(&req)?;
        let body = Request {
            model: &self.model,
            input: &req.inputs,
            encoding_format: FLOAT,
            dimensions: req.dimensions,
        };
        let text = http::post_json(&self.http, &self.url, &body)
            .await?
            .text()
            .await?;
        response(&text, wanted)
    }

    fn info(&self) -> Info {
        Info {
            wire: self.wire,
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }
}

impl Embeddings {
    /// What the server would refuse anyway, refused here where the caller's
    /// own row number is still in hand. A 400 saying `$.input is invalid`
    /// about a batch of two thousand is not a sentence anybody can act on.
    fn check(&self, req: &EmbedRequest) -> Result<()> {
        let invalid = |what: String| Error::Invalid {
            wire: self.wire,
            what,
        };
        if req.inputs.is_empty() {
            return Err(invalid("an embed request with no inputs".into()));
        }
        if let Some(at) = req.inputs.iter().position(|s| s.is_empty()) {
            return Err(invalid(format!("input {at} is empty")));
        }
        if req.dimensions == Some(0) {
            return Err(invalid("dimensions must be at least 1".into()));
        }
        Ok(())
    }
}

impl std::fmt::Debug for Embeddings {
    /// Everything a log line may have, and nothing else. Not a derive, which
    /// would print the `reqwest::Client` and walk its default headers.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embeddings")
            .field("wire", &self.wire)
            .field("model", &self.model)
            .field("url", &self.url.as_str())
            .finish()
    }
}

// ---------------------------------------------------------------- the request

/// What this wire decodes, named in the request.
///
/// The field's default is `float` today, but the vendor SDKs ask for `base64`
/// and a gateway that copies them would answer a shape this wire does not
/// read. Asking for what we decode turns that into their 400 instead of our
/// parse error — and where a server ignores the field anyway, the answer is
/// still checked below.
const FLOAT: &str = "float";

/// The body, exactly as it goes out.
#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
    /// Omitted when the caller did not ask, so a wire that has never heard of
    /// it sees a body it knows.
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

// --------------------------------------------------------------- the response

#[derive(Debug, Deserialize)]
struct Response {
    #[serde(default)]
    data: Vec<Row>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct Row {
    /// Which input this is the vector for. The wire promises the array is in
    /// order and every server sends it that way; the index is what says so,
    /// and it is cheaper to obey it than to find out one day that a gateway
    /// batched the work across workers.
    #[serde(default)]
    index: usize,
    #[serde(default)]
    embedding: WireEmbedding,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WireEmbedding {
    Floats(Vec<f32>),
    /// A server that ignored `encoding_format` and packed the floats little-
    /// endian into base64. Its own arm so the operator is told exactly that,
    /// rather than reading serde's "invalid type: string".
    Packed(String),
}

impl Default for WireEmbedding {
    fn default() -> WireEmbedding {
        WireEmbedding::Floats(Vec::new())
    }
}

#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

/// The vectors, one per input and in the input's order.
///
/// Every way the answer can fail to be that is an [`Error::Decode`] naming
/// which: a row missing, a row twice, a row for an input that was not sent, a
/// width that disagrees with the rest. The caller is about to store these
/// against its own rows by position and in a fixed-width column, so an answer
/// that is nearly right is worse than no answer.
fn response(text: &str, wanted: usize) -> Result<EmbedResponse> {
    let resp: Response = serde_json::from_str(text)
        .map_err(|e| Error::Decode(format!("{e} in {}", http::clip(text))))?;
    if resp.data.len() != wanted {
        return Err(Error::Decode(format!(
            "{wanted} inputs went out and {} vectors came back",
            resp.data.len()
        )));
    }

    let mut slots: Vec<Option<Vec<f32>>> = vec![None; wanted];
    for row in resp.data {
        let floats = match row.embedding {
            WireEmbedding::Floats(v) => v,
            WireEmbedding::Packed(b64) => {
                return Err(Error::Decode(format!(
                    "the provider answered a base64 embedding of {} characters although the \
                     request asked for encoding_format={FLOAT}",
                    b64.len()
                )));
            }
        };
        let Some(slot) = slots.get_mut(row.index) else {
            return Err(Error::Decode(format!(
                "a vector at index {} with only {wanted} inputs",
                row.index
            )));
        };
        if slot.replace(floats).is_some() {
            return Err(Error::Decode(format!("two vectors at index {}", row.index)));
        }
    }

    let mut vectors = Vec::with_capacity(wanted);
    for (at, slot) in slots.into_iter().enumerate() {
        let Some(v) = slot else {
            return Err(Error::Decode(format!("no vector for input {at}")));
        };
        vectors.push(v);
    }

    if let Some(width) = vectors.first().map(Vec::len) {
        if width == 0 {
            return Err(Error::Decode("a vector with no dimensions in it".into()));
        }
        if let Some((at, got)) = vectors
            .iter()
            .enumerate()
            .find_map(|(at, v)| (v.len() != width).then_some((at, v.len())))
        {
            return Err(Error::Decode(format!(
                "the vectors are not one width: {width} at input 0 and {got} at input {at}"
            )));
        }
    }

    Ok(EmbedResponse {
        vectors,
        usage: resp.usage.map(usage).unwrap_or_default(),
    })
}

/// What the batch cost. Only the input side exists here — nothing is
/// generated — and a gateway that reports the total and not the prompt is
/// read as the same number, because for this endpoint it is.
fn usage(u: WireUsage) -> Usage {
    Usage {
        input: if u.prompt_tokens > 0 {
            u.prompt_tokens
        } else {
            u.total_tokens
        },
        ..Usage::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> Embeddings {
        Embeddings::openai(
            "https://api.openai.com/v1",
            "text-embedding-3-small",
            &Headers::new(),
            &Secret::from("sk-live-do-not-log-me"),
        )
        .unwrap()
    }

    #[test]
    fn the_key_does_not_print() {
        let shown = format!("{:?}", client());
        assert!(!shown.contains("sk-live"), "{shown}");
        assert!(shown.contains("text-embedding-3-small"), "{shown}");
        assert!(
            shown.contains("https://api.openai.com/v1/embeddings"),
            "{shown}"
        );
    }

    #[test]
    fn a_batch_the_server_would_refuse_is_refused_here_with_the_row_number() {
        let c = client();
        let cases = [
            (EmbedRequest::default(), "an embed request with no inputs"),
            (EmbedRequest::of(["a", "", "c"]), "input 1 is empty"),
            (
                EmbedRequest::one("a").with_dimensions(0),
                "dimensions must be at least 1",
            ),
        ];
        for (req, want) in cases {
            match c.check(&req) {
                Err(Error::Invalid { wire, what }) => {
                    assert_eq!(wire, "openai");
                    assert_eq!(what, want);
                }
                other => panic!("expected Invalid {want}, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_body_names_the_encoding_it_decodes_and_omits_what_was_not_asked() {
        let inputs = vec!["one".to_string(), "two".to_string()];
        let body = Request {
            model: "text-embedding-3-small",
            input: &inputs,
            encoding_format: FLOAT,
            dimensions: None,
        };
        assert_eq!(
            serde_json::to_value(&body).unwrap(),
            serde_json::json!({
                "model": "text-embedding-3-small",
                "input": ["one", "two"],
                "encoding_format": "float",
            })
        );
    }

    #[test]
    fn an_answer_that_is_not_one_vector_per_input_is_named_not_guessed() {
        let cases = [
            (
                r#"{"data":[{"index":0,"embedding":[0.1,0.2]}]}"#,
                2,
                "2 inputs went out and 1 vectors came back",
            ),
            (
                r#"{"data":[{"index":0,"embedding":[0.1]},{"index":0,"embedding":[0.2]}]}"#,
                2,
                "two vectors at index 0",
            ),
            (
                r#"{"data":[{"index":0,"embedding":[0.1]},{"index":7,"embedding":[0.2]}]}"#,
                2,
                "a vector at index 7 with only 2 inputs",
            ),
            (
                r#"{"data":[{"index":0,"embedding":[0.1,0.2]},{"index":1,"embedding":[0.3]}]}"#,
                2,
                "the vectors are not one width: 2 at input 0 and 1 at input 1",
            ),
            (
                r#"{"data":[{"index":0,"embedding":[]}]}"#,
                1,
                "a vector with no dimensions in it",
            ),
            (
                r#"{"data":[{"index":0,"embedding":"Zm9vYmFy"}]}"#,
                1,
                "the provider answered a base64 embedding of 8 characters although the \
                 request asked for encoding_format=float",
            ),
        ];
        for (body, wanted, want) in cases {
            match response(body, wanted) {
                Err(Error::Decode(why)) => assert_eq!(why, want),
                other => panic!("expected Decode {want}, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_total_without_a_prompt_count_is_still_the_input_side() {
        assert_eq!(
            usage(WireUsage {
                prompt_tokens: 0,
                total_tokens: 8,
            }),
            Usage {
                input: 8,
                ..Usage::default()
            }
        );
        assert_eq!(
            usage(WireUsage {
                prompt_tokens: 8,
                total_tokens: 8,
            })
            .input,
            8
        );
    }
}
