//! The OpenAI embeddings wire against a real socket.
//!
//! The listener is `llm_wires::testing`, shared with the chat wires' pins and
//! with the judge's: what these are about is the **bytes** — which path, which
//! headers, which JSON — and a fake client can only ever agree with the code
//! that built it.

use std::collections::BTreeMap;

use llm_wires::testing::{Reply, closed_endpoint, fixture};
use llm_wires::{EmbedRequest, Error, Usage, Wire};
use serde_json::json;
use wire_secret::Secret;

// ----------------------------------------------------------------- the pins

#[tokio::test]
async fn the_request_carries_the_bearer_the_gateway_headers_and_the_batch() {
    let f = fixture(Reply::Json(TWO)).await;
    let mut headers = BTreeMap::new();
    headers.insert("X-Title".to_string(), "solid".to_string());

    let embed = llm_wires::build_embed(
        Wire::OpenAi {
            endpoint: f.endpoint(),
            model: "text-embedding-3-small".into(),
            headers,
        },
        Some(Secret::from("sk-test-key")),
    )
    .unwrap();

    let answer = embed
        .embed(EmbedRequest::of(["the first row", "the second row"]).with_dimensions(4))
        .await
        .unwrap();
    assert_eq!(
        answer.vectors,
        vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]]
    );
    assert_eq!(answer.dimensions(), Some(4));
    // Embeddings bill the input side and nothing else.
    assert_eq!(
        answer.usage,
        Usage {
            input: 11,
            ..Usage::default()
        }
    );

    let seen = f.request();
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.path, "/v1/embeddings");
    assert_eq!(seen.query, "");
    assert_eq!(seen.header("authorization"), Some("Bearer sk-test-key"));
    assert_eq!(seen.header("x-title"), Some("solid"));
    assert_eq!(seen.header("content-type"), Some("application/json"));
    assert_eq!(
        seen.json(),
        json!({
            "model": "text-embedding-3-small",
            "input": ["the first row", "the second row"],
            "encoding_format": "float",
            "dimensions": 4,
        })
    );
    assert_eq!(embed.info().wire, "openai");
    assert_eq!(embed.info().model, "text-embedding-3-small");
}

#[tokio::test]
async fn a_width_nobody_asked_for_is_not_in_the_body() {
    // The model's own width is the default, and the field a gateway may never
    // have heard of is not sent to find out.
    let f = fixture(Reply::Json(ONE)).await;
    let embed = llm_wires::build_embed(
        Wire::openai(f.endpoint_at("/llm/openai"), "nomic-embed-text"),
        Some(Secret::from("ollama")),
    )
    .unwrap();
    embed.embed(EmbedRequest::one("a query")).await.unwrap();

    let seen = f.request();
    // A gateway's prefix is kept, as it is on every other wire.
    assert_eq!(seen.path, "/llm/openai/embeddings");
    assert_eq!(
        seen.json(),
        json!({
            "model": "nomic-embed-text",
            "input": ["a query"],
            "encoding_format": "float",
        })
    );
}

#[tokio::test]
async fn the_vectors_come_back_in_the_order_the_inputs_went_out() {
    // The whole of "one vector per input, in the same order": the caller pairs
    // these with its own rows by position, so a server that answers out of
    // order — or a gateway that split the batch across workers — must not put
    // a vector against the wrong text. The index is obeyed, not assumed.
    let f = fixture(Reply::Json(SHUFFLED)).await;
    let embed = llm_wires::build_embed(
        Wire::openai(f.endpoint(), "text-embedding-3-small"),
        Some(Secret::from("sk-test-key")),
    )
    .unwrap();

    let answer = embed
        .embed(EmbedRequest::of(["zero", "one", "two"]))
        .await
        .unwrap();
    assert_eq!(
        answer.vectors,
        vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]]
    );
}

#[tokio::test]
async fn azure_puts_the_deployment_in_the_path_and_the_key_in_its_own_header() {
    let f = fixture(Reply::Json(ONE)).await;
    let embed = llm_wires::build_embed(
        Wire::Azure {
            endpoint: f.endpoint_at("/openai-embed"),
            deployment: "text-embedding-ada-002".into(),
            api_version: String::new(),
        },
        Some(Secret::from("azure-key")),
    )
    .unwrap();
    embed.embed(EmbedRequest::one("hello")).await.unwrap();

    let seen = f.request();
    assert_eq!(
        seen.path,
        "/openai-embed/openai/deployments/text-embedding-ada-002/embeddings"
    );
    // The card left the version blank, so the wire's default answers.
    assert_eq!(seen.query, "api-version=2024-08-01-preview");
    assert_eq!(seen.header("api-key"), Some("azure-key"));
    assert_eq!(seen.header("authorization"), None);
    assert_eq!(seen.json()["model"], json!("text-embedding-ada-002"));
    assert_eq!(embed.info().wire, "azure");
}

#[tokio::test]
async fn a_batch_the_server_would_refuse_never_reaches_a_socket() {
    // Bound and dropped: the connect that would follow is refused, so an
    // Error::Http here would mean the request went out.
    let embed = llm_wires::build_embed(
        Wire::openai(closed_endpoint().await, "text-embedding-3-small"),
        Some(Secret::from("sk-test-key")),
    )
    .unwrap();

    for (req, want) in [
        (EmbedRequest::default(), "an embed request with no inputs"),
        (EmbedRequest::of(["a", "", "c"]), "input 1 is empty"),
    ] {
        match embed.embed(req).await {
            Err(Error::Invalid { wire, what }) => {
                assert_eq!(wire, "openai");
                assert_eq!(what, want);
            }
            other => panic!("expected Invalid {want}, got {:?}", other.err()),
        }
    }
}

#[tokio::test]
async fn a_refusal_carries_the_providers_words_and_its_request_id() {
    let f = fixture(Reply::Headed(
        401,
        &[("x-request-id", "req_embed_1"), ("retry-after", "2")],
        r#"{"error":{"message":"Incorrect API key provided: sk-***","type":"invalid_request_error"}}"#,
    ))
    .await;
    let embed = llm_wires::build_embed(
        Wire::openai(f.endpoint(), "text-embedding-3-small"),
        Some(Secret::from("sk-wrong")),
    )
    .unwrap();

    match embed.embed(EmbedRequest::one("hello")).await {
        Err(Error::Api {
            status,
            message,
            request_id,
            retry_after,
        }) => {
            assert_eq!(status, 401);
            assert_eq!(message, "Incorrect API key provided: sk-***");
            assert_eq!(request_id.as_deref(), Some("req_embed_1"));
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(2)));
        }
        other => panic!("expected Api, got {:?}", other.err()),
    }
}

#[tokio::test]
async fn a_200_that_is_short_a_vector_is_an_error_and_not_a_hole() {
    // What stores these is fixed-width and pairs them by position: an answer
    // that is nearly right would put one row's vector under another's key for
    // ever.
    let f = fixture(Reply::Json(ONE)).await;
    let embed = llm_wires::build_embed(
        Wire::openai(f.endpoint(), "text-embedding-3-small"),
        Some(Secret::from("sk-test-key")),
    )
    .unwrap();

    match embed.embed(EmbedRequest::of(["one", "two"])).await {
        Err(Error::Decode(why)) => {
            assert_eq!(why, "2 inputs went out and 1 vectors came back");
        }
        other => panic!("expected Decode, got {:?}", other.err()),
    }
}

// -------------------------------------------------------------- the bodies

const ONE: &str = r#"{
  "object": "list",
  "model": "text-embedding-3-small",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}],
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}"#;

const TWO: &str = r#"{
  "object": "list",
  "model": "text-embedding-3-small",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
    {"object": "embedding", "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8]}
  ],
  "usage": {"prompt_tokens": 11, "total_tokens": 11}
}"#;

/// The same three vectors, out of order, each carrying the index it is for.
const SHUFFLED: &str = r#"{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 2, "embedding": [2.0, 2.0]},
    {"object": "embedding", "index": 0, "embedding": [0.0, 0.0]},
    {"object": "embedding", "index": 1, "embedding": [1.0, 1.0]}
  ],
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}"#;
