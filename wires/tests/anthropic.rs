//! The Anthropic messages wire against a real socket.
//!
//! The same listener the OpenAI pins use (`llm_wires::testing`), for the same
//! reason: what these are about is the **bytes** — which path, which headers,
//! which JSON — and a fake client can only ever agree with the code that built
//! it. The SSE transcripts are written by hand from the event sequence the
//! wire documents; there is no recorded one to port.

use std::collections::BTreeMap;

use futures_util::StreamExt;
use llm_wires::testing::{Reply, fixture};
use llm_wires::{ChatRequest, Error, Finish, Message, Tool, ToolCall, Wire};
use wire_secret::Secret;

#[tokio::test]
async fn the_request_carries_the_api_key_the_version_and_the_gateway_headers() {
    let f = fixture(Reply::Json(ANSWER)).await;
    let mut headers = BTreeMap::new();
    // What a self-hosted proxy or LiteLLM in front of Claude needs, and the
    // whole of "a new gateway is pure config".
    headers.insert("X-Tenant".to_string(), "acme".to_string());

    let client = llm_wires::build(
        Wire::Anthropic {
            endpoint: f.root(),
            model: "claude-sonnet-4".into(),
            headers,
        },
        Some(Secret::from("sk-ant-test-key")),
    )
    .unwrap();

    let answer = client
        .chat(ChatRequest {
            messages: vec![Message::user("Say hi")],
            temperature: Some(0.2),
            max_tokens: Some(64),
            system: "You are terse.".into(),
            ..ChatRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(answer.message.content, "hi");
    assert_eq!(answer.finish, Some(Finish::Stop));

    let seen = f.request();
    assert_eq!(seen.method, "POST");
    // The `/v1` is the wire's, not the operator's: the endpoint on the card is
    // the host.
    assert_eq!(seen.path, "/v1/messages");
    assert_eq!(seen.query, "");
    // Not a bearer, and not `Authorization` at all.
    assert_eq!(seen.header("x-api-key"), Some("sk-ant-test-key"));
    assert_eq!(seen.header("authorization"), None);
    // Required by this wire: without it every request is a 400.
    assert_eq!(seen.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(seen.header("x-tenant"), Some("acme"));
    assert_eq!(
        seen.json(),
        serde_json::json!({
            "model": "claude-sonnet-4",
            "max_tokens": 64,
            // Its own field, not the first message.
            "system": [{"type": "text", "text": "You are terse."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Say hi"}]},
            ],
            "temperature": 0.2,
        })
    );
}

#[tokio::test]
async fn a_gateway_pinned_to_another_version_says_so_on_the_card() {
    // The operator's headers go on first, so `anthropic-version` is one they
    // can set. The key is not: it goes on last and unconditionally, because a
    // header that could displace the credential is a 401 nobody can explain.
    let f = fixture(Reply::Json(ANSWER)).await;
    let mut headers = BTreeMap::new();
    headers.insert("anthropic-version".to_string(), "2024-10-22".to_string());
    headers.insert("x-api-key".to_string(), "not-the-key".to_string());

    let client = llm_wires::build(
        Wire::Anthropic {
            endpoint: f.root(),
            model: "claude-sonnet-4".into(),
            headers,
        },
        Some(Secret::from("sk-ant-real")),
    )
    .unwrap();
    client.chat(ChatRequest::ask("", "hi")).await.unwrap();

    let seen = f.request();
    assert_eq!(seen.header("anthropic-version"), Some("2024-10-22"));
    assert_eq!(seen.header("x-api-key"), Some("sk-ant-real"));
}

#[tokio::test]
async fn a_request_that_leaves_max_tokens_out_still_gets_a_turn() {
    // This wire requires it and has no "as many as it takes", so a caller that
    // did not think about it gets the default rather than a 400.
    let f = fixture(Reply::Json(ANSWER)).await;
    let client = llm_wires::build(
        Wire::anthropic(f.root(), "claude-sonnet-4"),
        Some(Secret::from("sk-ant-test")),
    )
    .unwrap();
    client.chat(ChatRequest::ask("", "hi")).await.unwrap();
    assert_eq!(f.request().json()["max_tokens"], serde_json::json!(16384));
}

#[tokio::test]
async fn the_cache_breakpoint_goes_out_only_with_the_flag_and_only_on_the_prefix() {
    // The one wire the flag changes a byte on. It caches
    // tools → system → messages, so one breakpoint on the last system block
    // covers the tools and the prompt; the last tool is marked too, so a
    // request with tools and no prompt still caches its definitions.
    let tools = || {
        vec![
            Tool {
                name: "search".into(),
                description: "search".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
            Tool {
                name: "now".into(),
                description: "the time".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ]
    };

    for cache in [false, true] {
        let f = fixture(Reply::Json(ANSWER)).await;
        let client = llm_wires::build(
            Wire::anthropic(f.root(), "claude-sonnet-4"),
            Some(Secret::from("sk-ant-test")),
        )
        .unwrap();
        client
            .chat(ChatRequest {
                messages: vec![Message::user("who owns acme?")],
                tools: tools(),
                system: "You are terse.".into(),
                cache_stable_prefix: cache,
                ..ChatRequest::default()
            })
            .await
            .unwrap();

        let sent = f.request().json();
        if !cache {
            // Byte-identical to a request that never heard of the flag.
            assert!(!sent.to_string().contains("cache_control"), "{sent}");
            continue;
        }
        let breakpoint = serde_json::json!({"type": "ephemeral"});
        assert_eq!(sent["system"][0]["cache_control"], breakpoint);
        assert!(sent["tools"][0].get("cache_control").is_none(), "{sent}");
        assert_eq!(sent["tools"][1]["cache_control"], breakpoint);
        assert_eq!(
            sent.to_string().matches("cache_control").count(),
            2,
            "the per-turn messages stay uncached: {sent}"
        );
    }
}

#[tokio::test]
async fn a_blocking_reply_gives_up_its_tool_calls_and_both_cache_counters() {
    let f = fixture(Reply::Json(
        r#"{
          "id": "msg_1", "type": "message", "role": "assistant",
          "model": "claude-sonnet-4",
          "content": [
            {"type": "text", "text": "Looking it up."},
            {"type": "tool_use", "id": "toolu_a", "name": "search_records",
             "input": {"query": "acme"}},
            {"type": "tool_use", "id": "toolu_b", "name": "list_owners", "input": {}}
          ],
          "stop_reason": "tool_use", "stop_sequence": null,
          "usage": {"input_tokens": 1200, "output_tokens": 180,
                    "cache_read_input_tokens": 1024,
                    "cache_creation_input_tokens": 512}
        }"#,
    ))
    .await;

    let client = llm_wires::build(
        Wire::anthropic(f.root(), "claude-sonnet-4"),
        Some(Secret::from("sk-ant-test")),
    )
    .unwrap();
    let answer = client
        .chat(ChatRequest {
            messages: vec![Message::user("who owns acme?")],
            tools: vec![Tool {
                name: "search_records".into(),
                description: "search".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            tool_choice: Some("search_records".into()),
            ..ChatRequest::default()
        })
        .await
        .unwrap();

    assert_eq!(answer.message.content, "Looking it up.");
    assert_eq!(answer.finish, Some(Finish::ToolCalls));
    assert_eq!(
        answer.tool_calls,
        vec![
            ToolCall {
                id: "toolu_a".into(),
                name: "search_records".into(),
                arguments: r#"{"query":"acme"}"#.into(),
            },
            ToolCall {
                id: "toolu_b".into(),
                name: "list_owners".into(),
                arguments: "{}".into(),
            },
        ]
    );
    assert_eq!(answer.message.tool_calls, answer.tool_calls);
    // `input` EXCLUDES the cached tokens on this wire, where OpenAI's includes
    // them, and this is the wire that bills a cache WRITE.
    assert_eq!(answer.usage.input, 1200);
    assert_eq!(answer.usage.output, 180);
    assert_eq!(answer.usage.cached, 1024);
    assert_eq!(answer.usage.cache_creation, 512);
    assert_eq!(answer.usage.reasoning, 0);

    // And the forced tool went out named, in this wire's shape.
    assert_eq!(
        f.request().json()["tool_choice"],
        serde_json::json!({"type": "tool", "name": "search_records"})
    );
}

#[tokio::test]
async fn a_streamed_reply_assembles_its_tool_call_and_bills_the_whole_turn() {
    // The event sequence the wire sends, `event:` lines and all — they are
    // ignored, because the payload names itself.
    const EVENTS: &[&str] = &[
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":83,\"output_tokens\":0,\"cache_read_input_tokens\":64,\"cache_creation_input_tokens\":19}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Looking\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" it up.\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_a\",\"name\":\"search_records\",\"input\":{}}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\":\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"acme\\\"}\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":59}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ];

    let f = fixture(Reply::Sse(EVENTS)).await;
    let client = llm_wires::build(
        Wire::anthropic(f.root(), "claude-sonnet-4"),
        Some(Secret::from("sk-ant-test")),
    )
    .unwrap();

    let mut stream = client
        .chat_stream(ChatRequest::ask("", "who owns acme?"))
        .await
        .unwrap();
    let mut chunks = Vec::new();
    while let Some(c) = stream.next().await {
        chunks.push(c.unwrap());
    }

    // The prompt's price is known from the first event, before a token of
    // output exists.
    let first = chunks[0]
        .usage
        .expect("message_start carries the input side");
    assert_eq!((first.input, first.output), (83, 0));
    assert_eq!((first.cached, first.cache_creation), (64, 19));

    let text: String = chunks.iter().map(|c| c.delta.as_str()).collect();
    assert_eq!(text, "Looking it up.");

    // The fragments, in arrival order, each carrying the identity of the call
    // it belongs to.
    let deltas: Vec<(u32, &str, &str)> = chunks
        .iter()
        .filter_map(|c| c.tool_call_delta.as_ref())
        .map(|d| (d.index, d.id.as_str(), d.delta.as_str()))
        .collect();
    assert_eq!(
        deltas,
        vec![
            // 0, though it is the second content block: the index names which
            // of the parallel calls, as it does on the other wire.
            (0, "toolu_a", r#"{"query":"#),
            (0, "toolu_a", r#""acme"}"#),
        ]
    );

    // The finish carries the assembled call — as it does on the other wire,
    // so a caller that runs tools does not have to know which answered it.
    let last = chunks.last().expect("a finish chunk");
    assert_eq!(last.finish, Some(Finish::ToolCalls));
    assert_eq!(
        last.tool_calls,
        vec![ToolCall {
            id: "toolu_a".into(),
            name: "search_records".into(),
            arguments: r#"{"query":"acme"}"#.into(),
        }]
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&last.tool_calls[0].arguments).unwrap(),
        serde_json::json!({"query": "acme"}),
        "complete JSON, which is the point"
    );
    // And the whole bill: the input side came from `message_start` and the
    // output side from `message_delta`, and neither event had both.
    let usage = last.usage.expect("the finish carries the whole bill");
    assert_eq!((usage.input, usage.output), (83, 59));
    assert_eq!((usage.cached, usage.cache_creation), (64, 19));

    // And `stream` went out on the request that asked for all this.
    assert_eq!(f.request().json()["stream"], serde_json::json!(true));
}

#[tokio::test]
async fn a_streamed_reply_survives_a_provider_that_just_closes_the_body() {
    // No `message_stop`, no blank line after the last event: the turn still
    // ends with everything it produced.
    const EVENTS: &[&str] = &[
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":9}}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n",
    ];
    let f = fixture(Reply::Sse(EVENTS)).await;
    let client = llm_wires::build(
        Wire::anthropic(f.root(), "claude-sonnet-4"),
        Some(Secret::from("sk-ant-test")),
    )
    .unwrap();
    let mut stream = client
        .chat_stream(ChatRequest::ask("", "hi"))
        .await
        .unwrap();
    let mut chunks = Vec::new();
    while let Some(c) = stream.next().await {
        chunks.push(c.unwrap());
    }
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[1].delta, "done");
    assert_eq!(chunks[2].finish, Some(Finish::Stop));
    assert_eq!(chunks[2].usage.unwrap().output, 2);
}

#[tokio::test]
async fn a_stream_that_fails_mid_turn_says_so_instead_of_ending_clean() {
    // This wire reports an overload as an `error` event on a 200 body. Without
    // the arm the turn would end in a clean EOF with no answer and nothing to
    // explain it.
    const EVENTS: &[&str] = &[
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":9}}}\n\n",
        "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
    ];
    let f = fixture(Reply::Sse(EVENTS)).await;
    let client = llm_wires::build(
        Wire::anthropic(f.root(), "claude-sonnet-4"),
        Some(Secret::from("sk-ant-test")),
    )
    .unwrap();
    let mut stream = client
        .chat_stream(ChatRequest::ask("", "hi"))
        .await
        .unwrap();

    let mut last = None;
    while let Some(c) = stream.next().await {
        last = Some(c);
    }
    match last.expect("the stream ends with the failure") {
        // Not an `Api { status: 200 }`: there was no status.
        Err(Error::Stream { message }) => assert_eq!(message, "Overloaded"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_4xx_carries_the_apis_message_and_not_the_request() {
    let f = fixture(Reply::Status(
        401,
        r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
    ))
    .await;
    let client = llm_wires::build(
        Wire::anthropic(f.root(), "claude-sonnet-4"),
        Some(Secret::from("sk-ant-live-do-not-log-me")),
    )
    .unwrap();

    let err = client
        .chat(ChatRequest::ask(
            "You are terse.",
            "a question nobody else should read",
        ))
        .await
        .expect_err("a 401 is an error");

    match &err {
        Error::Api { status, message } => {
            assert_eq!(*status, 401);
            assert_eq!(message, "invalid x-api-key");
        }
        other => panic!("{other:?}"),
    }
    // Neither the prompt nor the key is anywhere in what a pane would show.
    let shown = err.to_string();
    assert!(!shown.contains("nobody else should read"), "{shown}");
    assert!(!shown.contains("sk-ant-live-do-not-log-me"), "{shown}");
}

#[tokio::test]
async fn a_200_that_is_not_an_event_stream_is_an_error_and_not_an_empty_stream() {
    // A gateway that quietly ignored `"stream": true`. Framing this produces
    // no chunks at all, so without the guard the caller sees a turn that
    // succeeded and said nothing.
    let f = fixture(Reply::Json(ANSWER)).await;
    let client = llm_wires::build(
        Wire::anthropic(f.root(), "claude-sonnet-4"),
        Some(Secret::from("sk-ant-test")),
    )
    .unwrap();
    match client.chat_stream(ChatRequest::ask("", "hi")).await {
        Err(Error::Decode(why)) => {
            assert!(why.contains("not an event stream"), "{why}");
            assert!(why.contains("msg_1"), "{why}");
        }
        Ok(_) => panic!("a messages body must not come back as an empty stream"),
        Err(other) => panic!("{other:?}"),
    }
}

const ANSWER: &str = r#"{
  "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4",
  "content": [{"type": "text", "text": "hi"}],
  "stop_reason": "end_turn", "stop_sequence": null,
  "usage": {"input_tokens": 9, "output_tokens": 1}
}"#;
