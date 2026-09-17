//! The OpenAI wire against a real socket.
//!
//! The listener is `llm_wires::testing`, shared with the Anthropic wire's
//! pins and with `domains::llm`'s end-to-end: what these are about is the
//! **bytes**, and every wire answers to the same reader.

use std::collections::BTreeMap;
use std::time::Duration;

use futures_util::StreamExt;
use llm_wires::testing::{Reply, fixture};
use llm_wires::{ChatRequest, Error, Finish, Message, Tool, ToolCall, Wire};
use wire_secret::Secret;

// ----------------------------------------------------------------- the pins

#[tokio::test]
async fn the_request_carries_the_bearer_the_gateway_headers_and_the_body() {
    let f = fixture(Reply::Json(ANSWER)).await;
    let mut headers = BTreeMap::new();
    // The whole of "AWS Bedrock's OpenAI gateway works as pure config".
    headers.insert("OpenAI-Project".to_string(), "default".to_string());
    headers.insert("X-Title".to_string(), "solid".to_string());

    let client = llm_wires::build(
        Wire::OpenAi {
            endpoint: f.endpoint(),
            model: "gpt-4o-mini".into(),
            headers,
        },
        Some(Secret::from("sk-test-key")),
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

    let seen = f.request();
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.path, "/v1/chat/completions");
    assert_eq!(seen.query, "");
    assert_eq!(seen.header("authorization"), Some("Bearer sk-test-key"));
    assert_eq!(seen.header("openai-project"), Some("default"));
    assert_eq!(seen.header("x-title"), Some("solid"));
    assert_eq!(seen.header("content-type"), Some("application/json"));
    assert_eq!(
        seen.json(),
        serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Say hi"},
            ],
            "temperature": 0.2,
            "max_completion_tokens": 64,
        })
    );
}

#[tokio::test]
async fn azure_puts_the_deployment_in_the_path_and_the_key_in_its_own_header() {
    // Go's table, kept: the endpoint's own path is a prefix, not something to
    // replace — a gateway behind `/openai-chat` is a live deployment shape.
    let cases = [
        ("", "gpt-4o", "/openai/deployments/gpt-4o/chat/completions"),
        ("/", "gpt-4o", "/openai/deployments/gpt-4o/chat/completions"),
        (
            "/openai-chat",
            "gpt-4o-mini",
            "/openai-chat/openai/deployments/gpt-4o-mini/chat/completions",
        ),
        (
            "/api/v1/azure",
            "gpt-4o",
            "/api/v1/azure/openai/deployments/gpt-4o/chat/completions",
        ),
    ];

    for (prefix, deployment, want_path) in cases {
        let f = fixture(Reply::Json(ANSWER)).await;
        let client = llm_wires::build(
            Wire::Azure {
                endpoint: f.endpoint_at(prefix),
                deployment: deployment.into(),
                api_version: "2023-07-01-preview".into(),
            },
            Some(Secret::from("azure-key")),
        )
        .unwrap();
        client
            .chat(ChatRequest::ask("", "hello"))
            .await
            .unwrap_or_else(|e| panic!("{prefix}: {e}"));

        let seen = f.request();
        assert_eq!(seen.path, want_path, "prefix {prefix:?}");
        assert_eq!(seen.query, "api-version=2023-07-01-preview");
        // Azure names its own header and takes no bearer.
        assert_eq!(seen.header("api-key"), Some("azure-key"));
        assert_eq!(seen.header("authorization"), None);
        // The deployment is the model on the wire, and it is what Info says.
        assert_eq!(seen.json()["model"], serde_json::json!(deployment));
        assert_eq!(client.info().wire, "azure");
        assert_eq!(client.info().model, deployment);
    }
}

#[tokio::test]
async fn azure_falls_back_to_the_api_version_the_card_did_not_give() {
    let f = fixture(Reply::Json(ANSWER)).await;
    let client = llm_wires::build(
        Wire::Azure {
            endpoint: f.endpoint_at(""),
            deployment: "gpt-4o".into(),
            api_version: String::new(),
        },
        Some(Secret::from("azure-key")),
    )
    .unwrap();
    client.chat(ChatRequest::ask("", "hello")).await.unwrap();
    assert_eq!(
        f.request().query,
        format!("api-version={}", llm_wires::AZURE_DEFAULT_API_VERSION)
    );
}

#[tokio::test]
async fn a_blocking_reply_gives_up_its_tool_calls_and_its_usage() {
    let f = fixture(Reply::Json(
        r#"{
          "id": "chatcmpl-1",
          "choices": [{
            "index": 0,
            "message": {
              "role": "assistant",
              "content": "",
              "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "search_records", "arguments": "{\"query\":\"acme\"}"}},
                {"id": "call_b", "type": "function",
                 "function": {"name": "list_owners", "arguments": "{}"}}
              ]
            },
            "finish_reason": "tool_calls"
          }],
          "usage": {
            "prompt_tokens": 1200, "completion_tokens": 180, "total_tokens": 1380,
            "prompt_tokens_details": {"cached_tokens": 1024},
            "completion_tokens_details": {"reasoning_tokens": 128}
          }
        }"#,
    ))
    .await;

    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o"),
        Some(Secret::from("sk-test")),
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

    assert_eq!(answer.finish, Some(Finish::ToolCalls));
    assert_eq!(
        answer.tool_calls,
        vec![
            ToolCall {
                id: "call_a".into(),
                name: "search_records".into(),
                arguments: r#"{"query":"acme"}"#.into(),
            },
            ToolCall {
                id: "call_b".into(),
                name: "list_owners".into(),
                arguments: "{}".into(),
            },
        ]
    );
    // The same calls on the message, so a caller can push the assistant turn
    // straight back into the next request.
    assert_eq!(answer.message.tool_calls, answer.tool_calls);
    assert_eq!(answer.usage.input, 1200);
    assert_eq!(answer.usage.output, 180);
    assert_eq!(answer.usage.cached, 1024);
    assert_eq!(answer.usage.reasoning, 128);

    // And the forced choice went out with the tools.
    let sent = f.request().json();
    assert_eq!(
        sent["tool_choice"],
        serde_json::json!({"type": "function", "function": {"name": "search_records"}})
    );
}

#[tokio::test]
async fn a_streamed_reply_assembles_its_tool_call_deltas_in_order() {
    // A transcript of the shape the wire actually sends: a role-only opener,
    // text, two parallel tool calls in fragments, the finish, then the usage
    // frame that only arrives because stream_options.include_usage went out.
    const EVENTS: &[&str] = &[
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Looking\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" it up.\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"search_records\",\"arguments\":\"{\\\"query\\\":\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"acme\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"list_owners\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":83,\"completion_tokens\":59,\"prompt_tokens_details\":{\"cached_tokens\":64},\"completion_tokens_details\":{\"reasoning_tokens\":41}}}\n\n",
        "data: [DONE]\n\n",
    ];

    let f = fixture(Reply::Sse(EVENTS)).await;
    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o"),
        Some(Secret::from("sk-test")),
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

    // The text, in order.
    let text: String = chunks.iter().map(|c| c.delta.as_str()).collect();
    assert_eq!(text, "Looking it up.");

    // The fragments, in the order they arrived, each carrying the identity of
    // the call it belongs to.
    let deltas: Vec<(u32, &str, &str)> = chunks
        .iter()
        .filter_map(|c| c.tool_call_delta.as_ref())
        .map(|d| (d.index, d.id.as_str(), d.delta.as_str()))
        .collect();
    assert_eq!(
        deltas,
        vec![
            (0, "call_a", r#"{"query":"#),
            (0, "call_a", r#""acme"}"#),
            (1, "call_b", "{}"),
        ]
    );

    // The finish frame carries the assembled calls.
    let finish = chunks
        .iter()
        .find(|c| c.finish.is_some())
        .expect("a finish frame");
    assert_eq!(finish.finish, Some(Finish::ToolCalls));
    assert_eq!(
        finish.tool_calls,
        vec![
            ToolCall {
                id: "call_a".into(),
                name: "search_records".into(),
                arguments: r#"{"query":"acme"}"#.into(),
            },
            ToolCall {
                id: "call_b".into(),
                name: "list_owners".into(),
                arguments: "{}".into(),
            },
        ]
    );

    // The usage frame arrives AFTER the finish and is not dropped.
    let usage = chunks
        .last()
        .and_then(|c| c.usage)
        .expect("the trailing usage frame");
    assert_eq!((usage.input, usage.output), (83, 59));
    assert_eq!((usage.cached, usage.reasoning), (64, 41));

    // And what was asked for on the way out.
    let sent = f.request().json();
    assert_eq!(sent["stream"], serde_json::json!(true));
    assert_eq!(
        sent["stream_options"],
        serde_json::json!({"include_usage": true})
    );
}

#[tokio::test]
async fn a_streamed_reply_survives_a_provider_that_does_not_say_done() {
    // Some gateways just close the body. The last frame must still arrive.
    const EVENTS: &[&str] = &[
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n",
    ];
    let f = fixture(Reply::Sse(EVENTS)).await;
    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o"),
        Some(Secret::from("sk-test")),
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
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].delta, "done");
    assert_eq!(chunks[1].finish, Some(Finish::Stop));
}

#[tokio::test]
async fn a_frame_carrying_a_last_fragment_and_a_finish_reason_keeps_both() {
    // The shape an OpenAI-compatible server sends and native OpenAI does not:
    // the closing argument fragment and `finish_reason` in ONE frame. The
    // assembled call must be complete JSON, not truncated at the fragment
    // before it.
    const EVENTS: &[&str] = &[
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"search_records\",\"arguments\":\"{\\\"query\\\":\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" ok.\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"acme\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
    ];

    let f = fixture(Reply::Sse(EVENTS)).await;
    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "llama3.2:3b"),
        Some(Secret::from("ollama")),
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

    let text: String = chunks.iter().map(|c| c.delta.as_str()).collect();
    assert_eq!(text, " ok.", "the content on the finish frame is not lost");

    let finish = chunks.last().expect("a finish chunk");
    assert_eq!(finish.finish, Some(Finish::ToolCalls));
    assert_eq!(
        finish.tool_calls,
        vec![ToolCall {
            id: "call_a".into(),
            name: "search_records".into(),
            arguments: r#"{"query":"acme"}"#.into(),
        }],
        "complete, because the fragment in the finish frame ran before the finish"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&finish.tool_calls[0].arguments).unwrap(),
        serde_json::json!({"query": "acme"}),
        "and therefore parseable, which is the point"
    );
    assert_eq!(finish.usage.unwrap().input, 12);
}

#[tokio::test]
async fn a_200_that_is_not_an_event_stream_is_an_error_and_not_an_empty_stream() {
    // A gateway that quietly ignored `"stream": true` and answered with an
    // ordinary completion body. Framing it produces no chunks at all, so
    // without the guard the caller sees a turn that succeeded and said
    // nothing — the outcome the mid-stream `error` arm exists to prevent,
    // arriving through the door that arm does not watch.
    let f = fixture(Reply::Json(ANSWER)).await;
    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o"),
        Some(Secret::from("sk-test")),
    )
    .unwrap();
    match client.chat_stream(ChatRequest::ask("", "hi")).await {
        Err(Error::Decode(why)) => {
            assert!(why.contains("not an event stream"), "{why}");
            assert!(why.contains("application/json"), "{why}");
            // The body is in there, so an operator can see what came back.
            assert!(why.contains("chatcmpl-1"), "{why}");
        }
        Ok(_) => panic!("a completion body must not come back as an empty stream"),
        Err(other) => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_transport_error_does_not_carry_the_url_it_failed_on() {
    // reqwest's own Display appends " for url (…)", and an endpoint path can
    // hold a deployment name, a tenant, or a gateway route nobody meant to
    // put in a log line. `Info::endpoint` is where a caller gets the URL.
    let client = llm_wires::build(
        Wire::Azure {
            // Port 1 is reserved and nothing listens on it.
            endpoint: "http://127.0.0.1:1/tenant-42".into(),
            deployment: "finance-gpt".into(),
            api_version: String::new(),
        },
        Some(Secret::from("azure-key")),
    )
    .unwrap();

    let err = client
        .chat(ChatRequest::ask("", "hi"))
        .await
        .expect_err("nothing listens on port 1");
    let shown = err.to_string();
    assert!(matches!(err, Error::Http(_)), "{err:?}");
    assert!(!shown.contains("finance-gpt"), "{shown}");
    assert!(!shown.contains("tenant-42"), "{shown}");
    assert!(!shown.contains("127.0.0.1"), "{shown}");
    // The endpoint the caller configured is still theirs to report.
    assert_eq!(client.info().endpoint, "http://127.0.0.1:1/tenant-42");
}

#[tokio::test]
async fn a_transport_error_carries_the_operating_systems_own_words() {
    // The other half of the pin above, and the reason `Error::Http` is
    // `transparent`: reqwest's own `Display` here is `error sending request`
    // and nothing more — *connection refused* is in the cause it holds. What
    // an operator reads is `anyhow`'s `{:#}` (`solid/src/main.rs`), so that
    // is what this renders, rather than a formatter nobody uses.
    let client = llm_wires::build(
        Wire::openai(llm_wires::testing::closed_endpoint().await, "gpt-4o"),
        Some(Secret::from("sk-test")),
    )
    .unwrap();

    let err = client
        .chat(ChatRequest::ask("", "hi"))
        .await
        .expect_err("nothing is listening");

    // The chain is there at all — this is the regression the attribute
    // guards, and it fails loudly rather than by an OS string that differs
    // between platforms.
    let mut cause: &dyn std::error::Error = &err;
    let mut last = cause.to_string();
    let mut depth = 0;
    while let Some(next) = cause.source() {
        cause = next;
        last = cause.to_string();
        depth += 1;
    }
    assert!(depth >= 2, "the cause chain stops at {err:?}");

    // And the whole of it reaches the line the binary prints.
    let top = err.to_string();
    let shown = format!("{:#}", anyhow::Error::from(err));
    assert!(
        shown.contains(&last),
        "the last cause {last:?} is not in {shown:?}"
    );
    // Once each. `transparent` and not `#[source]` under a `{0}` message is
    // what keeps this true: those two together interpolate the very error
    // the chain starts at, and an operator reads reqwest's line twice.
    assert_eq!(shown.matches(&top).count(), 1, "{top:?} twice in {shown:?}");
    // Still no URL: the causes carry the operating system's message, not the
    // request.
    assert!(!shown.contains("127.0.0.1"), "{shown}");
}

#[tokio::test]
async fn dropping_the_stream_ends_the_body() {
    // The whole reason there is no `close()`: a caller that walks away from a
    // turn — a Stop button, a cancelled run — releases the connection by
    // dropping the stream, and the provider stops being charged for it.
    const EVENT: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"...\"},\"finish_reason\":null}]}\n\n";

    let f = fixture(Reply::SseForever(EVENT)).await;
    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o"),
        Some(Secret::from("sk-test")),
    )
    .unwrap();

    let mut stream = client
        .chat_stream(ChatRequest::ask("", "talk"))
        .await
        .unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap().delta, "...");
    drop(stream);

    tokio::time::timeout(Duration::from_secs(10), f.gone.notified())
        .await
        .expect("the listener must see the connection go when the stream is dropped");
}

#[tokio::test]
async fn a_4xx_carries_the_apis_message_and_not_the_request() {
    let f = fixture(Reply::Status(
        401,
        r#"{"error":{"message":"Incorrect API key provided: sk-li***key.","type":"invalid_request_error","code":"invalid_api_key"}}"#,
    ))
    .await;
    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o"),
        Some(Secret::from("sk-live-do-not-log-me")),
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
        Error::Api {
            status, message, ..
        } => {
            assert_eq!(*status, 401);
            assert_eq!(message, "Incorrect API key provided: sk-li***key.");
        }
        other => panic!("{other:?}"),
    }
    // Neither the prompt nor the key is anywhere in what a pane would show.
    let shown = err.to_string();
    assert!(!shown.contains("nobody else should read"), "{shown}");
    assert!(!shown.contains("sk-live-do-not-log-me"), "{shown}");
}

#[tokio::test]
async fn a_gateways_html_error_page_still_reaches_the_operator() {
    let f = fixture(Reply::Status(502, "<html>502 Bad Gateway</html>")).await;
    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o"),
        Some(Secret::from("sk-test")),
    )
    .unwrap();
    match client.chat(ChatRequest::ask("", "hi")).await {
        Err(Error::Api {
            status, message, ..
        }) => {
            assert_eq!(status, 502);
            assert_eq!(message, "<html>502 Bad Gateway</html>");
        }
        other => panic!("{:?}", other.err()),
    }
}

#[tokio::test]
async fn a_stream_that_opens_with_a_4xx_never_becomes_a_stream() {
    let f = fixture(Reply::Status(
        429,
        r#"{"error":{"message":"Rate limit reached for gpt-4o."}}"#,
    ))
    .await;
    let client = llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o"),
        Some(Secret::from("sk-test")),
    )
    .unwrap();
    match client.chat_stream(ChatRequest::ask("", "hi")).await {
        Err(Error::Api {
            status, message, ..
        }) => {
            assert_eq!(status, 429);
            assert_eq!(message, "Rate limit reached for gpt-4o.");
        }
        Ok(_) => panic!("a 429 must not come back as an empty stream"),
        Err(other) => panic!("{other:?}"),
    }
}

const ANSWER: &str = r#"{
  "id": "chatcmpl-1",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}
}"#;
