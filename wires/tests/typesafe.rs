//! The TypeSafe System One wire against a real socket.
//!
//! The same listener the chat wires' pins use (`llm_wires::testing`), for
//! the same reason: what these are about is the **bytes** — which path,
//! which headers, which JSON — and a fake client can only ever agree with the
//! code that built it. The request and response bodies are the ones the
//! published SDK (`@typesafe-ai/sdk` 0.6.0) declares.

use std::collections::BTreeMap;
use std::time::Duration;

use llm_wires::testing::{Reply, closed_endpoint, fixture};
use llm_wires::{Answer, Error, Judgement, Question, Usage, Wire};
use serde_json::{Value, json};
use wire_secret::Secret;

fn asked() -> Judgement {
    Judgement::of("Help! My payouts have been failing for 3 days.")
        .ask("is_urgent", Question::noul("Does this convey urgency?"))
        .ask(
            "department",
            Question::choice(
                "Which team should handle this?",
                [
                    ("billing", "Payments, invoicing, refunds"),
                    ("technical", "Bugs, outages, integrations"),
                    ("sales", "Pricing, upgrades, new accounts"),
                ],
            ),
        )
        .ask(
            "frustration",
            Question::score(
                "How frustrated is the customer?",
                ["Calm", "Frustrated", "Very angry"],
            ),
        )
}

#[tokio::test]
async fn the_request_carries_the_bearer_the_sdk_headers_and_the_gateway_headers() {
    let f = fixture(Reply::Json(ANSWERS)).await;
    let mut headers = BTreeMap::new();
    headers.insert("X-Tenant".to_string(), "acme".to_string());

    let judge = llm_wires::build_judge(
        Wire::TypeSafe {
            endpoint: f.root(),
            model: "jev-latest".into(),
            headers,
        },
        Some(Secret::from("ts-test-key")),
    )
    .unwrap();

    let verdict = judge.judge(asked()).await.unwrap();
    assert_eq!(verdict.model, "jev-latest");
    assert_eq!(
        verdict.usage,
        Usage {
            input: 312,
            output: 48,
            ..Usage::default()
        }
    );
    assert_eq!(verdict.answers["is_urgent"], Answer::Noul { yes: 0.92 });
    assert_eq!(
        verdict.answers["department"],
        Answer::Choice {
            choice: "technical".into(),
            probabilities: [("billing", 0.08), ("technical", 0.85), ("sales", 0.07)]
                .into_iter()
                .map(|(k, p)| (k.to_string(), p))
                .collect(),
            confidence: 0.82,
        }
    );
    assert_eq!(
        verdict.answers["frustration"],
        Answer::Score {
            score: 1.6,
            probabilities: vec![0.05, 0.3, 0.65],
            confidence: 0.78,
        }
    );

    let seen = f.request();
    assert_eq!(seen.method, "POST");
    // The `/v1` is the wire's, not the operator's: the endpoint on the card
    // is the host.
    assert_eq!(seen.path, "/v1/systemone");
    assert_eq!(seen.query, "");
    assert_eq!(seen.header("authorization"), Some("Bearer ts-test-key"));
    assert_eq!(seen.header("accept"), Some("application/json"));
    assert_eq!(seen.header("content-type"), Some("application/json"));
    // What their SDK sends about itself, spelled for this crate.
    let ident = format!("llm-wires/{}", env!("CARGO_PKG_VERSION"));
    assert_eq!(seen.header("user-agent"), Some(ident.as_str()));
    assert_eq!(seen.header("x-typesafe-sdk"), Some(ident.as_str()));
    assert_eq!(seen.header("x-typesafe-runtime"), Some("rust"));
    assert_eq!(seen.header("x-tenant"), Some("acme"));
    assert_eq!(
        seen.json(),
        json!({
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments, invoicing, refunds",
                        "technical": "Bugs, outages, integrations",
                        "sales": "Pricing, upgrades, new accounts"
                    }
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]
                }
            }
        })
    );
    // And the same request is the same bytes twice: the maps are sorted.
    assert!(
        seen.body.find("\"department\"").unwrap() < seen.body.find("\"frustration\"").unwrap()
            && seen.body.find("\"frustration\"").unwrap()
                < seen.body.find("\"is_urgent\"").unwrap(),
        "{}",
        seen.body
    );
}

#[tokio::test]
async fn a_gateway_prefix_is_kept_and_a_card_can_replace_the_user_agent() {
    // The operator's headers go on first, so `User-Agent` is one they can
    // set. The key is not: it goes on last and unconditionally.
    let f = fixture(Reply::Json(ANSWERS)).await;
    let mut headers = BTreeMap::new();
    headers.insert("User-Agent".to_string(), "acme-router/1".to_string());
    headers.insert(
        "Authorization".to_string(),
        "Bearer not-the-key".to_string(),
    );

    let judge = llm_wires::build_judge(
        Wire::TypeSafe {
            endpoint: f.endpoint_at("/judge/"),
            model: "jev-latest".into(),
            headers,
        },
        Some(Secret::from("ts-real")),
    )
    .unwrap();
    judge.judge(asked()).await.unwrap();

    let seen = f.request();
    assert_eq!(seen.path, "/judge/v1/systemone");
    assert_eq!(seen.header("user-agent"), Some("acme-router/1"));
    assert_eq!(seen.header("authorization"), Some("Bearer ts-real"));
}

#[tokio::test]
async fn structured_state_and_instructions_go_out_as_the_json_they_are() {
    // The wire takes text, an object, an array or null at every leaf, and
    // the Rust types are `Value` there so a solid step can hand a row over
    // without flattening it to a string first.
    let f = fixture(Reply::Json(
        r#"{"model":"jev-latest","answers":{"refund":{"type":"noul","noul":0.4}},"usage":{"input_tokens":50,"output_tokens":8}}"#,
    ))
    .await;
    let judge = llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-latest"),
        Some(Secret::from("ts-test")),
    )
    .unwrap();

    let state =
        json!({"ticket": 4471, "messages": [{"from": "customer", "text": "charged twice"}]});
    let verdict = judge
        .judge(Judgement::of(state.clone()).ask(
            "refund",
            Question::noul_described(
                json!({"question": "Is a refund being asked for?", "scope": "the last message"}),
                "The customer wants money back",
                Value::Null,
            ),
        ))
        .await
        .unwrap();
    assert_eq!(verdict.answers["refund"], Answer::Noul { yes: 0.4 });

    let body = f.request().json();
    assert_eq!(body["state"], state);
    assert_eq!(
        body["questions"]["refund"],
        json!({
            "type": "noul",
            "instructions": {"question": "Is a refund being asked for?", "scope": "the last message"},
            "criteria": {"true": "The customer wants money back"}
        })
    );
}

#[tokio::test]
async fn a_4xx_carries_the_apis_detail_its_request_id_and_never_the_state() {
    let f = fixture(Reply::Headed(
        401,
        &[("x-typesafe-request-id", "req_01HZX")],
        r#"{"detail":"Invalid API key"}"#,
    ))
    .await;
    let judge = llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-latest"),
        Some(Secret::from("ts-live-do-not-log-me")),
    )
    .unwrap();

    let err = judge
        .judge(
            Judgement::of("a record nobody else should read")
                .ask("q", Question::noul("is it private?")),
        )
        .await
        .expect_err("a 401 is an error");

    match &err {
        Error::Api {
            status,
            message,
            request_id,
            retry_after,
        } => {
            assert_eq!(*status, 401);
            assert_eq!(message, "Invalid API key");
            assert_eq!(request_id.as_deref(), Some("req_01HZX"));
            assert_eq!(*retry_after, None);
        }
        other => panic!("{other:?}"),
    }
    let shown = err.to_string();
    assert_eq!(
        shown,
        "the provider answered 401: Invalid API key (request req_01HZX)"
    );
    assert!(!shown.contains("nobody else should read"), "{shown}");
    assert!(!shown.contains("do-not-log-me"), "{shown}");
}

#[tokio::test]
async fn a_422_reads_like_their_sdks_message_and_a_429_carries_its_retry_hint() {
    let f = fixture(Reply::Status(
        422,
        r#"{"detail":[{"loc":["body","questions","tone","criteria"],"msg":"ensure this value has at least 2 items","type":"value_error.list.min_items"}]}"#,
    ))
    .await;
    let judge = llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-latest"),
        Some(Secret::from("ts-test")),
    )
    .unwrap();
    // The wire refuses a one-level score itself, so send two and let the
    // fixture play a server that found something else to dislike.
    let two = || Judgement::of("x").ask("tone", Question::score("?", ["a", "b"]));
    match judge.judge(two()).await {
        Err(Error::Api {
            status, message, ..
        }) => {
            assert_eq!(status, 422);
            assert_eq!(
                message,
                "questions.tone.criteria: ensure this value has at least 2 items"
            );
        }
        other => panic!("{other:?}"),
    }

    let f = fixture(Reply::Headed(
        429,
        &[("retry-after-ms", "1500"), ("retry-after", "2")],
        r#"{"detail":{"message":"Rate limit exceeded"}}"#,
    ))
    .await;
    let judge = llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-latest"),
        Some(Secret::from("ts-test")),
    )
    .unwrap();
    match judge.judge(two()).await {
        Err(Error::Api {
            status,
            message,
            retry_after,
            ..
        }) => {
            assert_eq!(status, 429);
            assert_eq!(message, "Rate limit exceeded");
            // Surfaced, not acted on: there was exactly one request.
            assert_eq!(retry_after, Some(Duration::from_millis(1500)));
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_200_that_skips_a_question_is_an_error_and_not_a_verdict_with_a_hole() {
    // A solid step routes on `answers["is_urgent"]`; a missing key must not
    // read as a no.
    let f = fixture(Reply::Json(
        r#"{"model":"jev-latest","answers":{"is_urgent":{"type":"noul","noul":0.9}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
    ))
    .await;
    let judge = llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-latest"),
        Some(Secret::from("ts-test")),
    )
    .unwrap();
    match judge.judge(asked()).await {
        Err(Error::Decode(why)) => assert!(why.contains("no answer for question"), "{why}"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_request_the_server_would_refuse_never_reaches_it() {
    // Nothing is listening, so a request that got out would be a connect
    // error, not an Invalid.
    let judge = llm_wires::build_judge(
        Wire::typesafe(closed_endpoint().await, "jev-latest"),
        Some(Secret::from("ts-test")),
    )
    .unwrap();
    match judge.judge(Judgement::of("x")).await {
        Err(Error::Invalid { wire, what }) => {
            assert_eq!(wire, "typesafe");
            assert!(what.contains("at least one question"), "{what}");
        }
        other => panic!("{other:?}"),
    }
    match judge
        .judge(Judgement::of("x").ask("tone", Question::score("?", ["only"])))
        .await
    {
        Err(Error::Invalid { what, .. }) => assert!(what.contains("at least two levels"), "{what}"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_closed_port_is_a_transport_error_with_the_operating_systems_words() {
    let judge = llm_wires::build_judge(
        Wire::typesafe(closed_endpoint().await, "jev-latest"),
        Some(Secret::from("ts-test")),
    )
    .unwrap();
    let err = judge
        .judge(Judgement::of("x").ask("q", Question::noul("?")))
        .await
        .expect_err("nothing is listening");
    assert!(matches!(err, Error::Http(_)), "{err:?}");
    let chain = format!("{:#}", anyhow::Error::from(err));
    assert!(chain.contains("refused"), "{chain}");
}

/// The docs' example response for [`asked`].
const ANSWERS: &str = r#"{
  "model": "jev-latest",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.92},
    "department": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
      "confidence": 0.82
    },
    "frustration": {
      "type": "score",
      "score": 1.6,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
      "confidence": 0.78
    }
  },
  "usage": {"input_tokens": 312, "output_tokens": 48}
}"#;
