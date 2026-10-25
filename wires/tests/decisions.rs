//! OpenAI's Decisions API, the OpenAI wire's `Judge`, against a real socket.
//!
//! Every reply below is a live one, recorded 2026-10-25 from
//! `https://api.openai.com/v1/decisions` with `gpt-6-luna` (the request
//! ids and the key are not kept): the three kinds in one call, a refusal, a
//! 400 for a one-level score and a 401. What these pin is the bytes, as the
//! TypeSafe pins do: which path, which body, and what a reply decodes to.

use std::collections::BTreeMap;
use std::time::Duration;

use llm_wires::testing::{Reply, fixture};
use llm_wires::{Answer, Error, Image, Judgement, Question, Usage, Wire};
use serde_json::json;
use wire_secret::Secret;

/// The live reply to [`asked`], 2026-10-25.
const ANSWERS: &str = r#"{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"q2","probability":1.0},{"type":"choice","name":"q0","choice":"engineering","probabilities":[{"value":"billing","probability":0.0},{"value":"engineering","probability":1.0},{"value":"sales","probability":0.0}],"confidence":1.0},{"type":"score","name":"q1","score":1.97,"probabilities":[{"value":0,"label":"0","probability":0.0},{"value":1,"label":"1","probability":0.03},{"value":2,"label":"2","probability":0.97}],"confidence":0.96}],"usage":{"input_tokens":429,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":429}}"#;

/// The questions [`ANSWERS`] answers. Sorted by the caller's names, so
/// `lane` goes out as `q0`, `severity` as `q1`, `urgent` as `q2`.
fn asked() -> Judgement {
    Judgement::of(
        "Our CSV export stopped working after a field was renamed. Please fix it today, the \
         board meets at 3pm.",
    )
    .ask(
        "urgent",
        Question::noul("Is the writer asking for something to be done urgently?"),
    )
    .ask(
        "lane",
        Question::choice(
            "Which team should take this?",
            [
                ("billing", "Payments, invoices, refunds."),
                ("engineering", "Bugs and broken features."),
                ("sales", "Pricing and new contracts."),
            ],
        ),
    )
    .ask(
        "severity",
        Question::score(
            "How severe is the problem?",
            ["Cosmetic.", "Annoying, has a workaround.", "Blocks work."],
        ),
    )
}

fn judge_at(endpoint: String) -> Box<dyn llm_wires::Judge> {
    llm_wires::build_judge(
        Wire::openai(endpoint, "gpt-6-luna"),
        Some(Secret::from("sk-test-key")),
    )
    .unwrap()
}

#[tokio::test]
async fn the_three_kinds_go_out_as_predicate_choice_and_score_and_keep_their_distributions() {
    let f = fixture(Reply::Json(ANSWERS)).await;
    let mut headers = BTreeMap::new();
    headers.insert("OpenAI-Project".to_string(), "proj_test".to_string());
    let judge = llm_wires::build_judge(
        Wire::OpenAi {
            endpoint: f.endpoint(),
            model: "gpt-6-luna".into(),
            headers,
        },
        Some(Secret::from("sk-test-key")),
    )
    .unwrap();

    let verdict = judge.judge(asked()).await.unwrap();

    let seen = f.request();
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.path, "/v1/decisions");
    assert_eq!(seen.header("authorization"), Some("Bearer sk-test-key"));
    assert_eq!(seen.header("openai-project"), Some("proj_test"));
    assert_eq!(
        seen.json(),
        json!({
            "model": "gpt-6-luna",
            "input": "Our CSV export stopped working after a field was renamed. Please fix it today, the board meets at 3pm.",
            "questions": [
                {
                    "type": "choice", "name": "q0",
                    "instructions": "Which team should take this?",
                    "choices": [
                        {"value": "billing", "description": "Payments, invoices, refunds."},
                        {"value": "engineering", "description": "Bugs and broken features."},
                        {"value": "sales", "description": "Pricing and new contracts."}
                    ]
                },
                {
                    "type": "score", "name": "q1",
                    "instructions": "How severe is the problem?",
                    "levels": [
                        {"label": "0", "description": "Cosmetic."},
                        {"label": "1", "description": "Annoying, has a workaround."},
                        {"label": "2", "description": "Blocks work."}
                    ]
                },
                {
                    "type": "predicate", "name": "q2",
                    "instructions": "Is the writer asking for something to be done urgently?"
                }
            ]
        })
    );
    // The caller's names never reach the model.
    for name in ["\"urgent\"", "\"lane\"", "\"severity\""] {
        assert!(!seen.body.contains(name), "{name} in {}", seen.body);
    }

    assert_eq!(verdict.model, "gpt-6-luna");
    assert_eq!(verdict.answers["urgent"], Answer::Noul { yes: 1.0 });
    assert_eq!(
        verdict.answers["lane"],
        Answer::Choice {
            choice: "engineering".into(),
            probabilities: [("billing", 0.0), ("engineering", 1.0), ("sales", 0.0)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            confidence: 1.0,
        }
    );
    assert_eq!(
        verdict.answers["severity"],
        Answer::Score {
            score: 1.97,
            probabilities: vec![0.0, 0.03, 0.97],
            confidence: 0.96,
        }
    );
    assert_eq!(
        verdict.usage,
        Usage {
            input: 429,
            ..Usage::default()
        }
    );
}

#[tokio::test]
async fn json_state_described_outcomes_and_images_go_in_one_user_message() {
    let f = fixture(Reply::Json(
        r#"{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"q0","probability":0.97}],"usage":{"input_tokens":377,"output_tokens":0}}"#,
    ))
    .await;
    let png = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/red-320.png"
    ))
    .unwrap();
    let judgement = Judgement::of(json!({"ticket": 42, "text": "charged twice"}))
        .image("photo", Image::from_bytes("image/png", &png))
        .ask(
            "refund",
            Question::noul_described(
                "Does the ticket ask for a refund, as `photo` shows?",
                "money back is asked for",
                json!({"not": "a question about money"}),
            ),
        );
    let verdict = judge_at(f.endpoint()).judge(judgement).await.unwrap();
    assert_eq!(verdict.answers["refund"], Answer::Noul { yes: 0.97 });

    let body = f.request().json();
    let content = &body["input"][0]["content"];
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(
        content[0],
        json!({"type": "input_json", "data": {"ticket": 42, "text": "charged twice"}})
    );
    assert_eq!(content[1], json!({"type": "input_text", "text": "photo"}));
    assert_eq!(content[2]["type"], "input_image");
    assert!(
        content[2]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,iVBOR")
    );
    assert_eq!(
        body["questions"][0]["instructions"],
        json!([
            {"type": "input_text", "text": "Does the ticket ask for a refund, as `photo` shows?"},
            {"type": "input_text", "text": "Yes means: money back is asked for"},
            {"type": "input_text", "text": "No means: {\"not\":\"a question about money\"}"}
        ])
    );
}

#[tokio::test]
async fn a_refusal_fails_the_judgement_by_the_callers_name_and_never_leaves_a_hole() {
    // Live: a two-option choice with no descriptions over the state "x".
    let f = fixture(Reply::Json(
        r#"{"model":"gpt-6-luna","answers":[{"type":"refusal","name":"q0"},{"type":"predicate","name":"q1","probability":0.2}],"usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":5}}"#,
    ))
    .await;
    let judgement = Judgement::of("x")
        .ask(
            "pick",
            Question::choice("?", [("a", json!(null)), ("b", json!(null))]),
        )
        .ask("sure", Question::noul("?"));
    let err = judge_at(f.endpoint()).judge(judgement).await.unwrap_err();
    match &err {
        Error::Refused { wire, questions } => {
            assert_eq!(*wire, "openai");
            assert_eq!(questions, &["pick".to_string()]);
        }
        other => panic!("expected Refused, got {other:?}"),
    }
    assert_eq!(err.to_string(), "openai: the model refused to decide pick");
    // An undescribed option goes out as its value alone.
    assert_eq!(
        f.request().json()["questions"][0]["choices"],
        json!([{"value": "a"}, {"value": "b"}])
    );
}

#[tokio::test]
async fn a_200_that_answers_a_kind_it_was_not_asked_or_skips_one_is_a_decode_error() {
    let f = fixture(Reply::Json(
        r#"{"model":"gpt-6-luna","answers":[{"type":"choice","name":"q0","choice":"a","probabilities":[{"value":"a","probability":1.0}],"confidence":1.0}]}"#,
    ))
    .await;
    match judge_at(f.endpoint())
        .judge(Judgement::of("x").ask("q", Question::noul("?")))
        .await
    {
        Err(Error::Decode(why)) => assert!(why.contains("is a predicate"), "{why}"),
        other => panic!("{other:?}"),
    }
    let f = fixture(Reply::Json(r#"{"model":"gpt-6-luna","answers":[]}"#)).await;
    match judge_at(f.endpoint()).judge(asked()).await {
        Err(Error::Decode(why)) => assert!(why.contains("no answer for question"), "{why}"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn the_apis_refusals_keep_their_code_and_a_429_its_retry_hint() {
    // Live, 2026-10-25: a one-level score sent past this wire's own check.
    let f = fixture(Reply::Headed(
        400,
        &[("x-request-id", "req_test400")],
        r#"{"error":{"message":"Invalid 'questions[0].levels': array too short. Expected an array with minimum length 2, but got an array with length 1 instead.","type":"invalid_request_error","param":"questions[0].levels","code":"array_below_min_length"}}"#,
    ))
    .await;
    let two = || Judgement::of("x").ask("s", Question::score("?", ["a", "b"]));
    match judge_at(f.endpoint()).judge(two()).await {
        Err(e @ Error::Api { .. }) => {
            assert_eq!(e.code(), Some("array_below_min_length"));
            assert!(e.to_string().contains("(request req_test400)"), "{e}");
        }
        other => panic!("{other:?}"),
    }

    // Live: a wrong key.
    let f = fixture(Reply::Status(
        401,
        r#"{"error":{"message":"Incorrect API key provided: sk-wrong. You can find your API key at https://platform.openai.com/account/api-keys.","type":"invalid_request_error","code":"invalid_api_key","param":null},"status":401}"#,
    ))
    .await;
    match judge_at(f.endpoint()).judge(two()).await {
        Err(e @ Error::Api { status: 401, .. }) => {
            assert_eq!(e.code(), Some("invalid_api_key"))
        }
        other => panic!("{other:?}"),
    }

    let f = fixture(Reply::Headed(
        429,
        &[("retry-after", "3")],
        r#"{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}"#,
    ))
    .await;
    match judge_at(f.endpoint()).judge(two()).await {
        Err(Error::Api {
            status: 429,
            retry_after,
            ..
        }) => assert_eq!(retry_after, Some(Duration::from_secs(3))),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_request_the_api_would_refuse_never_leaves() {
    // Nothing listens here; a request that got out would be a transport error.
    let endpoint = llm_wires::testing::closed_endpoint().await;
    let judge = judge_at(endpoint);
    let invalid = |r: Result<_, Error>| match r {
        Err(Error::Invalid {
            wire: "openai",
            what,
        }) => what,
        other => panic!("expected Invalid, got {:?}", other.err()),
    };
    assert!(
        invalid(
            judge
                .judge(Judgement::of("x").ask("b", Question::locate("the mug")))
                .await
        )
        .contains("no box question")
    );
    assert!(
        invalid(
            judge
                .judge(Judgement::of("x").ask("s", Question::score("?", ["one"])))
                .await
        )
        .contains("2 to 10 levels")
    );
    let eleven: Vec<String> = (0..11).map(|i| i.to_string()).collect();
    assert!(
        invalid(
            judge
                .judge(Judgement::of("x").ask("s", Question::score("?", eleven)))
                .await
        )
        .contains("2 to 10 levels")
    );
    assert!(invalid(judge.judge(Judgement::of("x")).await).contains("at least one question"));
}

#[tokio::test]
async fn a_choice_missing_an_options_probability_is_a_decode_error() {
    // An option absent from the map would read as never considered.
    let f = fixture(Reply::Json(
        r#"{"model":"gpt-6-luna","answers":[{"type":"choice","name":"q0","choice":"a","probabilities":[{"value":"a","probability":1.0}],"confidence":1.0}]}"#,
    ))
    .await;
    match judge_at(f.endpoint())
        .judge(Judgement::of("x").ask("c", Question::choice("?", [("a", "A"), ("b", "B")])))
        .await
    {
        Err(Error::Decode(why)) => assert!(why.contains("option \"b\""), "{why}"),
        other => panic!("{other:?}"),
    }
}
