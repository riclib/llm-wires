//! OpenRouter's Decisions API, against a real socket.
//!
//! The body and the answers are System One's, field for field, so the pins
//! here are what differs: the path, the headers that do not go, the cost
//! OpenRouter reports, and the two ways a refusal arrives. The replies are
//! OpenRouter's, recorded on 2026-11-23 against `api/alpha/decisions`.

use std::collections::BTreeMap;

use llm_wires::testing::{Reply, fixture};
use llm_wires::{Answer, Error, Judgement, Question, Usage, Wire};
use serde_json::json;
use wire_secret::Secret;

fn asked() -> Judgement {
    Judgement::of("Help! My payouts have been failing for 3 days.")
        .ask("is_urgent", Question::noul("Does this convey urgency?"))
        .ask(
            "department",
            Question::choice(
                "Which team?",
                [("billing", "Payments"), ("technical", "Bugs")],
            ),
        )
        .ask(
            "frustration",
            Question::score("How frustrated?", ["Calm", "Frustrated", "Very angry"]),
        )
}

/// `typesafe/jev-1.14` through OpenRouter, byte for byte.
const JEV: &str = r#"{"model":"typesafe/jev-1.14-20261014","answers":{"is_urgent":{"type":"noul","noul":0.95},"department":{"type":"choice","choice":"billing","probabilities":{"billing":0.96,"technical":0.04},"confidence":0.92},"frustration":{"type":"score","score":1.01,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0,"1":0.99,"2":0.01},"confidence":0.98}},"usage":{"input_tokens":366,"output_tokens":66,"cost":0.000015372,"is_byok":false,"cost_details":{"upstream_inference_cost":0.000015372,"upstream_inference_prompt_cost":0.000015372,"upstream_inference_completions_cost":0}},"id":"gen-dec-1795422758-VjtwTE8AdW3UysFYYWum","provider":"TypeSafe"}"#;

#[tokio::test]
async fn the_system_one_body_goes_to_decisions_with_a_bearer_and_no_typesafe_headers() {
    let f = fixture(Reply::Json(JEV)).await;
    let mut headers = BTreeMap::new();
    headers.insert(
        "HTTP-Referer".to_string(),
        "https://solid.example".to_string(),
    );
    let judge = llm_wires::build_judge(
        Wire::OpenRouterDecisions {
            endpoint: f.endpoint_at("/api/alpha"),
            model: "typesafe/jev-1.14".into(),
            headers,
        },
        Some(Secret::from("sk-or-test")),
    )
    .unwrap();
    assert_eq!(judge.info().wire, "openrouter_decisions");

    let verdict = judge.judge(asked()).await.unwrap();
    assert_eq!(verdict.model, "typesafe/jev-1.14-20261014");
    assert_eq!(verdict.answers["is_urgent"], Answer::Noul { yes: 0.95 });
    assert_eq!(
        verdict.answers["frustration"],
        Answer::Score {
            score: 1.01,
            probabilities: vec![0.0, 0.99, 0.01],
            confidence: 0.98,
        }
    );
    // What OpenRouter billed, in dollars, as it said.
    assert_eq!(verdict.cost, Some(0.000015372));
    assert_eq!(
        verdict.usage,
        Usage {
            input: 366,
            output: 66,
            ..Usage::default()
        }
    );

    let seen = f.request();
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.path, "/api/alpha/decisions");
    assert_eq!(seen.header("authorization"), Some("Bearer sk-or-test"));
    assert_eq!(seen.header("http-referer"), Some("https://solid.example"));
    // Nothing of TypeSafe's SDK: this crate is not their SDK to OpenRouter.
    for (name, _) in &seen.headers {
        assert!(!name.starts_with("x-typesafe-"), "{name} went out");
        assert!(!name.starts_with("typesafe-"), "{name} went out");
    }
    // The body TypeSafe's wire sends, with OpenRouter's model name.
    assert_eq!(
        seen.json(),
        json!({
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "typesafe/jev-1.14",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                "department": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"billing": "Payments", "technical": "Bugs"}},
                "frustration": {"type": "score", "instructions": "How frustrated?",
                                "criteria": ["Calm", "Frustrated", "Very angry"]}
            }
        })
    );
}

#[tokio::test]
async fn a_refusal_beside_the_answers_is_refused_naming_it_on_both_arms() {
    // A question the model will not decide, answered as `{"type":
    // "refusal"}` beside the others: never a verdict with a hole in it.
    const BODY: &str = r#"{"model":"m","answers":{"ok":{"type":"noul","noul":0.93},"which":{"type":"refusal"}},"usage":{"input_tokens":1,"output_tokens":0}}"#;
    for (wire, name) in [
        (
            Wire::openrouter_decisions("", "typesafe/jev-1.14"),
            "openrouter_decisions",
        ),
        (Wire::typesafe("", "jev-latest"), "typesafe"),
    ] {
        let f = fixture(Reply::Json(BODY)).await;
        let wire = match wire {
            Wire::OpenRouterDecisions { model, headers, .. } => Wire::OpenRouterDecisions {
                endpoint: f.endpoint_at("/api/alpha"),
                model,
                headers,
            },
            Wire::TypeSafe { model, headers, .. } => Wire::TypeSafe {
                endpoint: f.root(),
                model,
                headers,
            },
            other => unreachable!("{other:?}"),
        };
        let err = llm_wires::build_judge(wire, Some(Secret::from("k")))
            .unwrap()
            .judge(
                Judgement::of("x")
                    .ask("ok", Question::noul("?"))
                    .ask("which", Question::choice("?", [("a", "A"), ("b", "B")])),
            )
            .await
            .expect_err("a refusal is an error");
        match &err {
            Error::Refused { wire, questions } => {
                assert_eq!(*wire, name);
                assert_eq!(questions, &vec!["which".to_string()]);
            }
            other => panic!("{name}: expected Refused, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn an_upstream_refusal_is_openrouters_502_with_its_sentence_and_generation_id() {
    // What OpenRouter answered on 2026-11-23 when OpenAI's decider would not
    // answer one question: not a refusal beside the answers but a 502 for the
    // whole call, the question named only inside a sentence. It stays the
    // provider's no, with the generation id to quote to their support; its
    // `code` is the status again and names nothing.
    let f = fixture(Reply::Headed(
        502,
        &[
            ("x-generation-id", "gen-dec-1795422771-wKL3C8JmmOO5LFIp3mP1"),
            ("cf-ray", "a5eed3607967a863-RIX"),
        ],
        r#"{"error":{"message":"OpenAI refused to answer question \"which_person\"","code":502}}"#,
    ))
    .await;
    let err = llm_wires::build_judge(
        Wire::openrouter_decisions(f.endpoint_at("/api/alpha"), "openai/gpt-6-luna-decisions"),
        Some(Secret::from("k")),
    )
    .unwrap()
    .judge(Judgement::of("x").ask("which_person", Question::noul("?")))
    .await
    .expect_err("a 502 is an error");
    match &err {
        Error::Api {
            status,
            message,
            request_id,
            ..
        } => {
            assert_eq!(*status, 502);
            assert_eq!(
                message,
                "OpenAI refused to answer question \"which_person\""
            );
            assert_eq!(
                request_id.as_deref(),
                Some("gen-dec-1795422771-wKL3C8JmmOO5LFIp3mP1")
            );
        }
        other => panic!("expected Api, got {other:?}"),
    }
}

#[tokio::test]
async fn typesafe_itself_reports_no_cost_and_none_is_made_up() {
    let f = fixture(Reply::Json(
        r#"{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":366,"output_tokens":66}}"#,
    ))
    .await;
    let v = llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-latest"),
        Some(Secret::from("k")),
    )
    .unwrap()
    .judge(Judgement::of("x").ask("q", Question::noul("?")))
    .await
    .unwrap();
    assert_eq!(v.cost, None);
}

#[test]
fn the_wire_is_a_judge_only_and_needs_its_endpoint() {
    let wire = || Wire::openrouter_decisions(llm_wires::OPENROUTER_DECISIONS_ENDPOINT, "m");
    for (verb, got) in [
        (
            "chat",
            llm_wires::build(wire(), Some(Secret::from("k"))).err(),
        ),
        (
            "embed",
            llm_wires::build_embed(wire(), Some(Secret::from("k"))).err(),
        ),
        (
            "extract",
            llm_wires::build_extract(wire(), Some(Secret::from("k"))).err(),
        ),
    ] {
        match got {
            Some(Error::Cannot { wire, verb: v }) => {
                assert_eq!((wire, v), ("openrouter_decisions", verb))
            }
            other => panic!("{verb}: expected Cannot, got {other:?}"),
        }
    }
    let judge = llm_wires::build_judge(wire(), Some(Secret::from("k"))).unwrap();
    assert_eq!(judge.info().endpoint, "https://openrouter.ai/api/alpha");
    // No default inside the crate: the card fills the constant in.
    match llm_wires::build_judge(Wire::openrouter_decisions("", "m"), Some(Secret::from("k"))) {
        Err(Error::Missing { wire, field }) => {
            assert_eq!((wire, field), ("openrouter_decisions", "endpoint"))
        }
        other => panic!("expected Missing, got {:?}", other.err()),
    }
    match llm_wires::build_judge(wire(), None) {
        Err(Error::NoKey(w)) => assert_eq!(w, "openrouter_decisions"),
        other => panic!("expected NoKey, got {:?}", other.err()),
    }
}
