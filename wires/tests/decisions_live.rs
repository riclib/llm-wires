//! Real round trips to OpenAI's Decisions API, for the day the pins in
//! `decisions.rs` need checking against the server.
//!
//! Ignored by default: they need a key and spend money (a fraction of a cent
//! at $0.10 per million input tokens). Run them by hand with
//!
//! ```sh
//! OPENAI_API_KEY=… cargo test -p llm-wires --test decisions_live -- --ignored --nocapture
//! ```

use llm_wires::{Answer, Image, Judgement, Question, Wire};
use serde_json::json;
use wire_secret::Secret;

fn judge() -> Box<dyn llm_wires::Judge> {
    let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY in the environment");
    llm_wires::build_judge(
        Wire::openai("https://api.openai.com/v1", "gpt-6-luna"),
        Some(Secret::from(key.trim())),
    )
    .unwrap()
}

#[tokio::test]
#[ignore = "spends money; needs OPENAI_API_KEY"]
async fn the_three_kinds_round_trip_against_the_real_server() {
    let verdict = judge()
        .judge(
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
                ),
        )
        .await
        .unwrap();
    println!("{verdict:#?}");
    assert!(verdict.model.starts_with("gpt-6-luna"), "{}", verdict.model);
    match &verdict.answers["is_urgent"] {
        Answer::Noul { yes } => assert!(*yes > 0.5, "{yes}"),
        other => panic!("{other:?}"),
    }
    match &verdict.answers["frustration"] {
        Answer::Score { probabilities, .. } => assert_eq!(probabilities.len(), 3),
        other => panic!("{other:?}"),
    }
    assert!(verdict.usage.input > 0);
}

#[tokio::test]
#[ignore = "spends money; needs OPENAI_API_KEY"]
async fn json_state_described_outcomes_and_an_image_round_trip() {
    let png = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/red-320.png"
    ))
    .unwrap();
    let verdict = judge()
        .judge(
            Judgement::of(json!({"ticket": 42, "text": "I was charged twice, refund one please"}))
                .image("photo", Image::from_bytes("image/png", &png))
                .ask(
                    "refund",
                    Question::noul_described(
                        "Does the ticket ask for a refund?",
                        "money back is asked for",
                        "anything else",
                    ),
                )
                .ask(
                    "colour",
                    Question::choice(
                        "What colour is the image `photo`?",
                        [("red", "mostly red"), ("blue", "mostly blue")],
                    ),
                )
                .ask(
                    "undescribed",
                    Question::Noul {
                        instructions: json!(null),
                        yes: json!("the ticket is about money"),
                        no: json!(null),
                    },
                ),
        )
        .await
        .unwrap();
    println!("{verdict:#?}");
    match &verdict.answers["refund"] {
        Answer::Noul { yes } => assert!(*yes > 0.5, "{yes}"),
        other => panic!("{other:?}"),
    }
    match &verdict.answers["colour"] {
        Answer::Choice { choice, .. } => assert_eq!(choice, "red"),
        other => panic!("{other:?}"),
    }
}
