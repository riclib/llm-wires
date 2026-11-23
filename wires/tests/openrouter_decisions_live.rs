//! One real round trip per decider on OpenRouter's Decisions API, for the
//! day the pins need checking against the server.
//!
//! Ignored by default: it needs a key and spends a fraction of a cent (about
//! $0.00002 a call on 2026-11-23). Run it by hand with
//!
//! ```sh
//! OPENROUTER_API_KEY=… cargo test -p llm-wires --test openrouter_decisions_live -- --ignored --nocapture
//! ```

use llm_wires::{Answer, Judgement, OPENROUTER_DECISIONS_ENDPOINT, Question, Wire};
use wire_secret::Secret;

fn key() -> Secret {
    let key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY in the environment");
    Secret::from(key.trim())
}

#[tokio::test]
#[ignore = "spends money; needs OPENROUTER_API_KEY"]
async fn every_decider_answers_the_docs_example_and_says_what_it_cost() {
    for model in [
        "typesafe/jev-1.14",
        "cloudflare/clef-flash",
        "openai/gpt-6-luna-decisions",
    ] {
        let verdict = llm_wires::build_judge(
            Wire::openrouter_decisions(OPENROUTER_DECISIONS_ENDPOINT, model),
            Some(key()),
        )
        .unwrap()
        .judge(
            Judgement::of("Help! My payouts have been failing for 3 days.")
                .ask("is_urgent", Question::noul("Does this convey urgency?"))
                .ask(
                    "department",
                    Question::choice(
                        "Which team?",
                        [("billing", "Payments"), ("technical", "Bugs")],
                    ),
                ),
        )
        .await
        .unwrap_or_else(|e| panic!("{model}: {e}"));
        println!("{model}: {verdict:?}");

        match verdict.answers["is_urgent"] {
            Answer::Noul { yes } => assert!(yes > 0.5, "{model}: {yes}"),
            ref other => panic!("{model}: {other:?}"),
        }
        match &verdict.answers["department"] {
            Answer::Choice { choice, .. } => assert_eq!(choice, "billing", "{model}"),
            other => panic!("{model}: {other:?}"),
        }
        let cost = verdict.cost.unwrap_or_else(|| panic!("{model}: no cost"));
        assert!(cost > 0.0 && cost < 0.001, "{model}: {cost}");
    }
}
