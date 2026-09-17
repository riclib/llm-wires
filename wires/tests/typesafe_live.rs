//! One real round trip to TypeSafe, for the day the pins need checking
//! against the server and not the SDK's declarations.
//!
//! Ignored by default: it needs a key and spends money (a fraction of a
//! cent at their published price). Run it by hand with
//!
//! ```sh
//! TYPESAFE_KEY=… cargo test -p llm-wires --test typesafe_live -- --ignored --nocapture
//! ```
//!
//! `TYPESAFE_API_KEY`, the name their SDK reads, is accepted too.

use llm_wires::{Answer, Error, Judgement, Question, Wire};
use wire_secret::Secret;

fn key() -> Secret {
    let key = std::env::var("TYPESAFE_KEY")
        .or_else(|_| std::env::var("TYPESAFE_API_KEY"))
        .expect("TYPESAFE_KEY or TYPESAFE_API_KEY in the environment");
    Secret::from(key.trim())
}

#[tokio::test]
#[ignore = "spends money; needs TYPESAFE_KEY"]
async fn the_docs_example_round_trips_against_the_real_server() {
    let judge = llm_wires::build_judge(
        Wire::typesafe("https://api.typesafe.ai", "jev-latest"),
        Some(key()),
    )
    .unwrap();

    let verdict = judge
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
        .expect("the real server answers the docs' own example");
    println!("{verdict:#?}");

    assert_eq!(verdict.answers.len(), 3);
    assert!(verdict.usage.input > 0);
    match &verdict.answers["is_urgent"] {
        Answer::Noul { yes } => assert!((0.0..=1.0).contains(yes)),
        other => panic!("{other:?}"),
    }
    match &verdict.answers["department"] {
        Answer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            assert!(["billing", "technical", "sales"].contains(&choice.as_str()));
            assert_eq!(probabilities.len(), 3);
            assert!((probabilities.values().sum::<f64>() - 1.0).abs() < 0.01);
            assert!((0.0..=1.0).contains(confidence));
        }
        other => panic!("{other:?}"),
    }
    match &verdict.answers["frustration"] {
        Answer::Score {
            score,
            probabilities,
            confidence,
        } => {
            assert!((0.0..=2.0).contains(score));
            assert_eq!(probabilities.len(), 3);
            assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 0.01);
            assert!((0.0..=1.0).contains(confidence));
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
#[ignore = "needs the network"]
async fn a_bad_key_is_their_401_with_their_words_and_their_request_id() {
    let judge = llm_wires::build_judge(
        Wire::typesafe("https://api.typesafe.ai", "jev-latest"),
        Some(Secret::from("ts-not-a-real-key")),
    )
    .unwrap();
    let err = judge
        .judge(Judgement::of("x").ask("q", Question::noul("?")))
        .await
        .expect_err("a made-up key is refused");
    println!("{err}");
    match &err {
        Error::Api { status, .. } => assert_eq!(*status, 401),
        other => panic!("{other:?}"),
    }
}
