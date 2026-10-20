//! One real round trip to Clairic on Workers AI, for the day the pins need
//! checking against the server and not the docs' examples.
//!
//! Ignored by default: it needs a token and an account and spends money (a
//! fraction of a cent: Clairic bills input tokens only). Run it by hand with
//!
//! ```sh
//! CLOUDFLARE_ACCOUNT_ID=… CLOUDFLARE_AUTH_TOKEN=… \
//!   cargo test -p llm-wires --test workers_ai_live -- --ignored --nocapture
//! ```
//!
//! The names are the ones Cloudflare's own examples read. Where what prints
//! here disagrees with the docs, the live answer wins: change the wire and
//! the recorded fixtures in `tests/workers_ai.rs`, and say so in the pin.

use llm_wires::{DurationUnit, Error, Extraction, Fact, Field, FieldType, Item, Wire};
use wire_secret::Secret;

const ENDPOINT: &str = "https://api.cloudflare.com/client/v4";
const MODEL: &str = "@cf/cloudflare/clairic";

fn env(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("{name} in the environment"))
        .trim()
        .to_string()
}

#[tokio::test]
#[ignore = "spends money; needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN"]
async fn the_posts_email_answers_701_10_and_no_tracking_number() {
    let extract = llm_wires::build_extract(
        Wire::workers_ai(ENDPOINT, env("CLOUDFLARE_ACCOUNT_ID"), MODEL),
        Some(Secret::from(env("CLOUDFLARE_AUTH_TOKEN").as_str())),
    )
    .unwrap();

    let got = extract
        .extract(
            Extraction::of(
                "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented \
                 corner. I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this \
                 fixed or refunded. Thanks, Dana",
            )
            .field(
                "order_id",
                Field::from(FieldType::Text).instructions("Order number"),
            )
            .field(
                "delivered_on",
                Field::from(FieldType::Date).instructions("Delivery date"),
            )
            .field(
                "amount_paid",
                Field::from(FieldType::Number {
                    integer: false,
                    decimals: Some(2),
                    unit: Some("USD".into()),
                })
                .instructions("Total paid including tax"),
            )
            .field(
                "days_since_delivery",
                Field::from(FieldType::Duration {
                    unit: Some(DurationUnit::Days),
                    integer: true,
                    decimals: None,
                })
                .instructions("Days from delivery to the date the customer wrote"),
            )
            .field(
                "damage",
                Field::from(FieldType::list(Item::Text))
                    .instructions("Each kind of damage described"),
            )
            .field("tracking_number", FieldType::Text)
            .field(
                "repair_cost",
                Field::from(FieldType::Estimate {
                    min: 10.into(),
                    max: 5000.into(),
                    step: Some(llm_wires::Step::By(1.0)),
                    scale: llm_wires::Scale::Log,
                    unit: Some("USD".into()),
                })
                .instructions("Cost to repair the damage described"),
            ),
        )
        .await
        .expect("the real server answers the post's own example");
    println!("{got:#?}");

    assert_eq!(got.answers.len(), 7);
    assert!(got.usage.input > 0);
    match &got.answers["amount_paid"] {
        Fact::Number {
            value: Some(paid), ..
        } => assert!((paid - 701.1).abs() < 1e-9, "{paid}"),
        other => panic!("{other:?}"),
    }
    let tracking = &got.answers["tracking_number"];
    assert!(tracking.is_null(), "{tracking:?}");
    match &got.answers["repair_cost"] {
        Fact::Estimate { p10, p90, .. } => {
            assert!(p10.as_f64().unwrap() <= p90.as_f64().unwrap());
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
#[ignore = "needs the network and CLOUDFLARE_ACCOUNT_ID"]
async fn a_bad_token_is_their_no_with_their_words_and_the_ray() {
    let extract = llm_wires::build_extract(
        Wire::workers_ai(ENDPOINT, env("CLOUDFLARE_ACCOUNT_ID"), MODEL),
        Some(Secret::from("cf-not-a-real-token")),
    )
    .unwrap();
    let err = extract
        .extract(Extraction::of("x").field("f", FieldType::Text))
        .await
        .expect_err("a made-up token is refused");
    println!("{err}");
    match &err {
        Error::Api {
            status, request_id, ..
        } => {
            assert!([401, 403].contains(status), "{status}");
            assert!(request_id.is_some(), "no cf-ray on {err}");
        }
        other => panic!("{other:?}"),
    }
}
