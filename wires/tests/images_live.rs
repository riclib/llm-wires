//! One real round trip with an image on each wire that takes one, for the
//! day the pins need checking against the servers.
//!
//! Ignored by default: each needs a key and spends a fraction of a cent.
//! Run them by hand with
//!
//! ```sh
//! TYPESAFE_KEY=… ANTHROPIC_API_KEY=… OPENAI_API_KEY=… \
//!   cargo test -p llm-wires --test images_live -- --ignored --nocapture
//! ```
//!
//! The images are the fixtures `tests/images.rs` pins against: a solid red
//! square, and a blue square on white at left 0.25–0.5, top 0.5–0.75.

use llm_wires::{Answer, ChatRequest, Image, Judgement, Message, Question, Wire};
use serde_json::json;
use wire_secret::Secret;

const RED: &[u8] = include_bytes!("fixtures/red-320.png");
const BLUE_SQUARE: &[u8] = include_bytes!("fixtures/blue-square-400.png");

fn key(name: &str) -> Secret {
    let key = std::env::var(name).unwrap_or_else(|_| panic!("{name} in the environment"));
    Secret::from(key.trim())
}

#[tokio::test]
#[ignore = "spends money; needs TYPESAFE_KEY"]
async fn jev_sees_red_and_finds_the_blue_square() {
    let judge = |model: &str| {
        llm_wires::build_judge(
            Wire::typesafe("https://api.typesafe.ai", model),
            Some(key("TYPESAFE_KEY")),
        )
        .unwrap()
    };

    let v = judge("jev-latest")
        .judge(
            Judgement::of(json!({"photo": "ref-0c9e1d2a-red"}))
                .image("ref-0c9e1d2a-red", Image::from_bytes("image/png", RED))
                .ask("red", Question::noul("Is `photo` mostly red?")),
        )
        .await
        .expect("jev-latest takes an image");
    println!("{v:#?}");
    match v.answers["red"] {
        Answer::Noul { yes } => assert!(yes > 0.9, "{yes}"),
        ref other => panic!("{other:?}"),
    }

    // Box needs Jev 1.14; jev-latest answered 400 "Invalid request." to one
    // on 2026-10-21.
    let v = judge("jev-preview")
        .judge(
            Judgement::of(json!({"photo": "ref-7b3f4c5d-square"}))
                .image(
                    "ref-7b3f4c5d-square",
                    Image::from_bytes("image/png", BLUE_SQUARE),
                )
                .ask("square", Question::locate("the blue square in `photo`")),
        )
        .await
        .expect("jev-preview answers a box");
    println!("{v:#?}");
    match v.answers["square"] {
        Answer::Box {
            top,
            bottom,
            left,
            right,
            ..
        } => {
            // The truth is 0.5 / 0.75 / 0.25 / 0.5; within a tenth is found.
            for (got, want) in [(top, 0.5), (bottom, 0.75), (left, 0.25), (right, 0.5)] {
                assert!((got - want).abs() < 0.1, "{got} vs {want}");
            }
        }
        ref other => panic!("{other:?}"),
    }
}

fn what_colour() -> ChatRequest {
    let mut req = ChatRequest::ask(
        "Answer with one lowercase word.",
        "What colour fills this image?",
    );
    req.messages[0] = Message::user("What colour fills this image?")
        .with_images([Image::from_bytes("image/png", RED)]);
    req
}

#[tokio::test]
#[ignore = "spends money; needs ANTHROPIC_API_KEY"]
async fn claude_sees_red() {
    let answer = llm_wires::build(
        Wire::anthropic("https://api.anthropic.com", "claude-haiku-4-5"),
        Some(key("ANTHROPIC_API_KEY")),
    )
    .unwrap()
    .chat(what_colour())
    .await
    .expect("anthropic takes an image block");
    println!("{:?}", answer.message.content);
    assert!(answer.message.content.to_lowercase().contains("red"));
}

#[tokio::test]
#[ignore = "spends money; needs OPENAI_API_KEY"]
async fn gpt_sees_red() {
    let answer = llm_wires::build(
        Wire::openai("https://api.openai.com/v1", "gpt-4o-mini"),
        Some(key("OPENAI_API_KEY")),
    )
    .unwrap()
    .chat(what_colour())
    .await
    .expect("openai takes an image_url part");
    println!("{:?}", answer.message.content);
    assert!(answer.message.content.to_lowercase().contains("red"));
}
