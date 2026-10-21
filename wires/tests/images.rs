//! Images on every wire that takes them, against a real socket.
//!
//! One `Image` and three spellings of it: TypeSafe's `assets` with the
//! images header, Anthropic's `image` block, OpenAI's `image_url` part. The
//! pins are the bytes each wire sends, the refusals each makes before the
//! socket, and — for TypeSafe — the box answer, from a reply recorded live.
//! The fixture PNGs are real files: `red-320.png` is a solid red square,
//! `blue-square-400.png` white with a blue square at left 0.25–0.5, top
//! 0.5–0.75.

use llm_wires::testing::{Reply, fixture};
use llm_wires::{Answer, ChatRequest, Error, Image, Judgement, Message, Question, Role, Wire};
use serde_json::json;
use wire_secret::Secret;

const RED: &[u8] = include_bytes!("fixtures/red-320.png");
const BLUE_SQUARE: &[u8] = include_bytes!("fixtures/blue-square-400.png");

fn b64(bytes: &[u8]) -> String {
    match Image::from_bytes("image/png", bytes) {
        Image::Base64 { base64, .. } => base64,
        Image::DataUrl(_) => unreachable!(),
    }
}

/// A PNG's signature and IHDR at the given size: all the wire reads before
/// it refuses one.
fn png_header(width: u32, height: u32) -> Vec<u8> {
    let mut b = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
    b.extend(width.to_be_bytes());
    b.extend(height.to_be_bytes());
    b.extend([8, 2, 0, 0, 0]);
    b
}

// ---------------------------------------------------------------- TypeSafe

/// jev-preview's answer to the blue square, recorded 2026-10-21 against
/// api.typesafe.ai, byte for byte.
const BOX_ANSWER: &str = r#"{"model": "jev-1.14.0", "answers": {"square": {"type": "box", "box": {"top": 0.54, "bottom": 0.75, "left": 0.25, "right": 0.46}, "confidence": 0.96}, "there": {"type": "noul", "noul": 0.99}}, "usage": {"input_tokens": 758, "output_tokens": 36}}"#;

#[tokio::test]
async fn an_image_and_a_box_go_out_as_assets_with_the_images_header() {
    let f = fixture(Reply::Json(BOX_ANSWER)).await;
    let judge = llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-preview"),
        Some(Secret::from("ts-k")),
    )
    .unwrap();
    let verdict = judge
        .judge(
            Judgement::of(json!({"photo": "ref-5e6f7a8b"}))
                .image("ref-5e6f7a8b", Image::from_bytes("image/png", BLUE_SQUARE))
                .ask("square", Question::locate("the blue square in `photo`"))
                .ask(
                    "there",
                    Question::noul("Is there a blue square in `photo`?"),
                ),
        )
        .await
        .unwrap();

    assert_eq!(
        verdict.answers["square"],
        Answer::Box {
            top: 0.54,
            bottom: 0.75,
            left: 0.25,
            right: 0.46,
            confidence: Some(0.96),
        }
    );
    assert_eq!(verdict.answers["there"], Answer::Noul { yes: 0.99 });

    let seen = f.request();
    assert_eq!(seen.header("typesafe-danger-zone-images"), Some("true"));
    assert_eq!(
        seen.json(),
        json!({
            "state": {"photo": "ref-5e6f7a8b"},
            "model": "jev-preview",
            "questions": {
                // A box takes no criteria.
                "square": {"type": "box", "instructions": "the blue square in `photo`"},
                "there": {"type": "noul", "instructions": "Is there a blue square in `photo`?"}
            },
            "assets": {
                // Raw base64, no `data:` prefix, the format read from the bytes.
                "ref-5e6f7a8b": {"type": "image", "format": "png", "data_base64": b64(BLUE_SQUARE)}
            }
        })
    );
}

#[tokio::test]
async fn a_judgement_without_images_sends_neither_assets_nor_the_header() {
    let f = fixture(Reply::Json(
        r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
    ))
    .await;
    llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-latest"),
        Some(Secret::from("k")),
    )
    .unwrap()
    .judge(Judgement::of("x").ask("q", Question::noul("?")))
    .await
    .unwrap();
    let seen = f.request();
    assert_eq!(seen.header("typesafe-danger-zone-images"), None);
    assert!(seen.json().get("assets").is_none(), "{}", seen.body);
}

#[tokio::test]
async fn a_box_with_no_confidence_is_still_a_box() {
    let f = fixture(Reply::Json(
        r#"{"model":"m","answers":{"b":{"type":"box","box":{"top":0.1,"bottom":0.2,"left":0.3,"right":0.4},"confidence":null}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
    ))
    .await;
    let v = llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-preview"),
        Some(Secret::from("k")),
    )
    .unwrap()
    .judge(
        Judgement::of("ref-1")
            .image("ref-1", Image::from_bytes("image/png", RED))
            .ask("b", Question::locate("the thing")),
    )
    .await
    .unwrap();
    assert_eq!(
        v.answers["b"],
        Answer::Box {
            top: 0.1,
            bottom: 0.2,
            left: 0.3,
            right: 0.4,
            confidence: None,
        }
    );
}

#[tokio::test]
async fn what_typesafe_would_refuse_is_refused_before_the_socket_naming_the_asset() {
    let at = |bytes: Vec<u8>| Image::from_bytes("image/png", &bytes);
    let red = || Image::from_bytes("image/png", RED);
    let cases: Vec<(Judgement, &str)> = vec![
        // Measured live on 2026-10-21: 279 and 1569 a side are refused, 280
        // and 1568 answered — the documented bounds, inclusive.
        (
            Judgement::of("ref-1").image("ref-1", at(png_header(279, 279))),
            r#"asset "ref-1": 279 × 279 pixels; each side must be 280 to 1568"#,
        ),
        (
            Judgement::of("ref-1").image("ref-1", at(png_header(1568, 1569))),
            "1568 × 1569 pixels",
        ),
        // Referenced nowhere: a 400 from the server.
        (
            Judgement::of("no reference here").image("ref-1", red()),
            r#"asset "ref-1": not referenced"#,
        ),
        // A choice's label is an object key, which the server does not
        // replace: a reference there is not a reference.
        (
            Judgement::of("x").image("ref-1", red()).ask(
                "q",
                Question::choice("?", [("ref-1", "a label, not a reference")]),
            ),
            "not referenced",
        ),
        (
            Judgement::of("x").image(" ", red()),
            "a reference cannot be empty",
        ),
        (
            Judgement::of("ref-1").image("ref-1", at(b"GIF89a\x01\x00\x01\x00".to_vec())),
            "not a PNG or a JPEG",
        ),
        (
            Judgement::of("ref-1").image(
                "ref-1",
                Image::Base64 {
                    content_type: "image/png".into(),
                    base64: "iVBO Rw0K".into(),
                },
            ),
            "not standard base64",
        ),
        (
            Judgement::of("ref-1")
                .image("ref-1", Image::DataUrl("data:image/svg+xml,<svg/>".into())),
            "a data URL that is not base64",
        ),
    ];
    for (req, want) in cases {
        // A closed port: anything that reached the socket would be a
        // different error.
        let judge = llm_wires::build_judge(
            Wire::typesafe("http://127.0.0.1:9", "jev-latest"),
            Some(Secret::from("k")),
        )
        .unwrap();
        match judge.judge(req.ask("q", Question::noul("?"))).await {
            Err(Error::Invalid { wire, what }) => {
                assert_eq!(wire, "typesafe");
                assert!(what.contains(want), "{what}\nwanted {want}");
            }
            other => panic!("expected Invalid {want:?}, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_reference_inside_a_criterion_or_a_sentence_counts() {
    let f = fixture(Reply::Json(
        r#"{"model":"m","answers":{"q":{"type":"choice","choice":"a","confidence":1,"probabilities":{"a":1,"b":0}}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
    ))
    .await;
    llm_wires::build_judge(
        Wire::typesafe(f.root(), "jev-latest"),
        Some(Secret::from("k")),
    )
    .unwrap()
    .judge(
        Judgement::of("a photo")
            .image("ref-in-criterion", Image::from_bytes("image/png", RED))
            .image(
                "ref-mid-sentence",
                Image::DataUrl(format!("data:image/png;base64,{}", b64(BLUE_SQUARE))),
            )
            .ask(
                "q",
                Question::choice(
                    "Which matches this: ref-mid-sentence?",
                    [("a", "ref-in-criterion"), ("b", "neither")],
                ),
            ),
    )
    .await
    .unwrap();
    let assets = f.request().json()["assets"].clone();
    // A data URL goes out as its base64 alone.
    assert_eq!(
        assets["ref-mid-sentence"],
        json!({"type": "image", "format": "png", "data_base64": b64(BLUE_SQUARE)})
    );
    assert_eq!(assets["ref-in-criterion"]["format"], "png");
}

// ------------------------------------------------------------------- chat

fn shown() -> ChatRequest {
    let mut req = ChatRequest::ask("You are terse.", "What colour is this?");
    req.messages[0] =
        Message::user("What colour is this?").with_images([Image::from_bytes("image/png", RED)]);
    req
}

const ANTHROPIC_REPLY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[{"type":"text","text":"Red."}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":2}}"#;
const OPENAI_REPLY: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Red."},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}"#;

#[tokio::test]
async fn anthropic_gets_a_base64_image_block_before_the_text() {
    let f = fixture(Reply::Json(ANTHROPIC_REPLY)).await;
    let answer = llm_wires::build(
        Wire::anthropic(f.root(), "claude-haiku-4-5"),
        Some(Secret::from("k")),
    )
    .unwrap()
    .chat(shown())
    .await
    .unwrap();
    assert_eq!(answer.message.content, "Red.");
    assert!(answer.message.images.is_empty());
    assert_eq!(
        f.request().json()["messages"],
        json!([{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(RED)}},
            {"type": "text", "text": "What colour is this?"}
        ]}])
    );
}

#[tokio::test]
async fn openai_gets_a_data_url_part_before_the_text_and_plain_turns_stay_strings() {
    let f = fixture(Reply::Json(OPENAI_REPLY)).await;
    let mut req = shown();
    req.messages.push(Message::assistant("Red."));
    req.messages.push(Message::user("Sure?"));
    llm_wires::build(
        Wire::openai(f.endpoint(), "gpt-4o-mini"),
        Some(Secret::from("k")),
    )
    .unwrap()
    .chat(req)
    .await
    .unwrap();
    assert_eq!(
        f.request().json()["messages"],
        json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", b64(RED))}},
                {"type": "text", "text": "What colour is this?"}
            ]},
            // A turn without an image is the string it always was.
            {"role": "assistant", "content": "Red."},
            {"role": "user", "content": "Sure?"}
        ])
    );
}

#[tokio::test]
async fn an_image_off_a_user_turn_is_refused_on_both_chat_wires_alike() {
    for wire in [
        Wire::anthropic("http://127.0.0.1:9", "claude-haiku-4-5"),
        Wire::openai("http://127.0.0.1:9/v1", "gpt-4o-mini"),
    ] {
        let name = wire.name();
        let mut req = ChatRequest::ask("s", "hi");
        req.messages
            .push(Message::assistant("look").with_images([Image::from_bytes("image/png", RED)]));
        match llm_wires::build(wire, Some(Secret::from("k")))
            .unwrap()
            .chat(req)
            .await
        {
            Err(Error::Invalid { wire, what }) => {
                assert_eq!(wire, name);
                assert_eq!(
                    what,
                    "message 1 (assistant) carries images; only a user turn may"
                );
            }
            other => panic!("{name}: expected Invalid, got {other:?}"),
        }
    }
    // The role check is the wire's alone; a user turn with an image is fine.
    assert_eq!(shown().messages[0].role, Role::User);
}

#[tokio::test]
async fn anthropic_refuses_a_data_url_that_is_not_base64() {
    let mut req = ChatRequest::ask("s", "hi");
    req.messages[0] =
        Message::user("hi").with_images([Image::DataUrl("data:image/svg+xml,<svg/>".into())]);
    match llm_wires::build(
        Wire::anthropic("http://127.0.0.1:9", "claude-haiku-4-5"),
        Some(Secret::from("k")),
    )
    .unwrap()
    .chat(req)
    .await
    {
        Err(Error::Invalid { what, .. }) => {
            assert!(
                what.contains("message 0, image 0: a data URL that is not base64"),
                "{what}"
            )
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}
