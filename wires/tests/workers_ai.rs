//! The Workers AI wire, speaking Clairic, against a real socket.
//!
//! The same listener the other wires' pins use (`llm_wires::testing`), for
//! the same reason: what these are about is the **bytes**. The request is
//! pinned whole, as a golden string; the responses are the docs' own
//! examples — the model page's invoice, the launch post's email, receipt,
//! estimate and forecast — inside the `{result, success, errors, messages}`
//! envelope the REST route wraps every answer in, until a recorded live
//! answer replaces them.

use std::collections::BTreeMap;

use llm_wires::testing::{Reply, fixture};
use llm_wires::{
    DurationUnit, Error, Extraction, Fact, Field, FieldType, Image, Interval, Item, ListItem,
    Point, Scale, Step, Usage, Wire,
};
use wire_secret::Secret;

const MODEL: &str = "@cf/cloudflare/clairic";

/// The launch post's email, verbatim.
const EMAIL: &str = "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a \
                     dented corner. I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and \
                     I'd like this fixed or refunded. Thanks, Dana";

fn email() -> Extraction {
    Extraction::of(EMAIL)
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
            Field::from(FieldType::list(Item::Text)).instructions("Each kind of damage described"),
        )
        .field("tracking_number", FieldType::Text)
}

fn extractor(endpoint: String) -> Box<dyn llm_wires::Extract> {
    llm_wires::build_extract(
        Wire::workers_ai(endpoint, "acc123", MODEL),
        Some(Secret::from("cf-test-token")),
    )
    .unwrap()
}

#[tokio::test]
async fn the_request_is_one_golden_body_with_every_type_and_every_option() {
    let f = fixture(Reply::Json(INVOICE)).await;
    let mut headers = BTreeMap::new();
    headers.insert("cf-aig-authorization".to_string(), "Bearer gw".to_string());
    let extract = llm_wires::build_extract(
        Wire::WorkersAi {
            endpoint: f.endpoint_at("/client/v4/"),
            account: "acc123".into(),
            model: MODEL.into(),
            headers,
        },
        Some(Secret::from("cf-test-token")),
    )
    .unwrap();

    // One field of every type, every option set at least once, inserted out
    // of order: the body is sorted by name whatever order the caller used.
    let mut req =
        Extraction::of("Invoice INV-2291 from Northwind Supply, issued 2026-09-02, net 30.")
            .field(
                "travelers",
                FieldType::Forecast {
                    horizon: 2,
                    interval: Some(Interval::Day),
                    min: 500_000.into(),
                    max: 4_000_000.into(),
                    scale: Scale::Log,
                    unit: None,
                },
            )
            .field(
                "total",
                Field::from(FieldType::Number {
                    integer: false,
                    decimals: Some(2),
                    unit: Some("USD".into()),
                })
                .instructions("Amount due including shipping")
                .null_threshold(0.25),
            )
            .field("invoice_number", FieldType::Text)
            .field("issued", FieldType::Date)
            .field("cutoff", FieldType::Time)
            .field("sent_at", FieldType::DateTime)
            .field(
                "terms",
                FieldType::Duration {
                    unit: Some(DurationUnit::Days),
                    integer: true,
                    decimals: None,
                },
            )
            .field("contacts", FieldType::list(Item::Text))
            .field(
                "days_to_pay",
                FieldType::Estimate {
                    min: 0.into(),
                    max: 120.into(),
                    step: Some(Step::By(1.0)),
                    scale: Scale::Linear,
                    unit: Some("days".into()),
                },
            )
            .field(
                "paid_by",
                FieldType::Estimate {
                    min: Point::Date("2026-09-02".into()),
                    max: Point::Date("2027-09-02".into()),
                    step: Some(Step::Month),
                    scale: Scale::Linear,
                    unit: None,
                },
            )
            .image(Image::Base64 {
                content_type: "image/png".into(),
                base64: "iVBORw0KGgo=".into(),
            })
            .image(Image::DataUrl("data:image/webp;base64,UklGRg==".into()))
            .reference_date("2026-10-20");
    req.quantiles = true;
    req.state_text = true;

    // The INVOICE reply does not answer all of these; the request is what
    // this pin is about, so the error after it is expected.
    let err = extract.extract(req).await.unwrap_err();
    assert!(matches!(err, Error::Decode(_)), "{err}");

    let seen = f.request();
    assert_eq!(seen.method, "POST");
    assert_eq!(
        seen.path,
        "/client/v4/accounts/acc123/ai/run/@cf/cloudflare/clairic"
    );
    assert_eq!(seen.query, "");
    assert_eq!(seen.header("authorization"), Some("Bearer cf-test-token"));
    assert_eq!(seen.header("cf-aig-authorization"), Some("Bearer gw"));
    assert_eq!(seen.header("accept"), Some("application/json"));
    assert_eq!(seen.header("content-type"), Some("application/json"));
    let ident = format!("llm-wires/{}", env!("CARGO_PKG_VERSION"));
    assert_eq!(seen.header("user-agent"), Some(ident.as_str()));
    assert_eq!(
        seen.body,
        concat!(
            r#"{"model":"@cf/cloudflare/clairic","#,
            r#""state":"Invoice INV-2291 from Northwind Supply, issued 2026-09-02, net 30.","#,
            r#""fields":{"#,
            r#""contacts":{"type":"list","items":"text"},"#,
            r#""cutoff":{"type":"time"},"#,
            r#""days_to_pay":{"type":"estimate","unit":"days","min":0,"max":120,"step":1},"#,
            r#""invoice_number":{"type":"text"},"#,
            r#""issued":{"type":"date"},"#,
            r#""paid_by":{"type":"estimate","min":"2026-09-02","max":"2027-09-02","step":"month"},"#,
            r#""sent_at":{"type":"datetime"},"#,
            r#""terms":{"type":"duration","unit":"days","integer":true},"#,
            r#""total":{"type":"number","instructions":"Amount due including shipping","unit":"USD","decimals":2,"null_threshold":0.25},"#,
            r#""travelers":{"type":"forecast","min":500000,"max":4000000,"scale":"log","horizon":2,"interval":"day"}"#,
            r#"},"#,
            r#""images":[{"content_type":"image/png","base64":"iVBORw0KGgo="},"data:image/webp;base64,UklGRg=="],"#,
            r#""reference_date":"2026-10-20","#,
            r#""options":{"include_quantiles":true,"return_state_text":true}}"#,
        )
    );
}

#[tokio::test]
async fn the_posts_email_answers_701_10_and_no_tracking_number() {
    let f = fixture(Reply::Json(EMAIL_ANSWERS)).await;
    let got = extractor(f.endpoint_at("/client/v4"))
        .extract(email())
        .await
        .unwrap();

    assert_eq!(got.model, "clairic");
    assert!(!got.truncated);
    assert_eq!(got.state_text, None);
    assert_eq!(
        got.usage,
        Usage {
            input: 412,
            ..Usage::default()
        }
    );
    // A span is offsets into the state: the caller can cut the source with it.
    let Fact::Text { span, .. } = &got.answers["order_id"] else {
        panic!("{:?}", got.answers["order_id"]);
    };
    assert_eq!(&EMAIL[span.clone().unwrap()], "A-88412");
    assert_eq!(
        got.answers["order_id"],
        Fact::Text {
            value: Some("A-88412".into()),
            guess: Some("A-88412".into()),
            span: Some(10..17),
            p_null: 3.4e-6,
            confidence: 0.999957,
        }
    );
    assert_eq!(
        got.answers["delivered_on"],
        Fact::Date {
            value: Some("2026-09-28".into()),
            guess: Some("2026-09-28".into()),
            p_null: 6.5e-8,
            confidence: 0.999999,
        }
    );
    assert_eq!(
        got.answers["amount_paid"],
        Fact::Number {
            value: Some(701.1),
            guess: Some(701.1),
            p_null: 1.6e-5,
            confidence: 0.999944,
        }
    );
    assert_eq!(
        got.answers["days_since_delivery"],
        Fact::Duration {
            value: Some(11.0),
            guess: Some(11.0),
            unit: Some("days".into()),
            p_null: 6.1e-9,
            confidence: 0.999999,
        }
    );
    // The post's list carries no confidence; that is not an error.
    assert_eq!(
        got.answers["damage"],
        Fact::List {
            value: vec![
                ListItem::Text("cracked screen".into()),
                ListItem::Text("dented corner".into()),
            ],
            spans: vec![50..64, 71..84],
            confidence: None,
        }
    );
    // A null keeps its guess, and the guess keeps its span.
    let tracking = &got.answers["tracking_number"];
    assert!(tracking.is_null());
    assert_eq!(
        *tracking,
        Fact::Text {
            value: None,
            guess: Some("$52.10".into()),
            span: Some(106..112),
            p_null: 1.0,
            confidence: 0.416016,
        }
    );
    assert_eq!(&EMAIL[106..112], "$52.10");
}

#[tokio::test]
async fn the_model_pages_invoice_decodes_an_estimate_and_a_list_with_confidence() {
    let f = fixture(Reply::Json(INVOICE)).await;
    let req = Extraction::of("Invoice INV-2291 from Northwind Supply, issued 2026-09-02, net 30. 4 pallets at $250.00 each, shipping $80.00. Contact: Dana Ruiz, Priya Shah.")
        .field("invoice_number", FieldType::Text)
        .field("due_date", FieldType::Date)
        .field("total", FieldType::number())
        .field("contacts", FieldType::list(Item::Text))
        .field("po_number", FieldType::Text)
        .field("days_to_pay", FieldType::estimate(0, 120));
    let got = extractor(f.endpoint_at("/client/v4"))
        .extract(req)
        .await
        .unwrap();

    assert_eq!(
        got.answers["days_to_pay"],
        Fact::Estimate {
            value: Point::Number(30.0),
            median: Point::Number(30.0),
            mean: Some(Point::Number(29.535)),
            p10: Point::Number(21.0),
            p90: Point::Number(38.0),
            p_below: 5.9e-06,
            p_above: 1.2e-05,
            quantiles: vec![],
        }
    );
    assert_eq!(
        got.answers["contacts"],
        Fact::List {
            value: vec![
                ListItem::Text("Dana Ruiz".into()),
                ListItem::Text("Priya Shah".into())
            ],
            spans: vec![120..129, 131..141],
            confidence: Some(0.9999),
        }
    );
    assert!(got.answers["po_number"].is_null());
    assert_eq!(
        got.usage,
        Usage {
            input: 655,
            ..Usage::default()
        }
    );
}

#[tokio::test]
async fn a_receipt_decodes_datetimes_and_whole_numbers() {
    let f = fixture(Reply::Json(RECEIPT)).await;
    let req = Extraction::of("")
        .image(Image::DataUrl("data:image/jpeg;base64,/9j/4AAQ".into()))
        .field("order_number", FieldType::number())
        .field("purchased_at", FieldType::DateTime)
        .field("discount", FieldType::number());
    let got = extractor(f.endpoint_at("/client/v4"))
        .extract(req)
        .await
        .unwrap();
    assert_eq!(
        got.answers["purchased_at"],
        Fact::DateTime {
            value: Some("2026-07-01T13:34:00".into()),
            guess: Some("2026-07-01T13:34:00".into()),
            p_null: 4.136094133855295e-08,
            confidence: 0.831464227002992,
        }
    );
    assert_eq!(
        got.answers["order_number"],
        Fact::Number {
            value: Some(77245.0),
            guess: Some(77245.0),
            p_null: 0.0040206750854849815,
            confidence: 0.776249383996532,
        }
    );
    // A null number keeps its guess, which is a number.
    assert_eq!(
        got.answers["discount"],
        Fact::Number {
            value: None,
            guess: Some(0.0),
            p_null: 0.9994148015975952,
            confidence: 0.91852733929319,
        }
    );
    // The receipt's own fields nobody asked for are not answers here.
    assert_eq!(got.answers.len(), 3);
}

#[tokio::test]
async fn a_forecast_is_one_entry_per_step_in_each_array() {
    let f = fixture(Reply::Json(FORECAST)).await;
    let req = Extraction::of("Daily number of people screened…")
        .field("travelers", FieldType::forecast(1, 500_000, 4_000_000));
    let got = extractor(f.endpoint_at("/client/v4"))
        .extract(req)
        .await
        .unwrap();
    assert_eq!(
        got.answers["travelers"],
        Fact::Forecast {
            value: vec![Point::Number(2416117.0)],
            median: vec![Point::Number(2419283.0)],
            p10: vec![Point::Number(2270774.0)],
            p90: vec![Point::Number(2556769.0)],
        }
    );
}

#[tokio::test]
async fn a_mistyped_answer_is_a_decode_error_that_names_the_field() {
    // Each is a 200 the wire must refuse rather than coerce.
    let cases: [(&str, Extraction, &str); 5] = [
        // A number where text was declared.
        (
            r#"{"result":{"model":"clairic","answers":{"order_id":{"type":"text","value":88412,"p_null":0,"confidence":1}},"truncated":false,"usage":{"input_tokens":1,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#,
            Extraction::of("x").field("order_id", FieldType::Text),
            r#"field "order_id" is a text and its value is 88412"#,
        ),
        // Text where a number was declared — the string a caller would
        // otherwise be tempted to parse.
        (
            r#"{"result":{"model":"clairic","answers":{"amount_paid":{"type":"number","value":"701.10","p_null":0,"confidence":1}},"truncated":false,"usage":{"input_tokens":1,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#,
            Extraction::of("x").field("amount_paid", FieldType::number()),
            r#"field "amount_paid" is a number and its value is "701.10""#,
        ),
        // Answered as another type.
        (
            r#"{"result":{"model":"clairic","answers":{"amount_paid":{"type":"text","value":"701.10","p_null":0,"confidence":1}},"truncated":false,"usage":{"input_tokens":1,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#,
            Extraction::of("x").field("amount_paid", FieldType::number()),
            r#"field "amount_paid" is a number and was answered as a "text""#,
        ),
        // A field skipped.
        (
            r#"{"result":{"model":"clairic","answers":{},"truncated":false,"usage":{"input_tokens":1,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#,
            Extraction::of("x").field("amount_paid", FieldType::number()),
            r#"no answer for field "amount_paid""#,
        ),
        // A list with one item of the wrong kind is the wrong kind.
        (
            r#"{"result":{"model":"clairic","answers":{"damage":{"type":"list","value":["cracked",2]}},"truncated":false,"usage":{"input_tokens":1,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#,
            Extraction::of("x").field("damage", FieldType::list(Item::Text)),
            r#"field "damage" is a list and its value is ["cracked",2]"#,
        ),
    ];
    for (body, req, want) in cases {
        let body: &'static str = Box::leak(body.to_string().into_boxed_str());
        let f = fixture(Reply::Json(body)).await;
        match extractor(f.endpoint_at("/client/v4")).extract(req).await {
            Err(Error::Decode(why)) => assert!(why.contains(want), "{why}\nwanted {want}"),
            other => panic!("expected Decode {want:?}, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn the_bare_binding_shape_is_not_the_rest_envelope() {
    // What `env.AI.run` hands a Worker has no envelope; the REST route
    // always does, so a body without one is a gateway answering something
    // else and is said to be.
    let f = fixture(Reply::Json(
        r#"{"model":"clairic","answers":{},"truncated":false,"usage":{"input_tokens":1,"output_tokens":0}}"#,
    ))
    .await;
    match extractor(f.endpoint_at("/client/v4"))
        .extract(Extraction::of("x").field("f", FieldType::Text))
        .await
    {
        Err(Error::Decode(why)) => assert!(why.contains("no result in the envelope"), "{why}"),
        other => panic!("expected Decode, got {other:?}"),
    }
}

#[tokio::test]
async fn cloudflares_no_carries_its_errors_and_the_ray() {
    let f = fixture(Reply::Headed(
        401,
        &[("cf-ray", "8f1d2c3b4a5e6f70-LIS")],
        r#"{"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}],"messages":[]}"#,
    ))
    .await;
    match extractor(f.endpoint_at("/client/v4"))
        .extract(email())
        .await
    {
        Err(Error::Api {
            status,
            message,
            request_id,
            ..
        }) => {
            assert_eq!(status, 401);
            assert_eq!(message, "Authentication error (code 10000)");
            assert_eq!(request_id.as_deref(), Some("8f1d2c3b4a5e6f70-LIS"));
        }
        other => panic!("expected Api, got {other:?}"),
    }

    // And a 200 whose envelope says no is the same no.
    let f = fixture(Reply::Json(
        r#"{"result":null,"success":false,"errors":[{"code":5007,"message":"No such model"}],"messages":[]}"#,
    ))
    .await;
    match extractor(f.endpoint_at("/client/v4"))
        .extract(email())
        .await
    {
        Err(Error::Api {
            status, message, ..
        }) => {
            assert_eq!(status, 200);
            assert_eq!(message, "No such model (code 5007)");
        }
        other => panic!("expected Api, got {other:?}"),
    }
}

/// The launch post's answers to [`email`], in the REST envelope. `usage` is
/// not in the post; the numbers are made up and only the shape is the docs'.
const EMAIL_ANSWERS: &str = r#"{
  "result": {
    "model": "clairic",
    "answers": {
      "order_id": {"type": "text", "p_null": 3.4e-6, "span": [10, 17], "guess": "A-88412", "confidence": 0.999957, "value": "A-88412"},
      "delivered_on": {"type": "date", "p_null": 6.5e-8, "guess": "2026-09-28", "confidence": 0.999999, "value": "2026-09-28"},
      "amount_paid": {"type": "number", "p_null": 1.6e-5, "guess": 701.1, "confidence": 0.999944, "value": 701.1},
      "days_since_delivery": {"type": "duration", "p_null": 6.1e-9, "unit": "days", "guess": 11, "confidence": 0.999999, "value": 11},
      "damage": {"type": "list", "value": ["cracked screen", "dented corner"], "spans": [[50, 64], [71, 84]]},
      "tracking_number": {"type": "text", "p_null": 1, "span": [106, 112], "guess": "$52.10", "confidence": 0.416016, "value": null}
    },
    "truncated": false,
    "usage": {"state_tokens": 52, "image_tokens": 0, "schema_tokens": 360, "input_tokens": 412, "output_tokens": 0, "passes": 2}
  },
  "success": true,
  "errors": [],
  "messages": []
}"#;

/// The model page's example response, in the REST envelope.
const INVOICE: &str = r#"{
  "result": {
    "model": "clairic",
    "answers": {
      "invoice_number": {"type": "text", "p_null": 3.8e-10, "span": [8, 16], "guess": "INV-2291", "confidence": 1, "value": "INV-2291"},
      "due_date": {"type": "date", "p_null": 1.9e-08, "guess": "2026-10-02", "confidence": 0.9999, "value": "2026-10-02"},
      "total": {"type": "number", "p_null": 2.3e-08, "guess": 1080, "confidence": 0.9999, "value": 1080},
      "contacts": {"type": "list", "value": ["Dana Ruiz", "Priya Shah"], "spans": [[120, 129], [131, 141]], "confidence": 0.9999},
      "po_number": {"type": "text", "p_null": 1, "span": [8, 16], "guess": "INV-2291", "confidence": 0.9995, "value": null},
      "days_to_pay": {"type": "estimate", "value": 30, "mean": 29.535, "median": 30, "p10": 21, "p90": 38, "p_below": 5.9e-06, "p_above": 1.2e-05}
    },
    "truncated": false,
    "usage": {"state_tokens": 61, "image_tokens": 0, "schema_tokens": 472, "input_tokens": 655, "output_tokens": 0, "passes": 3}
  },
  "success": true,
  "errors": [],
  "messages": []
}"#;

/// Three of the launch post's receipt answers, in the REST envelope.
const RECEIPT: &str = r#"{
  "result": {
    "model": "clairic",
    "answers": {
      "order_number": {"type": "number", "p_null": 0.0040206750854849815, "guess": 77245, "confidence": 0.776249383996532, "value": 77245},
      "purchased_at": {"type": "datetime", "p_null": 4.136094133855295e-08, "guess": "2026-07-01T13:34:00", "confidence": 0.831464227002992, "value": "2026-07-01T13:34:00"},
      "discount": {"type": "number", "p_null": 0.9994148015975952, "guess": 0, "confidence": 0.91852733929319, "value": null},
      "tip": {"type": "number", "p_null": 5.953857908025384e-05, "guess": 2.39, "confidence": 0.9335768781214697, "value": 2.39}
    },
    "truncated": false,
    "usage": {"input_tokens": 900, "output_tokens": 0}
  },
  "success": true,
  "errors": [],
  "messages": []
}"#;

/// The launch post's TSA forecast, in the REST envelope, with the `steps`
/// array the live server sends beside the parallel arrays (recorded
/// 2026-10-20, where the post shows only the arrays and the schema only
/// names `steps`). The arrays are what the wire reads; `steps` is here so
/// that a reply carrying both is pinned to decode.
const FORECAST: &str = r#"{
  "result": {
    "model": "clairic",
    "answers": {
      "travelers": {"type": "forecast", "value": [2416117], "median": [2419283], "p10": [2270774], "p90": [2556769],
        "steps": [{"value": 2416117, "mean": 2416117, "median": 2419283, "p10": 2270774, "p90": 2556769, "p_below": 2.6e-05, "p_above": 6.5e-06}]}
    },
    "truncated": false,
    "usage": {"input_tokens": 3000, "output_tokens": 0}
  },
  "success": true,
  "errors": [],
  "messages": []
}"#;
