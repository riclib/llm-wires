//! Between an [`Extraction`] and Clairic's request, and between Cloudflare's
//! envelope and an [`Extracted`].
//!
//! The request side is a spelling of `schema-input.json`, with what the
//! schema would refuse refused here first, by the caller's field name. The
//! response side is where the **200-that-is-not-what-we-asked** rule lives
//! for this wire: a field skipped, answered as another type, or answered with
//! a value of the wrong JSON kind — a number where text was declared — is an
//! [`Error::Decode`] that names the field, never a value coerced into the
//! declared type. A solid step that routes on `answers["amount_paid"]` must
//! not be handed the string `"701.10"` as if the model had said a number.
//!
//! Every number is read from a [`Value`] rather than through a typed serde
//! struct, so a binary with `serde_json`'s `arbitrary_precision` on — solid
//! is one — decodes the same as one without.

use std::collections::BTreeMap;

use serde::{Serialize, Serializer};
use serde_json::{Map, Value};

use crate::extraction::{FieldType, Image, Point, Scale, Step};
use crate::{
    Distribution, Error, Extracted, Extraction, Fact, Field, Item, ListItem, Result, Span, Usage,
    http,
};

use super::WIRE;

// ------------------------------------------------------------------ request

/// `POST …/ai/run/{model}`.
#[derive(Debug, Serialize)]
pub(crate) struct Request<'a> {
    pub(crate) model: &'a str,
    /// Left out when empty, which only an image-only request may be: the
    /// schema's default is `""`, and a body without it is the body their
    /// own examples send for an image.
    #[serde(skip_serializing_if = "str::is_empty")]
    pub(crate) state: &'a str,
    pub(crate) fields: BTreeMap<&'a str, WireField<'a>>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub(crate) images: Vec<WireImage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reference_date: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) options: Option<Options>,
}

/// `options`, sent only when one of them is on.
#[derive(Debug, Serialize)]
pub(crate) struct Options {
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) include_quantiles: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) return_state_text: bool,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum WireImage<'a> {
    DataUrl(&'a str),
    Base64 {
        content_type: &'a str,
        base64: &'a str,
    },
}

/// One field, tagged by `type`. Every option is left out when the caller did
/// not set it, so the body a caller who did not think about it sends is the
/// body the docs' own examples send.
#[derive(Debug, Default, Serialize)]
pub(crate) struct WireField<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) unit: Option<&'a str>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) integer: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) decimals: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) items: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) min: Option<&'a Point>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max: Option<&'a Point>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) step: Option<&'a Step>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) scale: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) null_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) horizon: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) interval: Option<&'static str>,
}

/// A whole number goes out as an integer token: `"min": 0`, as the docs
/// write it, and not `0.0` — the same value, but not the same bytes, and the
/// bytes are what the golden request pins.
fn whole<S: Serializer>(n: f64, s: S) -> std::result::Result<S::Ok, S::Error> {
    const EXACT: f64 = 9_007_199_254_740_992.0; // 2^53
    if n.fract() == 0.0 && n.abs() < EXACT {
        s.serialize_i64(n as i64)
    } else {
        s.serialize_f64(n)
    }
}

impl Serialize for Point {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Point::Number(n) => whole(*n, s),
            Point::Date(d) => s.serialize_str(d),
        }
    }
}

impl Serialize for Step {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Step::By(n) => whole(*n, s),
            Step::Month => s.serialize_str("month"),
            Step::Year => s.serialize_str("year"),
        }
    }
}

fn scale(s: Scale) -> Option<&'static str> {
    match s {
        Scale::Linear => None,
        Scale::Log => Some("log"),
    }
}

pub(crate) fn request<'a>(req: &'a Extraction, model: &'a str) -> Result<Request<'a>> {
    let invalid = |what: String| Error::Invalid { wire: WIRE, what };
    if req.fields.is_empty() {
        return Err(invalid("an extraction needs at least one field".into()));
    }
    if req.state.trim().is_empty() && req.images.is_empty() {
        return Err(invalid("an extraction needs a state or an image".into()));
    }
    if req.images.len() > 4 {
        return Err(invalid(format!(
            "at most four images, got {}",
            req.images.len()
        )));
    }
    let mut images = Vec::with_capacity(req.images.len());
    for (i, image) in req.images.iter().enumerate() {
        images.push(match image {
            Image::DataUrl(url) => {
                // The wire takes no remote URL; one here is a 400 that does
                // not say which image.
                if !url
                    .get(..5)
                    .is_some_and(|s| s.eq_ignore_ascii_case("data:"))
                {
                    return Err(invalid(format!(
                        "image {i} is not a data: URL; the wire fetches nothing"
                    )));
                }
                WireImage::DataUrl(url)
            }
            Image::Base64 {
                content_type,
                base64,
            } => WireImage::Base64 {
                content_type,
                base64,
            },
        });
    }

    let mut fields = BTreeMap::new();
    for (name, field) in &req.fields {
        fields.insert(name.as_str(), wire_field(name, field)?);
    }

    let options = (req.quantiles || req.state_text).then_some(Options {
        include_quantiles: req.quantiles,
        return_state_text: req.state_text,
    });
    Ok(Request {
        model,
        state: &req.state,
        fields,
        images,
        reference_date: req.reference_date.as_deref(),
        options,
    })
}

fn wire_field<'a>(name: &str, field: &'a Field) -> Result<WireField<'a>> {
    let invalid = |what: String| Error::Invalid {
        wire: WIRE,
        what: format!("field {name:?}: {what}"),
    };
    if !valid_name(name) {
        return Err(invalid(
            "a name is a letter or underscore, then up to 63 of letters, digits, `_`, `.` or `-`"
                .into(),
        ));
    }
    if let Some(p) = field.null_threshold
        && !(p > 0.0 && p <= 1.0)
    {
        return Err(invalid(format!("null_threshold {p} is not in (0, 1]")));
    }
    let decimals = |d: Option<u8>| match d {
        Some(d) if d > 6 => Err(invalid(format!("decimals {d} is more than 6"))),
        d => Ok(d),
    };
    let bounds = |min: &Point, max: &Point| match (min, max) {
        (Point::Number(lo), Point::Number(hi)) if lo >= hi => {
            Err(invalid(format!("min {lo} is not below max {hi}")))
        }
        (Point::Number(_), Point::Date(_)) | (Point::Date(_), Point::Number(_)) => Err(invalid(
            "min and max are one a number and one a date".into(),
        )),
        _ => Ok(()),
    };

    let mut w = WireField {
        kind: field.kind(),
        instructions: field.instructions.as_deref(),
        null_threshold: field.null_threshold,
        ..WireField::default()
    };
    match &field.ty {
        FieldType::Text | FieldType::Date | FieldType::Time | FieldType::DateTime => {}
        FieldType::Number {
            integer,
            decimals: d,
            unit,
        } => {
            w.integer = *integer;
            w.decimals = decimals(*d)?;
            w.unit = unit.as_deref();
        }
        FieldType::Duration {
            unit,
            integer,
            decimals: d,
        } => {
            w.integer = *integer;
            w.decimals = decimals(*d)?;
            w.unit = unit.map(|u| u.as_str());
        }
        FieldType::List { items, decimals: d } => {
            w.items = Some(items.kind());
            w.decimals = decimals(*d)?;
        }
        FieldType::Estimate {
            min,
            max,
            step,
            scale: s,
            unit,
        } => {
            bounds(min, max)?;
            if let Some(Step::By(n)) = step
                && !(n.is_finite() && *n > 0.0)
            {
                return Err(invalid(format!("step {n} is not positive")));
            }
            w.min = Some(min);
            w.max = Some(max);
            w.step = step.as_ref();
            w.scale = scale(*s);
            w.unit = unit.as_deref();
        }
        FieldType::Forecast {
            horizon,
            interval,
            min,
            max,
            scale: s,
            unit,
        } => {
            if !(1..=64).contains(horizon) {
                return Err(invalid(format!("horizon {horizon} is not in 1..=64")));
            }
            bounds(min, max)?;
            w.horizon = Some(*horizon);
            w.interval = interval.map(|i| i.as_str());
            w.min = Some(min);
            w.max = Some(max);
            w.scale = scale(*s);
            w.unit = unit.as_deref();
        }
    }
    Ok(w)
}

/// `^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$`, the schema's `propertyNames`.
fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && name.len() <= 64
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

// ----------------------------------------------------------------- response

/// Cloudflare's envelope off, then every declared field read as its type.
///
/// `status` is the 2xx the body came with: a `success: false` inside one is
/// the provider saying no, and it becomes the [`Error::Api`] a non-2xx would
/// have, with the envelope's `errors` as the message and the code.
pub(crate) fn response(text: &str, status: u16, req: &Extraction) -> Result<Extracted> {
    let body: Value = serde_json::from_str(text)
        .map_err(|e| Error::Decode(format!("{e} in {}", http::clip(text))))?;
    if body.get("success").and_then(Value::as_bool) == Some(false) {
        return Err(Error::Api {
            status,
            message: http::api_message(text, status),
            code: http::api_code(text),
            request_id: None,
            retry_after: None,
        });
    }
    let Some(result) = body.get("result").and_then(Value::as_object) else {
        return Err(Error::Decode(format!(
            "no result in the envelope: {}",
            http::clip(text)
        )));
    };
    let Some(answers) = result.get("answers").and_then(Value::as_object) else {
        return Err(Error::Decode(format!(
            "no answers in the result: {}",
            http::clip(text)
        )));
    };

    let mut facts = BTreeMap::new();
    for (name, field) in &req.fields {
        let Some(got) = answers.get(name).and_then(Value::as_object) else {
            return Err(Error::Decode(format!(
                "no answer for field {name:?} in {}",
                http::clip(text)
            )));
        };
        facts.insert(name.clone(), fact(name, field, got)?);
    }
    // An answer to a field nobody asked for is ignored, not an error: the
    // caller reads by name.

    let usage = result.get("usage");
    let tokens = |key: &str| {
        usage
            .and_then(|u| u.get(key))
            .and_then(Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))
    };
    Ok(Extracted {
        model: result
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        answers: facts,
        truncated: result
            .get("truncated")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        state_text: result
            .get("state_text")
            .and_then(Value::as_str)
            .map(str::to_string),
        usage: Usage {
            input: tokens("input_tokens"),
            output: tokens("output_tokens"),
            ..Usage::default()
        },
    })
}

/// One answer, read as the type its field declared and nothing else.
fn fact(name: &str, field: &Field, got: &Map<String, Value>) -> Result<Fact> {
    let kind = field.kind();
    let answered = got.get("type").and_then(Value::as_str).unwrap_or_default();
    if answered != kind {
        return Err(Error::Decode(format!(
            "field {name:?} is a {kind} and was answered as a {answered:?}"
        )));
    }
    let a = Answer { name, kind, got };
    Ok(match &field.ty {
        FieldType::Text => Fact::Text {
            value: a.opt("value", text)?,
            guess: a.opt("guess", text)?,
            span: a.opt("span", span)?,
            p_null: a.req("p_null", number)?,
            confidence: a.req("confidence", number)?,
        },
        FieldType::Number { .. } => Fact::Number {
            value: a.opt("value", number)?,
            guess: a.opt("guess", number)?,
            p_null: a.req("p_null", number)?,
            confidence: a.req("confidence", number)?,
        },
        FieldType::Date => Fact::Date {
            value: a.opt("value", text)?,
            guess: a.opt("guess", text)?,
            p_null: a.req("p_null", number)?,
            confidence: a.req("confidence", number)?,
        },
        FieldType::Time => Fact::Time {
            value: a.opt("value", text)?,
            guess: a.opt("guess", text)?,
            p_null: a.req("p_null", number)?,
            confidence: a.req("confidence", number)?,
        },
        FieldType::DateTime => Fact::DateTime {
            value: a.opt("value", text)?,
            guess: a.opt("guess", text)?,
            p_null: a.req("p_null", number)?,
            confidence: a.req("confidence", number)?,
        },
        FieldType::Duration { .. } => Fact::Duration {
            value: a.opt("value", number)?,
            guess: a.opt("guess", number)?,
            unit: a.opt("unit", text)?,
            p_null: a.req("p_null", number)?,
            confidence: a.req("confidence", number)?,
        },
        FieldType::List { items, .. } => {
            let item = |v: &Value| -> Option<ListItem> {
                match items {
                    Item::Text => v.as_str().map(|s| ListItem::Text(s.into())),
                    Item::Number => number(v).map(ListItem::Number),
                    Item::Date => v.as_str().map(|s| ListItem::Date(s.into())),
                }
            };
            Fact::List {
                // An empty list is "nothing matched"; a `null` is read the
                // same, since a list has no other way to say it.
                value: a.opt("value", |v| array(v, item))?.unwrap_or_default(),
                spans: a.opt("spans", |v| array(v, span))?.unwrap_or_default(),
                confidence: a.opt("confidence", number)?,
            }
        }
        FieldType::Estimate { min, .. } => Fact::Estimate(distribution(&a, min)?),
        FieldType::Forecast { min, horizon, .. } => {
            let steps = a.req("steps", |v| v.as_array().cloned())?;
            // A forecast short of its horizon would bind fewer rows than the
            // caller asked for, with nothing to say which went missing.
            if steps.len() != usize::from(*horizon) {
                return Err(Error::Decode(format!(
                    "forecast answer for {name:?} has {} steps for a horizon of {horizon}",
                    steps.len()
                )));
            }
            let mut out = Vec::with_capacity(steps.len());
            for (i, step) in steps.iter().enumerate() {
                let label = format!("{name} step {i}");
                let Some(got) = step.as_object() else {
                    return Err(Error::Decode(format!(
                        "field {label:?} is not an object: {}",
                        http::clip(&step.to_string())
                    )));
                };
                out.push(distribution(
                    &Answer {
                        name: &label,
                        kind,
                        got,
                    },
                    min,
                )?);
            }
            Fact::Forecast { steps: out }
        }
    })
}

/// An estimate's numbers, or one forecast step's, as points of the kind the
/// field's `min` was.
fn distribution(a: &Answer<'_>, min: &Point) -> Result<Distribution> {
    let p = |v: &Value| point(min, v);
    Ok(Distribution {
        value: a.req("value", p)?,
        median: a.req("median", p)?,
        mean: a.opt("mean", p)?,
        p10: a.req("p10", p)?,
        p90: a.req("p90", p)?,
        p_below: a.req("p_below", number)?,
        p_above: a.req("p_above", number)?,
        quantiles: a.opt("quantiles", |v| array(v, p))?,
    })
}

/// One answer object, and the field it answers, for the errors.
struct Answer<'a> {
    name: &'a str,
    kind: &'static str,
    got: &'a Map<String, Value>,
}

impl Answer<'_> {
    /// A key that must be there and be the right kind.
    fn req<T>(&self, key: &str, read: impl Fn(&Value) -> Option<T>) -> Result<T> {
        match self.got.get(key) {
            None | Some(Value::Null) => Err(Error::Decode(format!(
                "{} answer for {:?} has no {key}",
                self.kind, self.name
            ))),
            Some(v) => read(v).ok_or_else(|| self.mistyped(key, v)),
        }
    }

    /// A key that may be absent or `null`, and is the right kind when not.
    fn opt<T>(&self, key: &str, read: impl Fn(&Value) -> Option<T>) -> Result<Option<T>> {
        match self.got.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => read(v).map(Some).ok_or_else(|| self.mistyped(key, v)),
        }
    }

    fn mistyped(&self, key: &str, v: &Value) -> Error {
        Error::Decode(format!(
            "field {:?} is a {} and its {key} is {}",
            self.name,
            self.kind,
            http::clip(&v.to_string())
        ))
    }
}

fn text(v: &Value) -> Option<String> {
    v.as_str().map(str::to_string)
}

fn number(v: &Value) -> Option<f64> {
    v.as_f64()
}

/// `[start, end]`, two offsets in order.
fn span(v: &Value) -> Option<Span> {
    match v.as_array()?.as_slice() {
        [a, b] => {
            let (a, b) = (a.as_u64()?, b.as_u64()?);
            (a <= b).then_some(a as usize..b as usize)
        }
        _ => None,
    }
}

/// Every element read, or none: one element of the wrong kind is the whole
/// array of the wrong kind.
fn array<T>(v: &Value, read: impl Fn(&Value) -> Option<T>) -> Option<Vec<T>> {
    v.as_array()?.iter().map(read).collect()
}

/// A point of the kind the field's `min` was.
fn point(min: &Point, v: &Value) -> Option<Point> {
    match min {
        Point::Number(_) => number(v).map(Point::Number),
        Point::Date(_) => v.as_str().map(|s| Point::Date(s.into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DurationUnit, Interval};
    use serde_json::json;

    #[test]
    fn a_name_is_the_schemas_pattern() {
        for ok in ["a", "_x", "amount_paid", "a.b-c_9", &"x".repeat(64)] {
            assert!(valid_name(ok), "{ok}");
        }
        for bad in ["", "9a", "-a", "a b", "é", &"x".repeat(65)] {
            assert!(!valid_name(bad), "{bad}");
        }
    }

    #[test]
    fn whole_numbers_go_out_as_integer_tokens() {
        let p = |p: Point| serde_json::to_string(&p).unwrap();
        assert_eq!(p(Point::Number(0.0)), "0");
        assert_eq!(p(Point::Number(5_000_000.0)), "5000000");
        assert_eq!(p(Point::Number(0.25)), "0.25");
        assert_eq!(p(Point::Date("2026-10-01".into())), r#""2026-10-01""#);
        assert_eq!(serde_json::to_string(&Step::Month).unwrap(), r#""month""#);
        assert_eq!(serde_json::to_string(&Step::By(1.0)).unwrap(), "1");
    }

    #[test]
    fn what_the_schema_would_refuse_is_refused_before_the_socket() {
        let one = |f: FieldType| Extraction::of("x").field("f", f);
        let cases: Vec<(Extraction, &str)> = vec![
            (Extraction::of("x"), "at least one field"),
            (
                Extraction::of("  ").field("f", FieldType::Text),
                "a state or an image",
            ),
            (
                Extraction::of("x")
                    .field("f", FieldType::Text)
                    .image(Image::DataUrl("https://example.com/a.png".into())),
                "image 0 is not a data: URL",
            ),
            (
                (0..5).fold(Extraction::of("x").field("f", FieldType::Text), |e, _| {
                    e.image(Image::DataUrl("data:image/png;base64,AA==".into()))
                }),
                "at most four images, got 5",
            ),
            (
                Extraction::of("x").field("9lives", FieldType::Text),
                "field \"9lives\": a name is",
            ),
            (
                one(FieldType::Number {
                    integer: false,
                    decimals: Some(7),
                    unit: None,
                }),
                "decimals 7 is more than 6",
            ),
            (
                one(FieldType::estimate(10, 10)),
                "min 10 is not below max 10",
            ),
            (
                one(FieldType::estimate(0, Point::Date("2026-01-01".into()))),
                "one a number and one a date",
            ),
            (
                one(FieldType::Estimate {
                    min: 0.into(),
                    max: 1.into(),
                    step: Some(Step::By(0.0)),
                    scale: Scale::Linear,
                    unit: None,
                }),
                "step 0 is not positive",
            ),
            (
                one(FieldType::forecast(0, 0, 1)),
                "horizon 0 is not in 1..=64",
            ),
            (
                one(FieldType::forecast(65, 0, 1)),
                "horizon 65 is not in 1..=64",
            ),
            (
                Extraction::of("x").field("f", Field::from(FieldType::Text).null_threshold(0.0)),
                "null_threshold 0 is not in (0, 1]",
            ),
        ];
        for (req, want) in cases {
            match request(&req, "@cf/cloudflare/clairic") {
                Err(Error::Invalid { wire, what }) => {
                    assert_eq!(wire, "workers_ai");
                    assert!(what.contains(want), "{what} / {want}");
                }
                other => panic!("expected Invalid {want}, got {:?}", other.err()),
            }
        }
        // An image alone is a state enough; a data URL's scheme is any case.
        let req = Extraction::of("")
            .field("f", FieldType::Text)
            .image(Image::DataUrl("DATA:image/png;base64,AA==".into()));
        let body = serde_json::to_value(request(&req, "m").unwrap()).unwrap();
        assert_eq!(
            body,
            json!({"model": "m", "fields": {"f": {"type": "text"}}, "images": ["DATA:image/png;base64,AA=="]})
        );
    }

    #[test]
    fn options_go_out_only_when_asked() {
        let mut req = Extraction::of("x").field("f", FieldType::Text);
        let body = serde_json::to_value(request(&req, "m").unwrap()).unwrap();
        assert!(body.get("options").is_none(), "{body}");
        req.quantiles = true;
        let body = serde_json::to_value(request(&req, "m").unwrap()).unwrap();
        assert_eq!(body["options"], json!({"include_quantiles": true}));
        req.state_text = true;
        let body = serde_json::to_value(request(&req, "m").unwrap()).unwrap();
        assert_eq!(
            body["options"],
            json!({"include_quantiles": true, "return_state_text": true})
        );
    }

    #[test]
    fn quantiles_are_kept_when_sent_and_none_when_not() {
        let req = Extraction::of("x").field("d", FieldType::estimate(0, 10));
        let reply = |q: &str| {
            format!(
                r#"{{"result":{{"answers":{{"d":{{"type":"estimate","value":5,"median":5,"p10":2,"p90":8,"p_below":0,"p_above":0{q}}}}}}},"success":true}}"#
            )
        };
        let got = response(&reply(r#","quantiles":[1,5,9]"#), 200, &req).unwrap();
        let Fact::Estimate(d) = &got.answers["d"] else {
            panic!("{got:?}")
        };
        assert_eq!(
            d.quantiles,
            Some(vec![
                Point::Number(1.0),
                Point::Number(5.0),
                Point::Number(9.0)
            ])
        );
        let got = response(&reply(""), 200, &req).unwrap();
        let Fact::Estimate(d) = &got.answers["d"] else {
            panic!("{got:?}")
        };
        assert_eq!(d.quantiles, None);
    }

    #[test]
    fn duration_and_interval_spell_the_schemas_enums() {
        let req = Extraction::of("x")
            .field(
                "d",
                FieldType::Duration {
                    unit: Some(DurationUnit::Days),
                    integer: true,
                    decimals: None,
                },
            )
            .field(
                "f",
                FieldType::Forecast {
                    horizon: 3,
                    interval: Some(Interval::Quarter),
                    min: 0.into(),
                    max: 10.into(),
                    scale: Scale::Log,
                    unit: None,
                },
            );
        let body = serde_json::to_value(request(&req, "m").unwrap()).unwrap();
        assert_eq!(
            body["fields"],
            json!({
                "d": {"type": "duration", "unit": "days", "integer": true},
                "f": {"type": "forecast", "horizon": 3, "interval": "quarter",
                      "min": 0, "max": 10, "scale": "log"}
            })
        );
    }
}
