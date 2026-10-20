//! What a caller hands an [`Extract`](crate::Extract) client and what it gets
//! back.
//!
//! Not a chat, not a judgement and not an embedding. An [`Extraction`] is one
//! `state` (text, and up to four images placed before it) plus a map of typed
//! fields; an [`Extracted`] is one typed [`Fact`] per field, each the declared
//! type or `null`, with the model's own probability that it is `null` and a
//! confidence. The shapes are Cloudflare's Clairic
//! (`@cf/cloudflare/clairic`), read from the model's published JSON schemas
//! (`schema-input.json`, `schema-output.json`, 2026-10) rather than from the
//! launch post, which leaves names out: the percentiles are a request-wide
//! `options.include_quantiles` and not a field's option, a forecast takes an
//! `interval` and not a unit, and the images are data URLs or
//! `{content_type, base64}` objects under `images`.
//!
//! Dates, times and datetimes stay ISO text here, as the wire sends them:
//! this crate links no calendar, and a caller that wants one parses the
//! string it was handed.
//!
//! There is no `model` here, for the reason there is none on
//! [`Judgement`](crate::Judgement): the model is part of the
//! [`Wire`](crate::Wire) a client was built from.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::Usage;

/// One state and the fields to read out of it.
///
/// `fields` is a `BTreeMap` for the reason [`Headers`](crate::Headers) is:
/// sorted, so the same request is the same bytes twice, which is what makes
/// it pinnable. The keys are the caller's and come back as the answers' keys;
/// unlike a judgement's, the model **does** see them, and the docs ask for
/// names that say what the field is.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Extraction {
    /// The text to read. Text answers are spans of it. May be empty only when
    /// there is an image; refused before the socket otherwise.
    pub state: String,
    /// Up to four, placed before the state. The wire takes no remote URL.
    pub images: Vec<Image>,
    pub fields: BTreeMap<String, Field>,
    /// An ISO date that anchors a date written without a year: "arrived on
    /// Sept 28" in a mail the caller knows was sent in 2026.
    pub reference_date: Option<String>,
    /// Ask for 99 percentiles on every estimate (`options.include_quantiles`).
    /// Request-wide on the wire, so request-wide here.
    pub quantiles: bool,
    /// Ask the server to send back the text the spans index into
    /// (`options.return_state_text`), for a state built from images.
    pub state_text: bool,
}

impl Extraction {
    /// An extraction from `state`, with no fields yet. Add them with
    /// [`Extraction::field`].
    pub fn of(state: impl Into<String>) -> Extraction {
        Extraction {
            state: state.into(),
            ..Extraction::default()
        }
    }

    /// One more field, under the name its answer will come back as.
    pub fn field(mut self, name: impl Into<String>, field: impl Into<Field>) -> Extraction {
        self.fields.insert(name.into(), field.into());
        self
    }

    /// One more image, after the ones already added.
    pub fn image(mut self, image: Image) -> Extraction {
        self.images.push(image);
        self
    }

    /// Anchor yearless dates on this ISO date.
    pub fn reference_date(mut self, date: impl Into<String>) -> Extraction {
        self.reference_date = Some(date.into());
        self
    }
}

/// An image, as the wire takes one: embedded, never fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Image {
    /// A whole `data:` URL, `data:image/png;base64,…`.
    DataUrl(String),
    /// The bytes as base64, and their media type.
    Base64 {
        content_type: String,
        base64: String,
    },
}

/// One typed field: what to read, and how the model should read it.
///
/// A struct around [`FieldType`] rather than an `instructions` in each of
/// nine variants, because the two options here are every type's.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub ty: FieldType,
    /// What the field should contain. `None` sends none, which the wire
    /// allows; the name then carries the whole of it.
    pub instructions: Option<String>,
    /// The probability of `null` above which the answer is `null`. `None`
    /// leaves the server's default (0.5). In `(0, 1]`.
    pub null_threshold: Option<f64>,
}

impl Field {
    /// The same field, with instructions.
    pub fn instructions(mut self, text: impl Into<String>) -> Field {
        self.instructions = Some(text.into());
        self
    }

    /// The same field, `null` past `p` rather than the server's 0.5.
    pub fn null_threshold(mut self, p: f64) -> Field {
        self.null_threshold = Some(p);
        self
    }

    /// The wire's name for the type.
    pub fn kind(&self) -> &'static str {
        self.ty.kind()
    }
}

impl From<FieldType> for Field {
    fn from(ty: FieldType) -> Field {
        Field {
            ty,
            instructions: None,
            null_threshold: None,
        }
    }
}

/// The nine types the wire declares, each with its own options.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    /// A span of the state. The answer carries its character offsets.
    Text,
    Number {
        integer: bool,
        /// Digits after the point, `0..=6`.
        decimals: Option<u8>,
        /// A label for the model, `USD`; not converted.
        unit: Option<String>,
    },
    Date,
    Time,
    DateTime,
    Duration {
        /// The unit the answer is in. `None` is the server's seconds.
        unit: Option<DurationUnit>,
        integer: bool,
        decimals: Option<u8>,
    },
    List {
        items: Item,
        /// For a list of numbers only, `0..=6`.
        decimals: Option<u8>,
    },
    /// A value the input does not determine, as a distribution over a grid.
    Estimate {
        min: Point,
        max: Point,
        /// The grid. `None` is the server's.
        step: Option<Step>,
        scale: Scale,
        unit: Option<String>,
    },
    /// The next `horizon` values of a series, one estimate per step.
    Forecast {
        /// Steps ahead, `1..=64`.
        horizon: u8,
        interval: Option<Interval>,
        min: Point,
        max: Point,
        scale: Scale,
        unit: Option<String>,
    },
}

impl FieldType {
    /// The wire's name for the type.
    pub fn kind(&self) -> &'static str {
        match self {
            FieldType::Text => "text",
            FieldType::Number { .. } => "number",
            FieldType::Date => "date",
            FieldType::Time => "time",
            FieldType::DateTime => "datetime",
            FieldType::Duration { .. } => "duration",
            FieldType::List { .. } => "list",
            FieldType::Estimate { .. } => "estimate",
            FieldType::Forecast { .. } => "forecast",
        }
    }

    /// A number, with neither integer nor decimals nor unit asked for.
    pub fn number() -> FieldType {
        FieldType::Number {
            integer: false,
            decimals: None,
            unit: None,
        }
    }

    /// A list of `items`.
    pub fn list(items: Item) -> FieldType {
        FieldType::List {
            items,
            decimals: None,
        }
    }

    /// An estimate between `min` and `max`, on the server's grid.
    pub fn estimate(min: impl Into<Point>, max: impl Into<Point>) -> FieldType {
        FieldType::Estimate {
            min: min.into(),
            max: max.into(),
            step: None,
            scale: Scale::Linear,
            unit: None,
        }
    }

    /// A forecast `horizon` steps ahead, between `min` and `max`.
    pub fn forecast(horizon: u8, min: impl Into<Point>, max: impl Into<Point>) -> FieldType {
        FieldType::Forecast {
            horizon,
            interval: None,
            min: min.into(),
            max: max.into(),
            scale: Scale::Linear,
            unit: None,
        }
    }
}

/// What a list holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Item {
    Text,
    Number,
    Date,
}

impl Item {
    pub fn kind(self) -> &'static str {
        match self {
            Item::Text => "text",
            Item::Number => "number",
            Item::Date => "date",
        }
    }
}

/// A duration's unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

impl DurationUnit {
    pub fn as_str(self) -> &'static str {
        match self {
            DurationUnit::Seconds => "seconds",
            DurationUnit::Minutes => "minutes",
            DurationUnit::Hours => "hours",
            DurationUnit::Days => "days",
            DurationUnit::Weeks => "weeks",
            DurationUnit::Months => "months",
            DurationUnit::Years => "years",
        }
    }
}

/// A point on an estimate's axis: a number, or an ISO date for an estimate
/// of a date. A bound of the request and every point of the answer are one
/// or the other, and the answer's are the kind the field's `min` was.
#[derive(Debug, Clone, PartialEq)]
pub enum Point {
    Number(f64),
    Date(String),
}

impl Point {
    /// The number, when the field was estimated in numbers.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Point::Number(n) => Some(*n),
            Point::Date(_) => None,
        }
    }
}

impl From<f64> for Point {
    fn from(n: f64) -> Point {
        Point::Number(n)
    }
}

impl From<i32> for Point {
    fn from(n: i32) -> Point {
        Point::Number(n.into())
    }
}

/// The estimate's grid: a positive number (days, for a date), or calendar
/// months or years.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    By(f64),
    Month,
    Year,
}

/// How the grid is spaced. Linear is the server's default and is left out of
/// the body.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Scale {
    #[default]
    Linear,
    Log,
}

/// A forecast's step. `Step` is "whatever the series' rows are".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    Step,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

impl Interval {
    pub fn as_str(self) -> &'static str {
        match self {
            Interval::Step => "step",
            Interval::Minute => "minute",
            Interval::Hour => "hour",
            Interval::Day => "day",
            Interval::Week => "week",
            Interval::Month => "month",
            Interval::Quarter => "quarter",
            Interval::Year => "year",
        }
    }
}

/// Character offsets into the state, as the wire counts them.
pub type Span = Range<usize>;

/// One typed answer, of the type its field was.
///
/// A scalar's `value` is `None` when the model found the input does not
/// determine it; `guess` is what it would have said, kept beside it so a
/// caller can show a reviewer the near miss. `p_null` is the model's
/// probability that the answer is `null`, `confidence` its probability that
/// the value (or the guess) is right. What a caller does with either is the
/// caller's threshold and not this crate's.
#[derive(Debug, Clone, PartialEq)]
pub enum Fact {
    Text {
        value: Option<String>,
        guess: Option<String>,
        /// Where `guess` sits in the state. Present for a `null` too, as the
        /// guess's span.
        span: Option<Span>,
        p_null: f64,
        confidence: f64,
    },
    Number {
        value: Option<f64>,
        guess: Option<f64>,
        p_null: f64,
        confidence: f64,
    },
    /// An ISO date, time or datetime, as the field asked.
    Date {
        value: Option<String>,
        guess: Option<String>,
        p_null: f64,
        confidence: f64,
    },
    Time {
        value: Option<String>,
        guess: Option<String>,
        p_null: f64,
        confidence: f64,
    },
    DateTime {
        value: Option<String>,
        guess: Option<String>,
        p_null: f64,
        confidence: f64,
    },
    Duration {
        value: Option<f64>,
        guess: Option<f64>,
        /// The unit the server says it answered in.
        unit: Option<String>,
        p_null: f64,
        confidence: f64,
    },
    /// Empty when nothing matched; a list is never `null`.
    List {
        value: Vec<ListItem>,
        /// One per item, for a list of text; empty otherwise.
        spans: Vec<Span>,
        /// Not always sent.
        confidence: Option<f64>,
    },
    Estimate(Distribution),
    /// One distribution per step, in order: a row per step is how a caller
    /// binds a forecast. Read from the reply's `steps`, which carries every
    /// step whole; the parallel `value`/`median`/`p10`/`p90` arrays the
    /// server also sends are those same numbers, and are not read.
    Forecast {
        steps: Vec<Distribution>,
    },
}

/// What the model believes about a value the input does not determine: an
/// estimate's one, and each step of a forecast.
///
/// Every point is the kind the field's `min` was, a number or a date.
#[derive(Debug, Clone, PartialEq)]
pub struct Distribution {
    /// The point the model would pick.
    pub value: Point,
    pub median: Point,
    /// Not always sent.
    pub mean: Option<Point>,
    /// The 80% interval.
    pub p10: Point,
    pub p90: Point,
    /// The mass the model put below the field's `min` and above its `max`.
    pub p_below: f64,
    pub p_above: f64,
    /// 99 percentiles, when [`Extraction::quantiles`] asked for them; `None`
    /// when it did not. Points and not `f64`, because an estimate of a date
    /// has dates for percentiles.
    pub quantiles: Option<Vec<Point>>,
}

impl Fact {
    /// The wire's name for the type.
    pub fn kind(&self) -> &'static str {
        match self {
            Fact::Text { .. } => "text",
            Fact::Number { .. } => "number",
            Fact::Date { .. } => "date",
            Fact::Time { .. } => "time",
            Fact::DateTime { .. } => "datetime",
            Fact::Duration { .. } => "duration",
            Fact::List { .. } => "list",
            Fact::Estimate { .. } => "estimate",
            Fact::Forecast { .. } => "forecast",
        }
    }

    /// Whether the model found the input does not determine it. Never true
    /// of a list, an estimate or a forecast.
    pub fn is_null(&self) -> bool {
        match self {
            Fact::Text { value, .. }
            | Fact::Date { value, .. }
            | Fact::Time { value, .. }
            | Fact::DateTime { value, .. } => value.is_none(),
            Fact::Number { value, .. } | Fact::Duration { value, .. } => value.is_none(),
            Fact::List { .. } | Fact::Estimate { .. } | Fact::Forecast { .. } => false,
        }
    }
}

/// One item of a list, of the type the list declared.
#[derive(Debug, Clone, PartialEq)]
pub enum ListItem {
    Text(String),
    Number(f64),
    Date(String),
}

/// The answers, one per field asked, under the caller's names.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    /// The model that answered, as the provider names it.
    pub model: String,
    pub answers: BTreeMap<String, Fact>,
    /// The state was cut to fit; answers past the cut were read from less
    /// than the caller sent.
    pub truncated: bool,
    /// The text the spans index into, when [`Extraction::state_text`] asked.
    pub state_text: Option<String>,
    /// `input` only: the wire bills input and reports zero output.
    pub usage: Usage,
}
