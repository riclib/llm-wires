//! One error for the crate.

use std::time::Duration;

/// What can go wrong between a `ChatRequest` (or a `Judgement`) and an answer.
///
/// The rule the [`Error::Api`] variant carries: an HTTP status arrives with
/// the **body's** error message and nothing of the request. A request holds
/// the prompt, the tool schemas and, for a wire that ever moved to a query
/// parameter, the key — none of which belongs in a message that ends up on a
/// pane, in a log line, or in a run's failure row.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A wire the enum declares and this crate does not speak yet.
    #[error("the {0} wire is declared but not built yet")]
    Unsupported(&'static str),

    /// [`crate::build`] got `None` where the wire must authenticate. Every
    /// wire must: a local ollama is served by passing the constant `"ollama"`,
    /// which is what the server itself accepts and ignores.
    #[error("the {0} wire needs a key")]
    NoKey(&'static str),

    /// A field of the wire that has no default worth guessing.
    #[error("{wire}: {field} is required")]
    Missing {
        wire: &'static str,
        field: &'static str,
    },

    /// A wire asked for the trait it does not speak: [`crate::build`] on the
    /// TypeSafe wire, which cannot chat, or [`crate::build_judge`] on Anthropic,
    /// Azure or Workers AI, which do not judge. A separate trait was the point — one
    /// interface would have forced a method that always errors — so the
    /// refusal happens once, at `build`, and not on every call.
    #[error("the {wire} wire cannot {verb}")]
    Cannot {
        wire: &'static str,
        verb: &'static str,
    },

    /// A request the wire will not send, because the server would refuse it
    /// and the reason is already known here: no questions, a score with one
    /// level. Refused before the socket, with the caller's own words for the
    /// field, so a solid step sees the mistake without a round trip.
    #[error("{wire}: {what}")]
    Invalid { wire: &'static str, what: String },

    /// The credential body is not text. Every wire we speak puts the key in a
    /// header, and a header value is bytes but a key that is not UTF-8 is a
    /// wrong body, not an exotic one.
    #[error("the {0} key is not text")]
    KeyNotText(&'static str),

    /// The endpoint, joined with the wire's path, is not a URL.
    #[error("endpoint {endpoint}: {why}")]
    Endpoint { endpoint: String, why: String },

    /// One of the card's extra headers is not a header. Refused rather than
    /// dropped: a gateway that needs a header does not work without it, and a
    /// silent drop makes that a 401 nobody can explain.
    #[error("{wire}: {what}")]
    BadHeader { wire: &'static str, what: String },

    /// The transport: connect, TLS, timeout, a body that stopped early.
    ///
    /// The URL is stripped on the way in (`without_url`). reqwest's own
    /// `Display` appends ` for url (…)`, which is the request again — and an
    /// endpoint's path can hold a deployment name, a tenant, or a gateway
    /// route an operator did not mean to put in a log line. The endpoint a
    /// caller is entitled to is [`crate::Info::endpoint`], which it already
    /// has, because it configured it.
    ///
    /// **`transparent`, so the operating system's words reach the operator.**
    /// reqwest's `Display` here is `error sending request` and nothing else;
    /// *connection refused*, *certificate has expired*, *timed out* all live
    /// in the cause it holds, and with no chain at all a person debugging an
    /// unreachable provider is told only that a request was sent. `anyhow`'s
    /// `{:#}` — what `solid`'s `main` prints — then reads
    /// `error sending request: client error (Connect): tcp connect error:
    /// Connection refused (os error 111)`.
    ///
    /// `transparent` and not `#[source]` on a `{0}` message, because those
    /// two together print reqwest's own line twice: the `Display` would
    /// interpolate the very error the chain starts at. Here the `Display` is
    /// the inner one and the chain begins **below** it, so each sentence
    /// appears once.
    ///
    /// It does not put the URL back: `without_url` already took it off the
    /// error this wraps, and the causes underneath are hyper's and the
    /// operating system's, which never carried it.
    #[error(transparent)]
    Http(reqwest::Error),

    /// The provider answered, and it answered no. `message` is the body's own
    /// `error.message` (or TypeSafe's `detail`) — never the request.
    ///
    /// `request_id` is the provider's own id for the exchange, from whichever
    /// header the wire uses (`request-id`, `x-request-id`,
    /// `x-typesafe-request-id`), so a failure row can be quoted to their
    /// support without the request. `retry_after` is a 429's `retry-after-ms`
    /// or `Retry-After`, surfaced and **not acted on**: this crate does not
    /// retry, the run above it does, and this is the number the run wants.
    ///
    /// `code` is the provider's machine-readable name for the refusal, when
    /// the body carries one: a systemone daemon's `STATE_TRUNCATED`, OpenAI's
    /// `context_length_exceeded`, Cloudflare's `10000`. It is what a caller
    /// routes on — send a truncated state to a judge with a longer window —
    /// so that nobody has to match the sentence, which is the provider's to
    /// reword. `None` when the body named no code; never made up here.
    ///
    /// `non_exhaustive`, so that the next thing a refusal carries is a field
    /// and not a breaking release: match it with `..`.
    #[error("the provider answered {status}{}: {message}{}", code_tag(.code), tag(.request_id))]
    #[non_exhaustive]
    Api {
        status: u16,
        message: String,
        code: Option<String>,
        request_id: Option<String>,
        retry_after: Option<Duration>,
    },

    /// The stream itself failed: an `error` frame mid-turn, or a body that
    /// stopped with a tool call half built. Its own variant and not an
    /// [`Error::Api`] with a made-up status, because there was no status —
    /// the head said 200 and the body changed its mind.
    #[error("the provider ended the stream: {message}")]
    Stream { message: String },

    /// A 200 in which the model declined to decide one or more questions:
    /// OpenAI's Decisions API answers each with `{"type": "refusal"}` while
    /// answering the rest. A verdict has no holes, so the whole judgement
    /// fails, naming every refused question by the caller's name.
    #[error("{wire}: the model refused to decide {}", .questions.join(", "))]
    Refused {
        wire: &'static str,
        questions: Vec<String>,
    },

    /// A 2xx, or a stream frame, that is not the shape the wire promises.
    #[error("the provider's reply did not parse: {0}")]
    Decode(String),

    /// The client's TLS could not be built: no roots on the box, or a crypto
    /// provider that refuses the default protocol versions.
    #[error("tls: {0}")]
    Tls(String),
}

/// ` (request …)` on the message when the provider gave an id, nothing when
/// it did not.
fn tag(request_id: &Option<String>) -> String {
    request_id
        .as_deref()
        .map_or_else(String::new, |id| format!(" (request {id})"))
}

/// ` STATE_TRUNCATED` after the status when the provider named a code.
fn code_tag(code: &Option<String>) -> String {
    code.as_deref()
        .map_or_else(String::new, |c| format!(" {c}"))
}

impl Error {
    /// The provider's code for a refusal, when it gave one; `None` for every
    /// other error, and for a refusal that named none.
    pub fn code(&self) -> Option<&str> {
        match self {
            Error::Api { code, .. } => code.as_deref(),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for Error {
    /// Written out rather than derived, for the one call it makes:
    /// `without_url`. Everything else about the error — the kind, the source,
    /// the operating system's own words — survives.
    fn from(e: reqwest::Error) -> Error {
        Error::Http(e.without_url())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
