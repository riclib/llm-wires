//! What a caller hands a [`Judge`](crate::Judge) and what it gets back.
//!
//! Not a chat. There is no message list, no tool and no generated text: a
//! [`Judgement`] is one `state` plus a map of typed questions, and a
//! [`Verdict`] is one typed [`Answer`] per question, each a distribution the
//! model calibrated rather than a string we parsed. The shapes are the
//! TypeSafe System One API's, read from the published SDK's declarations
//! (`@typesafe-ai/sdk` 0.6.0, `index.d.mts`), with the leaves left as
//! [`serde_json::Value`] because the wire takes text, an object, an array or
//! `null` at every one of them and a `String` would have been the narrower
//! type for no gain.
//!
//! There is no `model` here, for the reason there is none on
//! [`ChatRequest`](crate::ChatRequest): the model is part of the
//! [`Wire`](crate::Wire) a client was built from, so one client is one
//! deployment and a caller cannot address a model the operator did not
//! configure.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::Usage;

/// One state and the questions to ask of it.
///
/// `questions` is a `BTreeMap` for the reason [`Headers`](crate::Headers)
/// is: sorted, so the same request is the same bytes twice, which is what
/// makes it pinnable. The keys are the caller's; the answers come back under
/// them and the model never sees them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Judgement {
    /// What to judge: text, or a JSON object or array for a chat log, a
    /// record, the current state of an application. `Value::Null` is sent as
    /// `null`, which the wire accepts.
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
}

impl Judgement {
    /// A judgement of `state`, with no questions yet. Add them with
    /// [`Judgement::ask`].
    pub fn of(state: impl Into<Value>) -> Judgement {
        Judgement {
            state: state.into(),
            questions: BTreeMap::new(),
        }
    }

    /// One more question, under the name its answer will come back as.
    pub fn ask(mut self, name: impl Into<String>, question: Question) -> Judgement {
        self.questions.insert(name.into(), question);
        self
    }
}

/// One typed question. The three kinds are the wire's, and each answer type
/// matches its question's.
///
/// `instructions` is what the model is asked, as text, an object or an
/// array; `Value::Null` sends no instructions, which the wire allows. The
/// criteria are the kind's own: descriptions of yes and no, a labelled set of
/// options, or an ordered ladder of levels.
#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    /// A yes/no question, answered as the probability of yes.
    Noul {
        instructions: Value,
        /// What a yes means. `Value::Null` leaves it undescribed.
        yes: Value,
        /// What a no means. `Value::Null` leaves it undescribed.
        no: Value,
    },
    /// One option from a set the caller names, answered with the chosen
    /// label and the whole distribution.
    Choice {
        instructions: Value,
        /// Label to description. `Value::Null` leaves that label
        /// undescribed; the label itself still goes out.
        options: BTreeMap<String, Value>,
    },
    /// A rating along an ordered rubric, answered as a probability-weighted
    /// score across the levels.
    Score {
        instructions: Value,
        /// The levels, in order, indexed from zero. The wire requires at
        /// least two, and [`build_judge`](crate::build_judge)'s client refuses
        /// fewer before the socket.
        levels: Vec<Value>,
    },
}

impl Question {
    /// A yes/no question with neither outcome described.
    pub fn noul(instructions: impl Into<Value>) -> Question {
        Question::Noul {
            instructions: instructions.into(),
            yes: Value::Null,
            no: Value::Null,
        }
    }

    /// A yes/no question with both outcomes described.
    pub fn noul_described(
        instructions: impl Into<Value>,
        yes: impl Into<Value>,
        no: impl Into<Value>,
    ) -> Question {
        Question::Noul {
            instructions: instructions.into(),
            yes: yes.into(),
            no: no.into(),
        }
    }

    /// A choice among labelled options.
    pub fn choice<L, D>(
        instructions: impl Into<Value>,
        options: impl IntoIterator<Item = (L, D)>,
    ) -> Question
    where
        L: Into<String>,
        D: Into<Value>,
    {
        Question::Choice {
            instructions: instructions.into(),
            options: options
                .into_iter()
                .map(|(l, d)| (l.into(), d.into()))
                .collect(),
        }
    }

    /// A score along the given levels, lowest first.
    pub fn score<D>(instructions: impl Into<Value>, levels: impl IntoIterator<Item = D>) -> Question
    where
        D: Into<Value>,
    {
        Question::Score {
            instructions: instructions.into(),
            levels: levels.into_iter().map(Into::into).collect(),
        }
    }

    /// The wire's name for the kind: `noul`, `choice`, `score`.
    pub fn kind(&self) -> &'static str {
        match self {
            Question::Noul { .. } => "noul",
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
        }
    }
}

/// One typed answer, of the kind its question was.
///
/// Every probability is the model's own, in `0..=1`. `confidence` on a
/// choice or a score is the provider's number, derived from the
/// distribution; a noul carries none, because its one probability already
/// is one. What a caller does with a confidence — act, review, or hand the
/// row to a reasoning model — is the caller's threshold and not this
/// crate's.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// The probability the answer is yes.
    Noul { yes: f64 },
    Choice {
        /// The highest-probability label.
        choice: String,
        /// Every option the question named, to its probability.
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Score {
        /// The probability-weighted level, which may fall between two.
        score: f64,
        /// One probability per level, in the order the question gave them.
        probabilities: Vec<f64>,
        confidence: f64,
    },
}

impl Answer {
    /// The wire's name for the kind: `noul`, `choice`, `score`.
    pub fn kind(&self) -> &'static str {
        match self {
            Answer::Noul { .. } => "noul",
            Answer::Choice { .. } => "choice",
            Answer::Score { .. } => "score",
        }
    }
}

/// The answers, one per question asked, under the caller's names.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// The model that answered, as the provider names it.
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    /// `input` and `output`; the cache counters stay zero, the wire has no
    /// cache to report.
    pub usage: Usage,
}
