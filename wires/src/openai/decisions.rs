//! OpenAI's Decisions API: the OpenAI wire's [`Judge`].
//!
//! `POST {endpoint}/decisions` with the key in `Authorization: Bearer`, the
//! third body the OpenAI wire speaks beside `/chat/completions` and
//! `/embeddings`, and built the way embeddings are: [`build_judge`] on the
//! same [`Wire::OpenAi`]. A separate client, because a decision model cannot
//! chat and a chat model cannot decide (`gpt-6-luna` is the only model the
//! endpoint takes at the writing).
//!
//! Read from OpenAI's Decisions guide and pinned against live replies
//! (2026-10-25, `gpt-6-luna`). The four kinds of [`Question`] map onto the
//! three the API has:
//!
//! | [`Question`] | `/v1/decisions` | answered as |
//! |---|---|---|
//! | `Noul` | `predicate` | `probability` → [`Answer::Noul`]`{ yes }` |
//! | `Choice` | `choice`, `choices: [{value, description}]` | the choice, every option's probability, `confidence` |
//! | `Score` | `score`, `levels: [{label, description}]`, 2 to 10 | the weighted index, one probability per level, `confidence` |
//! | `Box` | — | refused before the socket: the API has no locating question |
//!
//! Every answer keeps the distribution the API returned; nothing here
//! collapses one to its top option.
//!
//! **The question ids never reach the model**, the rule TypeSafe's wire
//! keeps by sending them as map keys: here every question goes out named
//! `q0`, `q1`, … in the caller's order, and the answers come back under the
//! caller's names.
//!
//! **A refusal is an error, and the verdict has no hole.** The API answers a
//! question it will not decide with `{"type": "refusal", "name": …}` while
//! answering the others. A step that routes on one answer must not find it
//! missing, so a reply carrying any refusal is [`Error::Refused`], naming
//! every refused question by the caller's name.
//!
//! Where the state goes: text is `input` as it is; a JSON object or array is
//! one user message holding an `input_json` part; images go in that same
//! message, each after a text part holding its reference, so a question that
//! names `photo` reads the image right after the word. No retries, as on
//! every wire here: a 429's `retry-after` is on the error and the run above
//! decides.
//!
//! [`build_judge`]: crate::build_judge
//! [`Wire::OpenAi`]: crate::Wire::OpenAi

use std::collections::BTreeMap;

use async_trait::async_trait;
use reqwest::header::AUTHORIZATION;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use wire_secret::Secret;

use crate::{
    Answer, Error, Headers, Info, Judge, Judgement, Question, Result, Usage, Verdict, http,
};

const WIRE: &str = "openai";

/// The API's bounds, checked here so the caller is told which question and
/// why rather than reading a 400's `param`.
const MAX_QUESTIONS: usize = 400;
const LEVELS: std::ops::RangeInclusive<usize> = 2..=10;

/// A client for one decision model.
///
/// No `Debug` derive and no key field, for the reason the chat and embedding
/// clients have neither.
pub(crate) struct Decisions {
    http: reqwest::Client,
    url: reqwest::Url,
    model: String,
    endpoint: String,
}

impl Decisions {
    /// OpenAI, and anything that serves `/decisions` at another host behind
    /// the same endpoint. The operator's headers go on first and the key
    /// last, as on the chat wire.
    pub(crate) fn openai(
        endpoint: &str,
        model: &str,
        headers: &Headers,
        key: &Secret,
    ) -> Result<Decisions> {
        let url = http::join(endpoint, "decisions", None)?;
        let mut map = http::extra(WIRE, headers)?;
        map.insert(AUTHORIZATION, http::key_header(WIRE, key, "Bearer ")?);
        Ok(Decisions {
            http: http::client(map)?,
            url,
            model: model.to_string(),
            endpoint: endpoint.to_string(),
        })
    }
}

#[async_trait]
impl Judge for Decisions {
    async fn judge(&self, req: Judgement) -> Result<Verdict> {
        let (body, names) = request(&req, &self.model)?;
        let text = http::post_json(&self.http, &self.url, &body)
            .await?
            .text()
            .await?;
        response(&text, &req.questions, &names)
    }

    fn info(&self) -> Info {
        Info {
            wire: WIRE,
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }
}

impl std::fmt::Debug for Decisions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decisions")
            .field("wire", &WIRE)
            .field("model", &self.model)
            .field("url", &self.url.as_str())
            .finish()
    }
}

// ------------------------------------------------------------------ request

/// The request, and the caller's name for each `qN`, in order.
pub(crate) fn request(req: &Judgement, model: &str) -> Result<(Value, Vec<String>)> {
    let invalid = |what: String| Error::Invalid { wire: WIRE, what };
    if req.questions.is_empty() {
        return Err(invalid("a judgement needs at least one question".into()));
    }
    if req.questions.len() > MAX_QUESTIONS {
        return Err(invalid(format!(
            "{} questions; the decisions API takes at most {MAX_QUESTIONS}",
            req.questions.len()
        )));
    }

    let mut names = Vec::with_capacity(req.questions.len());
    let mut questions = Vec::with_capacity(req.questions.len());
    for (i, (name, q)) in req.questions.iter().enumerate() {
        let wire_name = format!("q{i}");
        let question = match q {
            Question::Noul {
                instructions,
                yes,
                no,
            } => {
                // A predicate has no criteria of its own; described outcomes
                // go in as further instruction parts, after the question.
                let mut parts = vec![instructions.clone()];
                if !yes.is_null() {
                    parts.push(Value::String(format!("Yes means: {}", text(yes))));
                }
                if !no.is_null() {
                    parts.push(Value::String(format!("No means: {}", text(no))));
                }
                json!({
                    "type": "predicate",
                    "name": wire_name,
                    "instructions": instruction_parts(&parts),
                })
            }
            Question::Choice {
                instructions,
                options,
            } => {
                if options.is_empty() {
                    return Err(invalid(format!(
                        "question {name:?}: a choice needs at least one option"
                    )));
                }
                let choices: Vec<Value> = options
                    .iter()
                    .map(|(label, d)| {
                        let mut c = json!({ "value": label });
                        if !d.is_null() {
                            c["description"] = Value::String(text(d));
                        }
                        c
                    })
                    .collect();
                json!({
                    "type": "choice",
                    "name": wire_name,
                    "instructions": instruction_parts(std::slice::from_ref(instructions)),
                    "choices": choices,
                })
            }
            Question::Score {
                instructions,
                levels,
            } => {
                if !LEVELS.contains(&levels.len()) {
                    return Err(invalid(format!(
                        "question {name:?}: a score takes 2 to 10 levels on this wire, got {}",
                        levels.len()
                    )));
                }
                // The level's index is its label: the question gave the
                // levels in order, and the answer's probabilities come back
                // keyed by index.
                let levels: Vec<Value> = levels
                    .iter()
                    .enumerate()
                    .map(|(i, d)| json!({ "label": i.to_string(), "description": text(d) }))
                    .collect();
                json!({
                    "type": "score",
                    "name": wire_name,
                    "instructions": instruction_parts(std::slice::from_ref(instructions)),
                    "levels": levels,
                })
            }
            Question::Box { .. } => {
                return Err(invalid(format!(
                    "question {name:?}: the decisions API has no box question; ask it of a \
                     typesafe judge"
                )));
            }
        };
        names.push(name.clone());
        questions.push(question);
    }

    Ok((
        json!({ "model": model, "input": input(req)?, "questions": questions }),
        names,
    ))
}

/// The shared evidence. Text alone is the string; anything else is one user
/// message, which is where JSON and images may go.
fn input(req: &Judgement) -> Result<Value> {
    if req.assets.is_empty()
        && let Value::String(s) = &req.state
    {
        return Ok(Value::String(s.clone()));
    }
    let mut content = vec![state_part(&req.state)];
    for (reference, image) in &req.assets {
        if image.base64_parts().is_none() {
            return Err(Error::Invalid {
                wire: WIRE,
                what: format!(
                    "asset {reference:?}: a data URL that is not base64; the wire takes PNG or \
                     JPEG bytes"
                ),
            });
        }
        content.push(json!({ "type": "input_text", "text": reference }));
        content.push(json!({ "type": "input_image", "image_url": image.data_url() }));
    }
    Ok(json!([{ "role": "user", "content": content }]))
}

fn state_part(state: &Value) -> Value {
    match state {
        Value::Object(_) | Value::Array(_) => json!({ "type": "input_json", "data": state }),
        other => json!({ "type": "input_text", "text": text(other) }),
    }
}

/// One instruction string, or the parts when there are several. A structured
/// instruction (an object or an array, the TypeSafe docs' advice for a hard
/// question) goes in as its JSON text: an instruction here is text or text
/// parts.
fn instruction_parts(parts: &[Value]) -> Value {
    let texts: Vec<String> = parts.iter().map(text).filter(|t| !t.is_empty()).collect();
    match texts.as_slice() {
        [] => Value::String(String::new()),
        [one] => Value::String(one.clone()),
        many => Value::Array(
            many.iter()
                .map(|t| json!({ "type": "input_text", "text": t }))
                .collect(),
        ),
    }
}

/// A leaf as the text the API reads: a string as itself, `null` as nothing,
/// anything else as its JSON.
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

// ----------------------------------------------------------------- response

#[derive(Debug, Deserialize)]
struct Response {
    #[serde(default)]
    model: String,
    #[serde(default)]
    answers: Vec<WireAnswer>,
    usage: Option<WireUsage>,
}

/// One answer. Every field optional, so the shape check names what is
/// missing.
#[derive(Debug, Deserialize)]
struct WireAnswer {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    name: String,
    probability: Option<f64>,
    choice: Option<String>,
    score: Option<f64>,
    confidence: Option<f64>,
    probabilities: Option<Vec<WireProbability>>,
}

#[derive(Debug, Deserialize)]
struct WireProbability {
    /// A choice's label, or a score level's index.
    value: Value,
    probability: f64,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    input_tokens_details: Option<InputDetails>,
    #[serde(default)]
    output_tokens_details: Option<OutputDetails>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct InputDetails {
    #[serde(default)]
    cached_tokens: u32,
    #[serde(default)]
    cache_write_tokens: u32,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct OutputDetails {
    #[serde(default)]
    reasoning_tokens: u32,
}

pub(crate) fn response(
    text: &str,
    asked: &BTreeMap<String, Question>,
    names: &[String],
) -> Result<Verdict> {
    let resp: Response = serde_json::from_str(text)
        .map_err(|e| Error::Decode(format!("{e} in {}", http::clip(text))))?;
    let by_name: BTreeMap<&str, &WireAnswer> =
        resp.answers.iter().map(|a| (a.name.as_str(), a)).collect();

    let mut refused = Vec::new();
    let mut answers = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        let question = &asked[name];
        let Some(got) = by_name.get(format!("q{i}").as_str()) else {
            return Err(Error::Decode(format!(
                "no answer for question {name:?} in {}",
                http::clip(text)
            )));
        };
        if got.kind == "refusal" {
            refused.push(name.clone());
            continue;
        }
        let expected = match question {
            Question::Noul { .. } => "predicate",
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Box { .. } => unreachable!("refused before the socket"),
        };
        if got.kind != expected {
            return Err(Error::Decode(format!(
                "question {name:?} is a {expected} and was answered as a {:?}",
                got.kind
            )));
        }
        answers.insert(name.clone(), answer(name, question, got)?);
    }
    if !refused.is_empty() {
        return Err(Error::Refused {
            wire: WIRE,
            questions: refused,
        });
    }

    let usage = resp.usage.unwrap_or_default();
    let details = usage.input_tokens_details.unwrap_or_default();
    Ok(Verdict {
        model: resp.model,
        answers,
        usage: Usage {
            input: usage.input_tokens,
            output: usage.output_tokens,
            cached: details.cached_tokens,
            cache_creation: details.cache_write_tokens,
            reasoning: usage
                .output_tokens_details
                .map(|d| d.reasoning_tokens)
                .unwrap_or(0),
        },
    })
}

fn answer(name: &str, question: &Question, got: &WireAnswer) -> Result<Answer> {
    let field =
        |what: &str| Error::Decode(format!("{} answer for {name:?} has no {what}", got.kind));
    Ok(match question {
        Question::Noul { .. } => Answer::Noul {
            yes: got.probability.ok_or_else(|| field("probability"))?,
        },
        Question::Choice { options, .. } => {
            let given = got
                .probabilities
                .as_ref()
                .ok_or_else(|| field("probabilities"))?;
            let mut probabilities = BTreeMap::new();
            for p in given {
                if let Value::String(label) = &p.value {
                    probabilities.insert(label.clone(), p.probability);
                }
            }
            // Every option the question named, or the map would read an
            // absent one as never considered.
            if let Some(missing) = options.keys().find(|o| !probabilities.contains_key(*o)) {
                return Err(Error::Decode(format!(
                    "choice answer for {name:?} has no probability for option {missing:?}"
                )));
            }
            Answer::Choice {
                choice: got.choice.clone().ok_or_else(|| field("choice"))?,
                probabilities,
                confidence: got.confidence.ok_or_else(|| field("confidence"))?,
            }
        }
        Question::Score { levels, .. } => {
            let given = got
                .probabilities
                .as_ref()
                .ok_or_else(|| field("probabilities"))?;
            let mut probabilities = Vec::with_capacity(levels.len());
            for i in 0..levels.len() {
                let p = given
                    .iter()
                    .find(|p| p.value.as_u64() == Some(i as u64))
                    .ok_or_else(|| {
                        Error::Decode(format!(
                            "score answer for {name:?} has no probability for level {i} of {}",
                            levels.len()
                        ))
                    })?;
                probabilities.push(p.probability);
            }
            Answer::Score {
                score: got.score.ok_or_else(|| field("score"))?,
                probabilities,
                confidence: got.confidence.ok_or_else(|| field("confidence"))?,
            }
        }
        Question::Box { .. } => unreachable!("refused before the socket"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_does_not_print() {
        let c = Decisions::openai(
            "https://api.openai.com/v1",
            "gpt-6-luna",
            &Headers::new(),
            &Secret::from("sk-do-not-log-me"),
        )
        .unwrap();
        let shown = format!("{c:?}");
        assert!(!shown.contains("do-not-log-me"), "{shown}");
        assert!(
            shown.contains("https://api.openai.com/v1/decisions"),
            "{shown}"
        );
    }
}
