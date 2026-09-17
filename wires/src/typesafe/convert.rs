//! Between a [`Judgement`] and the System One request, and between its
//! response and a [`Verdict`].
//!
//! The request side is a straight spelling of the SDK's `SystemOneRequest`.
//! The response side is where the **200-that-is-not-what-we-asked** rule
//! lives for this wire: a body that parses but answers a question we did not
//! ask, skips one we did, or answers a score question with a choice is a
//! [`Error::Decode`], not a verdict with a hole in it — a solid step that
//! routes on `answers["is_urgent"]` must not find it missing and call that a
//! no.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Answer, Error, Judgement, Question, Result, Usage, Verdict, http};

use super::WIRE;

// ------------------------------------------------------------------ request

/// `POST /v1/systemone`.
#[derive(Debug, Serialize)]
pub(crate) struct Request<'a> {
    pub(crate) state: &'a Value,
    pub(crate) model: &'a str,
    pub(crate) questions: BTreeMap<&'a str, WireQuestion<'a>>,
}

/// One question, tagged by `type`. `instructions` is optional on the wire
/// and is left out when the caller gave `Value::Null`, so the body a caller
/// who did not think about it sends is the body their SDK would send.
#[derive(Debug, Serialize)]
pub(crate) struct WireQuestion<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    #[serde(skip_serializing_if = "Value::is_null")]
    pub(crate) instructions: &'a Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) criteria: Option<Criteria<'a>>,
}

/// The kind's own criteria shape.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum Criteria<'a> {
    /// `{ "true": …, "false": … }`, each present only when described.
    Noul {
        #[serde(rename = "true", skip_serializing_if = "Value::is_null")]
        yes: &'a Value,
        #[serde(rename = "false", skip_serializing_if = "Value::is_null")]
        no: &'a Value,
    },
    /// Label to description. A `null` description goes out as `null`: the
    /// label is the option, and the wire reads `null` as "undescribed".
    Choice(&'a BTreeMap<String, Value>),
    /// The levels in order.
    Score(&'a [Value]),
}

pub(crate) fn request<'a>(req: &'a Judgement, model: &'a str) -> Result<Request<'a>> {
    if req.questions.is_empty() {
        return Err(Error::Invalid {
            wire: WIRE,
            what: "a judgement needs at least one question".into(),
        });
    }
    let mut questions = BTreeMap::new();
    for (name, q) in &req.questions {
        let (instructions, criteria) = match q {
            Question::Noul {
                instructions,
                yes,
                no,
            } => (
                instructions,
                (!yes.is_null() || !no.is_null()).then_some(Criteria::Noul { yes, no }),
            ),
            Question::Choice {
                instructions,
                options,
            } => {
                if options.is_empty() {
                    return Err(Error::Invalid {
                        wire: WIRE,
                        what: format!("question {name:?}: a choice needs at least one option"),
                    });
                }
                (instructions, Some(Criteria::Choice(options)))
            }
            Question::Score {
                instructions,
                levels,
            } => {
                if levels.len() < 2 {
                    return Err(Error::Invalid {
                        wire: WIRE,
                        what: format!(
                            "question {name:?}: a score needs at least two levels, got {}",
                            levels.len()
                        ),
                    });
                }
                (instructions, Some(Criteria::Score(levels)))
            }
        };
        questions.insert(
            name.as_str(),
            WireQuestion {
                kind: q.kind(),
                instructions,
                criteria,
            },
        );
    }
    Ok(Request {
        state: &req.state,
        model,
        questions,
    })
}

// ----------------------------------------------------------------- response

#[derive(Debug, Deserialize)]
struct Response {
    #[serde(default)]
    model: String,
    #[serde(default)]
    answers: BTreeMap<String, WireAnswer>,
    usage: Option<WireUsage>,
}

/// One answer, tagged by `type`. Every field is optional at this layer so
/// that the *shape* check below can name what is missing, rather than serde
/// reporting `missing field` for whichever it noticed first.
#[derive(Debug, Deserialize)]
struct WireAnswer {
    #[serde(rename = "type", default)]
    kind: String,
    noul: Option<f64>,
    choice: Option<String>,
    score: Option<f64>,
    confidence: Option<f64>,
    probabilities: Option<BTreeMap<String, f64>>,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

pub(crate) fn response(text: &str, asked: &BTreeMap<String, Question>) -> Result<Verdict> {
    let resp: Response = serde_json::from_str(text)
        .map_err(|e| Error::Decode(format!("{e} in {}", http::clip(text))))?;

    let mut answers = BTreeMap::new();
    for (name, question) in asked {
        let Some(got) = resp.answers.get(name) else {
            return Err(Error::Decode(format!(
                "no answer for question {name:?} in {}",
                http::clip(text)
            )));
        };
        if got.kind != question.kind() {
            return Err(Error::Decode(format!(
                "question {name:?} is a {} and was answered as a {:?}",
                question.kind(),
                got.kind
            )));
        }
        answers.insert(name.clone(), answer(name, question, got)?);
    }
    // An answer to a question nobody asked is ignored, not an error: a
    // server that adds one must not end a step, and the caller reads by name.

    Ok(Verdict {
        model: resp.model,
        answers,
        usage: resp
            .usage
            .map(|u| Usage {
                input: u.input_tokens,
                output: u.output_tokens,
                ..Usage::default()
            })
            .unwrap_or_default(),
    })
}

fn answer(name: &str, question: &Question, got: &WireAnswer) -> Result<Answer> {
    let field = |what: &str| {
        Error::Decode(format!(
            "{} answer for {name:?} has no {what}",
            question.kind()
        ))
    };
    Ok(match question {
        Question::Noul { .. } => Answer::Noul {
            yes: got.noul.ok_or_else(|| field("noul"))?,
        },
        Question::Choice { .. } => Answer::Choice {
            choice: got.choice.clone().ok_or_else(|| field("choice"))?,
            probabilities: got
                .probabilities
                .clone()
                .ok_or_else(|| field("probabilities"))?,
            confidence: got.confidence.ok_or_else(|| field("confidence"))?,
        },
        Question::Score { levels, .. } => {
            let by_level = got
                .probabilities
                .as_ref()
                .ok_or_else(|| field("probabilities"))?;
            // The wire keys these by the level's index as a string; the
            // question gave the levels in order, so a `Vec` in that order is
            // the honest shape. Every level must be present, or the vector
            // would silently shift.
            let mut probabilities = Vec::with_capacity(levels.len());
            for i in 0..levels.len() {
                let p = by_level.get(&i.to_string()).ok_or_else(|| {
                    Error::Decode(format!(
                        "score answer for {name:?} has no probability for level {i} of {}",
                        levels.len()
                    ))
                })?;
                probabilities.push(*p);
            }
            Answer::Score {
                score: got.score.ok_or_else(|| field("score"))?,
                probabilities,
                confidence: got.confidence.ok_or_else(|| field("confidence"))?,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn asked() -> Judgement {
        Judgement::of("Help! My payouts have been failing for 3 days.")
            .ask("is_urgent", Question::noul("Does this convey urgency?"))
            .ask(
                "department",
                Question::choice(
                    "Which team should handle this?",
                    [
                        ("billing", json!("Payments, invoicing, refunds")),
                        ("technical", json!("Bugs, outages, integrations")),
                        ("sales", Value::Null),
                    ],
                ),
            )
            .ask(
                "frustration",
                Question::score(
                    "How frustrated is the customer?",
                    ["Calm", "Frustrated", "Very angry"],
                ),
            )
    }

    #[test]
    fn the_request_is_the_sdks_payload() {
        let req = asked();
        let body = serde_json::to_value(request(&req, "jev-latest").unwrap()).unwrap();
        assert_eq!(
            body,
            json!({
                "state": "Help! My payouts have been failing for 3 days.",
                "model": "jev-latest",
                "questions": {
                    // No criteria on a noul that described neither outcome.
                    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                    "department": {
                        "type": "choice",
                        "instructions": "Which team should handle this?",
                        // Sorted, and a null description goes out as null.
                        "criteria": {
                            "billing": "Payments, invoicing, refunds",
                            "sales": null,
                            "technical": "Bugs, outages, integrations",
                        }
                    },
                    "frustration": {
                        "type": "score",
                        "instructions": "How frustrated is the customer?",
                        // An array, indexed from zero — not a map.
                        "criteria": ["Calm", "Frustrated", "Very angry"]
                    }
                }
            })
        );
    }

    #[test]
    fn a_described_noul_and_structured_leaves_go_out_as_given() {
        let req = Judgement::of(json!({"messages": [{"role": "user", "text": "hi"}]})).ask(
            "greeting",
            Question::noul_described(
                json!(["Is the first message a greeting?", "Ignore the rest."]),
                "Opens with a salutation",
                Value::Null,
            ),
        );
        let body = serde_json::to_value(request(&req, "jev-latest").unwrap()).unwrap();
        assert_eq!(
            body["state"],
            json!({"messages": [{"role": "user", "text": "hi"}]})
        );
        assert_eq!(
            body["questions"]["greeting"],
            json!({
                "type": "noul",
                "instructions": ["Is the first message a greeting?", "Ignore the rest."],
                "criteria": {"true": "Opens with a salutation"}
            })
        );
        // No instructions at all is allowed, and is the field left out.
        let req = Judgement::of("x").ask("q", Question::noul(Value::Null));
        let body = serde_json::to_value(request(&req, "m").unwrap()).unwrap();
        assert_eq!(body["questions"]["q"], json!({"type": "noul"}));
    }

    #[test]
    fn what_the_server_would_refuse_is_refused_before_the_socket() {
        let cases: Vec<(Judgement, &str)> = vec![
            (Judgement::of("x"), "at least one question"),
            (
                Judgement::of("x").ask("tone", Question::score("?", ["only one"])),
                "at least two levels",
            ),
            (
                Judgement::of("x").ask("team", Question::choice("?", Vec::<(&str, Value)>::new())),
                "at least one option",
            ),
        ];
        for (req, want) in cases {
            match request(&req, "m") {
                Err(Error::Invalid { wire, what }) => {
                    assert_eq!(wire, "typesafe");
                    assert!(what.contains(want), "{what}");
                }
                other => panic!("expected Invalid {want}, got {:?}", other.err()),
            }
        }
    }

    #[test]
    fn the_response_comes_back_typed_under_the_callers_names() {
        let v = response(ANSWERS, &asked().questions).unwrap();
        assert_eq!(v.model, "jev-latest");
        assert_eq!(
            v.usage,
            Usage {
                input: 312,
                output: 48,
                ..Usage::default()
            }
        );
        assert_eq!(v.answers["is_urgent"], Answer::Noul { yes: 0.92 });
        assert_eq!(
            v.answers["department"],
            Answer::Choice {
                choice: "technical".into(),
                probabilities: [("billing", 0.08), ("technical", 0.85), ("sales", 0.07)]
                    .into_iter()
                    .map(|(k, p)| (k.to_string(), p))
                    .collect(),
                confidence: 0.82,
            }
        );
        assert_eq!(
            v.answers["frustration"],
            Answer::Score {
                score: 1.6,
                // In level order, from the string-keyed map.
                probabilities: vec![0.05, 0.3, 0.65],
                confidence: 0.78,
            }
        );
    }

    #[test]
    fn a_200_that_is_not_what_we_asked_is_a_decode_error() {
        let asked = asked().questions;
        let cases = [
            // A question skipped.
            (
                r#"{"model":"m","answers":{"is_urgent":{"type":"noul","noul":0.9}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
                "no answer for question \"department\"",
            ),
            // A question answered as another kind.
            (
                r#"{"model":"m","answers":{
                    "is_urgent":{"type":"choice","choice":"yes","confidence":1,"probabilities":{"yes":1}},
                    "department":{"type":"choice","choice":"sales","confidence":1,"probabilities":{"sales":1}},
                    "frustration":{"type":"score","score":0,"confidence":1,"probabilities":{"0":1,"1":0,"2":0}}
                }}"#,
                "\"is_urgent\" is a noul and was answered as a \"choice\"",
            ),
            // A score with a level missing from its distribution.
            (
                r#"{"model":"m","answers":{
                    "is_urgent":{"type":"noul","noul":0.9},
                    "department":{"type":"choice","choice":"sales","confidence":1,"probabilities":{"sales":1}},
                    "frustration":{"type":"score","score":0,"confidence":1,"probabilities":{"0":1,"2":0}}
                }}"#,
                "no probability for level 1 of 3",
            ),
            // A choice with no confidence.
            (
                r#"{"model":"m","answers":{
                    "is_urgent":{"type":"noul","noul":0.9},
                    "department":{"type":"choice","choice":"sales","probabilities":{"sales":1}},
                    "frustration":{"type":"score","score":0,"confidence":1,"probabilities":{"0":1,"1":0,"2":0}}
                }}"#,
                "choice answer for \"department\" has no confidence",
            ),
            // Not JSON at all.
            ("<html>ok</html>", "expected value"),
        ];
        for (body, want) in cases {
            match response(body, &asked) {
                Err(Error::Decode(why)) => assert!(why.contains(want), "{why}"),
                other => panic!("expected Decode {want:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_answer_nobody_asked_for_is_ignored() {
        let asked = Judgement::of("x")
            .ask("is_urgent", Question::noul("?"))
            .questions;
        let v = response(ANSWERS, &asked).unwrap();
        assert_eq!(v.answers.len(), 1);
        assert_eq!(v.answers["is_urgent"], Answer::Noul { yes: 0.92 });
    }

    /// The docs' example response for the three example questions.
    const ANSWERS: &str = r#"{
      "model": "jev-latest",
      "answers": {
        "is_urgent": {"type": "noul", "noul": 0.92},
        "department": {
          "type": "choice",
          "choice": "technical",
          "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
          "confidence": 0.82
        },
        "frustration": {
          "type": "score",
          "score": 1.6,
          "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
          "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
          "confidence": 0.78
        }
      },
      "usage": {"input_tokens": 312, "output_tokens": 48}
    }"#;
}
