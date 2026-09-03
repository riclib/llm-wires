//! The SSE reader: response body → [`Chunk`]s.
//!
//! The port of Go's `openai/stream.go`, with the two bugs its own doc names
//! designed out. There is no `Recv` that ignores a context — a caller selects
//! on a `Stream` — and there is no `Close` to forget: the body hangs off the
//! stream, so dropping the stream is what closes the connection.

use std::collections::{BTreeMap, VecDeque};

use serde::Deserialize;

use super::convert;
use crate::{Chunk, ChunkStream, Error, Finish, Result, ToolCall, ToolCallDelta, sse};

/// The body, framed and mapped.
pub(crate) fn chunks(resp: reqwest::Response) -> ChunkStream {
    let reader = Reader {
        frames: sse::Frames::new(resp),
        done: false,
        pending: VecDeque::new(),
        calls: BTreeMap::new(),
    };
    Box::pin(futures_util::stream::unfold(reader, |mut r| async move {
        loop {
            if let Some(chunk) = r.pending.pop_front() {
                return Some((Ok(chunk), r));
            }
            if r.done {
                return None;
            }
            match r.frames.next_data().await {
                // `[DONE]` is this wire's end-of-body marker, and the wire's
                // to know: SSE has no such thing.
                Ok(Some(data)) if data != DONE => {
                    if let Err(e) = r.frame(&data) {
                        r.done = true;
                        return Some((Err(e), r));
                    }
                }
                Ok(_) => {
                    r.done = true;
                    if let Err(e) = r.ended() {
                        return Some((Err(e), r));
                    }
                }
                Err(e) => {
                    r.done = true;
                    return Some((Err(e), r));
                }
            }
        }
    }))
}

/// What this wire says instead of just closing the body.
const DONE: &str = "[DONE]";

struct Reader {
    frames: sse::Frames,
    done: bool,
    /// Chunks a single frame produced, in order. A frame can carry several
    /// tool-call fragments; Go returned on the first and lost the rest.
    pending: VecDeque<Chunk>,
    /// Tool calls under construction, by the wire's index.
    calls: BTreeMap<u32, ToolCall>,
}

impl Reader {
    /// One frame, mapped onto zero or more chunks.
    fn frame(&mut self, data: &str) -> Result<()> {
        let frame: WireChunk = serde_json::from_str(data)
            .map_err(|e| Error::Decode(format!("{e} in a stream frame")))?;

        // Some gateways report a mid-stream failure as an ordinary frame
        // rather than an HTTP status. Without this arm the turn would end in
        // a clean EOF with no answer and nothing to explain it. It is its own
        // variant and not an `Api` with a made-up status: there was no status,
        // and "the provider answered 200: overloaded" reads as a contradiction
        // to whoever is holding the failure row.
        if let Some(err) = frame.error {
            return Err(Error::Stream {
                message: err.message,
            });
        }

        // Read up front, not inside the empty-choices arm: native OpenAI and
        // Azure send usage on a trailing chunk with NO choices, while
        // OpenRouter puts it on the same final chunk that carries the finish
        // reason. Reading it here and attaching it to whatever this frame
        // emits captures both shapes — the bug that logged 0 tokens for every
        // OpenRouter-backed model.
        let mut usage = frame.usage.map(convert::usage).filter(|u| !u.is_zero());

        if let Some(choice) = frame.choices.into_iter().next() {
            // Content and fragments FIRST, the finish LAST, because a frame
            // may carry both. Native OpenAI splits them; an OpenAI-compatible
            // server — ollama, a gateway — does not have to, and this wire is
            // the one that serves them. Finishing first would assemble a tool
            // call without the fragment sitting beside the reason in the same
            // frame: `{"query":` with a `Finish::ToolCalls` next to it saying
            // the turn is ready to run. Go's order, and Go's bug.
            if let Some(text) = choice.delta.content.filter(|c| !c.is_empty()) {
                self.pending.push_back(Chunk {
                    delta: text,
                    ..Chunk::default()
                });
            }

            for d in choice.delta.tool_calls {
                let index = d.index;
                let entry = self.calls.entry(index).or_default();
                // Identity arrives on the first fragment, but a gateway that
                // splits it across two must not leave the call anonymous.
                if entry.id.is_empty()
                    && let Some(id) = d.id
                {
                    entry.id = id;
                }
                if entry.name.is_empty()
                    && let Some(name) = d.function.name
                {
                    entry.name = name;
                }
                let Some(args) = d.function.arguments.filter(|a| !a.is_empty()) else {
                    continue;
                };
                entry.arguments.push_str(&args);
                let (id, name) = (entry.id.clone(), entry.name.clone());
                self.pending.push_back(Chunk {
                    tool_call_delta: Some(ToolCallDelta {
                        index,
                        id,
                        name,
                        delta: args,
                    }),
                    ..Chunk::default()
                });
            }

            if let Some(reason) = choice.finish_reason.filter(|s| !s.is_empty()) {
                // Where accumulated tool calls become complete. Note what does
                // NOT happen here: the reader is not marked done. With
                // include_usage set the usage frame arrives AFTER this one,
                // and a reader that closed here would drop it.
                let calls = std::mem::take(&mut self.calls);
                self.pending.push_back(Chunk {
                    tool_calls: calls.into_values().collect(),
                    finish: Some(Finish::parse(&reason)),
                    usage: usage.take(),
                    ..Chunk::default()
                });
            }
        }

        // Usage with no frame of its own to ride on gets one — the trailing
        // empty-choices frame, and any shape nobody has sent yet.
        if let Some(usage) = usage {
            self.pending.push_back(Chunk {
                usage: Some(usage),
                ..Chunk::default()
            });
        }
        // Anything else — the role-only opening frame, an empty delta — is
        // not an event a caller has any use for.
        Ok(())
    }

    /// What the end of the body means when a tool call was still being built.
    ///
    /// It is an error, not a last chunk. Arguments assembled from fragments
    /// with no finish frame behind them are not known to be complete, and the
    /// caller this feeds runs them: handing on `{"query":` as if it were an
    /// instruction is worse than saying the turn failed, and the turn *did*
    /// fail — a body that stops mid-call is a gateway that died, which the
    /// run engine should retry rather than half-execute. Text is not treated
    /// the same way, because partial text is partial and a tool call is not.
    fn ended(&mut self) -> Result<()> {
        if self.calls.is_empty() {
            return Ok(());
        }
        let names: Vec<String> = std::mem::take(&mut self.calls)
            .into_values()
            .map(|c| {
                if c.name.is_empty() {
                    "?".into()
                } else {
                    c.name
                }
            })
            .collect();
        Err(Error::Stream {
            message: format!(
                "the body ended mid tool call, with {} still unfinished ({})",
                names.len(),
                names.join(", ")
            ),
        })
    }
}

// ------------------------------------------------------------ the frame shape

#[derive(Debug, Deserialize)]
struct WireChunk {
    #[serde(default)]
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<convert::WireUsage>,
    #[serde(default)]
    error: Option<WireError>,
}

#[derive(Debug, Deserialize)]
struct WireError {
    #[serde(default)]
    message: String,
}

#[derive(Debug, Deserialize)]
struct WireChoice {
    #[serde(default)]
    delta: WireDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct WireToolCallDelta {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: WireFunctionDelta,
}

#[derive(Debug, Default, Deserialize)]
struct WireFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader() -> Reader {
        Reader {
            frames: sse::Frames::over(Box::pin(futures_util::stream::empty())),
            done: false,
            pending: VecDeque::new(),
            calls: BTreeMap::new(),
        }
    }

    fn frames(r: &mut Reader, data: &[&str]) -> Vec<Chunk> {
        for d in data {
            r.frame(d).unwrap();
        }
        r.pending.drain(..).collect()
    }

    #[test]
    fn a_role_only_frame_is_not_an_event() {
        let mut r = reader();
        let out = frames(
            &mut r,
            &[r#"{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#],
        );
        assert!(out.is_empty(), "{out:?}");
    }

    #[test]
    fn the_finish_frame_does_not_close_the_reader_before_the_usage_frame() {
        let mut r = reader();
        let out = frames(
            &mut r,
            &[
                r#"{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
                // Native OpenAI's trailing frame: no choices, all the tokens.
                r#"{"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":17,
                   "prompt_tokens_details":{"cached_tokens":8},
                   "completion_tokens_details":{"reasoning_tokens":4}}}"#,
            ],
        );
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].delta, "Hello");
        assert_eq!(out[1].finish, Some(Finish::Stop));
        assert_eq!(out[1].usage, None);
        let usage = out[2].usage.expect("the trailing frame carries usage");
        assert_eq!(usage.input, 42);
        assert_eq!(usage.output, 17);
        // Cached is a SUBSET of input on this wire, not an addition.
        assert_eq!(usage.cached, 8);
        assert_eq!(usage.reasoning, 4);
        assert_eq!(
            usage.cache_creation, 0,
            "this wire does not bill cache writes"
        );
    }

    #[test]
    fn usage_rides_the_finish_frame_when_the_gateway_puts_it_there() {
        // OpenRouter's shape: choices AND usage on the same final frame, so
        // an empty-choices guard never sees it.
        let mut r = reader();
        let out = frames(
            &mut r,
            &[
                r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],
                   "usage":{"prompt_tokens":83,"completion_tokens":59,
                   "prompt_tokens_details":{"cached_tokens":64},
                   "completion_tokens_details":{"reasoning_tokens":41}}}"#,
            ],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].finish, Some(Finish::Stop));
        let usage = out[0].usage.expect("usage must not be dropped here");
        assert_eq!((usage.input, usage.output), (83, 59));
        assert_eq!((usage.cached, usage.reasoning), (64, 41));
    }

    #[test]
    fn parallel_tool_calls_in_one_frame_all_survive() {
        // Go returned on the first fragment in a frame and lost every other
        // one, along with the call it belonged to.
        let mut r = reader();
        let out = frames(
            &mut r,
            &[
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[
                     {"index":0,"id":"call_a","type":"function","function":{"name":"a","arguments":"{\"x\":"}},
                     {"index":1,"id":"call_b","type":"function","function":{"name":"b","arguments":"{}"}}
                   ]},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[
                     {"index":0,"function":{"arguments":"1}"}}
                   ]},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            ],
        );
        let deltas: Vec<&ToolCallDelta> = out
            .iter()
            .filter_map(|c| c.tool_call_delta.as_ref())
            .collect();
        assert_eq!(deltas.len(), 3);
        assert_eq!(
            (
                deltas[0].index,
                deltas[0].name.as_str(),
                deltas[0].delta.as_str()
            ),
            (0, "a", "{\"x\":")
        );
        assert_eq!(
            (
                deltas[1].index,
                deltas[1].name.as_str(),
                deltas[1].delta.as_str()
            ),
            (1, "b", "{}")
        );
        // The second fragment of call 0 carries the identity it was given
        // first, not an empty one.
        assert_eq!(
            (
                deltas[2].index,
                deltas[2].id.as_str(),
                deltas[2].delta.as_str()
            ),
            (0, "call_a", "1}")
        );

        let last = out.last().unwrap();
        assert_eq!(last.finish, Some(Finish::ToolCalls));
        assert_eq!(
            last.tool_calls,
            vec![
                ToolCall {
                    id: "call_a".into(),
                    name: "a".into(),
                    arguments: r#"{"x":1}"#.into()
                },
                ToolCall {
                    id: "call_b".into(),
                    name: "b".into(),
                    arguments: "{}".into()
                },
            ],
            "assembled in index order, whichever order the fragments arrived"
        );
    }

    #[test]
    fn a_frame_that_finishes_and_carries_payload_keeps_both() {
        // Native OpenAI splits the last fragment from the finish reason; an
        // OpenAI-compatible server need not, and this wire is the one that
        // serves them. Finishing first assembled `{"x":` and called the turn
        // ready to run — invalid JSON, silently, with a Finish::ToolCalls
        // beside it. Go's ordering; not ours.
        let mut r = reader();
        let out = frames(
            &mut r,
            &[
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[
                     {"index":0,"id":"call_a","type":"function","function":{"name":"weigh","arguments":"{\"x\":"}}
                   ]},"finish_reason":null}]}"#,
                // The frame under test: a final fragment AND the reason.
                r#"{"choices":[{"index":0,"delta":{"content":" done.","tool_calls":[
                     {"index":0,"function":{"arguments":"1}"}}
                   ]},"finish_reason":"tool_calls"}],
                   "usage":{"prompt_tokens":9,"completion_tokens":3}}"#,
            ],
        );

        // Three chunks out of two frames: the first fragment, then the
        // second frame's content and fragment, then its finish — in that
        // order, with the finish last.
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].tool_call_delta.as_ref().unwrap().delta, r#"{"x":"#);
        assert_eq!(
            out[1].delta, " done.",
            "the content is not lost to the finish"
        );
        assert_eq!(out[2].tool_call_delta.as_ref().unwrap().delta, "1}");

        let last = &out[3];
        assert_eq!(last.finish, Some(Finish::ToolCalls));
        assert_eq!(
            last.tool_calls,
            vec![ToolCall {
                id: "call_a".into(),
                name: "weigh".into(),
                arguments: r#"{"x":1}"#.into(),
            }],
            "the arguments are complete, because the fragment ran before the finish"
        );
        // And the usage on that same frame still rides the finish chunk.
        let usage = last.usage.expect("usage on the finish frame");
        assert_eq!((usage.input, usage.output), (9, 3));
    }

    #[test]
    fn an_error_frame_ends_the_turn_with_the_reason_and_no_invented_status() {
        let mut r = reader();
        match r.frame(r#"{"error":{"message":"The model is overloaded.","code":"server_error"}}"#) {
            // Not `Api { status: 200 }`: "the provider answered 200: the
            // model is overloaded" reads as a contradiction on a failure row.
            Err(Error::Stream { message }) => {
                assert_eq!(message, "The model is overloaded.");
                assert_eq!(
                    Error::Stream { message }.to_string(),
                    "the provider ended the stream: The model is overloaded."
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_body_that_ends_mid_tool_call_is_refused_rather_than_handed_on() {
        // Fragments, then the body stops: no finish, no [DONE]. The half-built
        // arguments are not known to be complete and the caller would run
        // them, so the turn fails instead of handing on `{"query":`.
        let mut r = reader();
        frames(
            &mut r,
            &[r#"{"choices":[{"index":0,"delta":{"tool_calls":[
                 {"index":0,"id":"call_a","type":"function","function":{"name":"search","arguments":"{\"query\":"}}
               ]},"finish_reason":null}]}"#],
        );
        match r.ended() {
            Err(Error::Stream { message }) => {
                assert!(message.contains("ended mid tool call"), "{message}");
                assert!(message.contains("search"), "{message}");
            }
            other => panic!("{other:?}"),
        }

        // A body that ends after a finish frame has nothing outstanding, and
        // is the benign shape a missing [DONE] produces.
        let mut r = reader();
        frames(
            &mut r,
            &[r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#],
        );
        assert!(r.ended().is_ok());
    }
}
