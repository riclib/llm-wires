//! The SSE reader: message events → [`Chunk`]s.
//!
//! Where the OpenAI wire sends one frame shape with a delta in it, this one
//! sends a small state machine: `message_start`, then a
//! `content_block_start` / `content_block_delta` / `content_block_stop` run
//! per block, then `message_delta` with the stop reason and the output
//! tokens, then `message_stop`. Go leaned on the vendor SDK's accumulator for
//! this; [`Reader::blocks`] is that accumulator, and it is twenty lines.
//!
//! Two things are deliberately not Go's:
//!
//! - **Tool calls ride the finish chunk**, as they do on the other wire, not
//!   the `content_block_stop` that completed them. [`crate::Chunk`] says
//!   "complete tool calls, emitted once with the finish frame", and a caller
//!   that runs tools should not have to know which wire answered it.
//! - **A body that stops with a tool call still open is an error**, not a
//!   quiet EOF — the same rule, and the same reason: the caller *runs* these.

use std::collections::{BTreeMap, VecDeque};

use serde::Deserialize;

use super::convert;
use crate::{Chunk, ChunkStream, Error, Result, ToolCall, ToolCallDelta, Usage, sse};

/// The body, framed and mapped.
pub(crate) fn chunks(resp: reqwest::Response) -> ChunkStream {
    let reader = Reader {
        frames: sse::Frames::new(resp),
        done: false,
        stopped: false,
        pending: VecDeque::new(),
        blocks: BTreeMap::new(),
        usage: Usage::default(),
    };
    Box::pin(futures_util::stream::unfold(reader, |mut r| async move {
        loop {
            if let Some(chunk) = r.pending.pop_front() {
                return Some((Ok(chunk), r));
            }
            if r.done {
                return None;
            }
            // `message_stop` is the end of the turn: nothing follows it, so
            // the body is not read again even if the connection stays open.
            if r.stopped {
                r.done = true;
                if let Err(e) = r.ended() {
                    return Some((Err(e), r));
                }
                continue;
            }
            match r.frames.next_data().await {
                Ok(Some(data)) => {
                    if let Err(e) = r.event(&data) {
                        r.done = true;
                        return Some((Err(e), r));
                    }
                }
                Ok(None) => {
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

/// One `tool_use` content block, assembling.
#[derive(Debug, Default)]
struct ToolBlock {
    /// Which of the turn's calls this is — 0, 1, 2 — and NOT the content
    /// block's index, which counts the text blocks too. [`ToolCallDelta`]'s
    /// `index` means "which of the parallel calls", and it has to mean the
    /// same thing on both wires or a caller has to know which answered it.
    ordinal: u32,
    id: String,
    name: String,
    /// The `input_json_delta` fragments, joined. Empty when the tool takes no
    /// arguments — this wire sends no fragments at all for those.
    arguments: String,
}

struct Reader {
    frames: sse::Frames,
    done: bool,
    /// `message_stop` was seen.
    stopped: bool,
    /// Chunks one event produced, in order.
    pending: VecDeque<Chunk>,
    /// `tool_use` blocks under construction, by content-block index. Text
    /// blocks are not in here: their deltas carry their own text and nothing
    /// has to be remembered about them.
    blocks: BTreeMap<u32, ToolBlock>,
    /// What the turn has cost so far. `message_start` reports the input side
    /// before a token of output exists and `message_delta` the output side at
    /// the end, so neither event has the whole of it and the reader keeps the
    /// running total.
    usage: Usage,
}

impl Reader {
    /// One event, mapped onto zero or more chunks.
    fn event(&mut self, data: &str) -> Result<()> {
        let event: WireEvent = serde_json::from_str(data)
            .map_err(|e| Error::Decode(format!("{e} in a stream event")))?;

        match event.kind.as_str() {
            // A failure the provider reports mid-stream rather than with a
            // status: overloaded, or a turn it will not finish. Without this
            // arm the turn ends in a clean EOF with no answer and nothing to
            // explain it.
            "error" => {
                return Err(Error::Stream {
                    message: event
                        .error
                        .map(|e| e.message)
                        .filter(|m| !m.is_empty())
                        .unwrap_or_else(|| "the provider sent an error event".into()),
                });
            }

            // The input side of the bill, before any output exists. Emitted
            // rather than only remembered: a caller showing cost as a turn
            // runs has the prompt's price from the first event.
            "message_start" => {
                if let Some(u) = event.message.and_then(|m| m.usage) {
                    self.merge(u);
                    self.emit_usage();
                }
            }

            "content_block_start" => {
                if let Some(block) = event.content_block
                    && block.kind == "tool_use"
                {
                    // The map holds only tool_use blocks, and is drained only
                    // by a finish, so its length is how many calls this turn
                    // has opened so far.
                    let ordinal = self.blocks.len() as u32;
                    self.blocks.insert(
                        event.index,
                        ToolBlock {
                            ordinal,
                            id: block.id,
                            name: block.name,
                            arguments: String::new(),
                        },
                    );
                }
            }

            "content_block_delta" => {
                let Some(delta) = event.delta else {
                    return Ok(());
                };
                match delta.kind.as_str() {
                    "text_delta" if !delta.text.is_empty() => {
                        self.pending.push_back(Chunk {
                            delta: delta.text,
                            ..Chunk::default()
                        });
                    }
                    "input_json_delta" if !delta.partial_json.is_empty() => {
                        // A fragment for a block that never started: not a
                        // panic and not a lost turn. Go indexed the
                        // accumulator and logged a warning; there is nothing
                        // to attribute the fragment to either way.
                        let Some(block) = self.blocks.get_mut(&event.index) else {
                            return Ok(());
                        };
                        block.arguments.push_str(&delta.partial_json);
                        self.pending.push_back(Chunk {
                            tool_call_delta: Some(ToolCallDelta {
                                index: block.ordinal,
                                id: block.id.clone(),
                                name: block.name.clone(),
                                delta: delta.partial_json,
                            }),
                            ..Chunk::default()
                        });
                    }
                    // A `thinking_delta`, a `signature_delta`, an empty one:
                    // nothing a caller of this crate reads yet.
                    _ => {}
                }
            }

            // The block is complete. Nothing is emitted here — the assembled
            // call goes out with the finish, so one wire's callers are the
            // other's.
            "content_block_stop" => {}

            // The stop reason and the output side of the bill.
            "message_delta" => {
                if let Some(u) = event.usage {
                    self.merge(u);
                }
                let reason = event.delta.and_then(|d| d.stop_reason);
                match reason.filter(|s| !s.is_empty()) {
                    Some(reason) => {
                        // The only place the blocks are drained, so a caller
                        // never sees half an argument list.
                        let calls: Vec<ToolCall> = std::mem::take(&mut self.blocks)
                            .into_values()
                            .map(|b| ToolCall {
                                id: b.id,
                                name: b.name,
                                // No fragments at all is a tool that takes no
                                // arguments, and `{}` is what it was called
                                // with — not an empty string, which is not
                                // JSON.
                                arguments: if b.arguments.is_empty() {
                                    "{}".into()
                                } else {
                                    b.arguments
                                },
                            })
                            .collect();
                        self.pending.push_back(Chunk {
                            tool_calls: calls,
                            finish: Some(convert::finish(&reason)),
                            usage: self.reportable(),
                            ..Chunk::default()
                        });
                    }
                    // A usage-only `message_delta`, which the wire does not
                    // send today but which costs nothing to carry.
                    None => self.emit_usage(),
                }
            }

            "message_stop" => self.stopped = true,

            // `ping`, and any event type added after this was written. Neither
            // is a failure.
            _ => {}
        }
        Ok(())
    }

    /// Fold an event's counters into the running total.
    ///
    /// Field by field, keeping what was already reported: `message_start`
    /// carries the input tokens and a zero output, `message_delta` the output
    /// tokens and — depending on the deployment — the input ones again or not
    /// at all. Overwriting wholesale would lose whichever half the last event
    /// left out.
    fn merge(&mut self, u: convert::WireUsage) {
        let next = convert::usage(u);
        for (field, value) in [
            (&mut self.usage.input, next.input),
            (&mut self.usage.output, next.output),
            (&mut self.usage.cached, next.cached),
            (&mut self.usage.cache_creation, next.cache_creation),
        ] {
            if value != 0 {
                *field = value;
            }
        }
    }

    /// The running total, or `None` when nothing has been reported — so a
    /// caller can tell "no usage on this event" from "a turn that cost
    /// nothing", which does not happen.
    fn reportable(&self) -> Option<Usage> {
        (!self.usage.is_zero()).then_some(self.usage)
    }

    fn emit_usage(&mut self) {
        if let Some(usage) = self.reportable() {
            self.pending.push_back(Chunk {
                usage: Some(usage),
                ..Chunk::default()
            });
        }
    }

    /// What the end of the body means when a tool call was still open.
    ///
    /// The OpenAI wire's rule, for the OpenAI wire's reason: arguments with no
    /// finish behind them are not known to be complete, and the caller this
    /// feeds *runs* them. A body that stops there is a gateway that died,
    /// which the run engine should retry rather than half-execute. Partial
    /// text is not treated the same way, because partial text is partial and
    /// a tool call is not.
    fn ended(&mut self) -> Result<()> {
        if self.blocks.is_empty() {
            return Ok(());
        }
        let names: Vec<String> = std::mem::take(&mut self.blocks)
            .into_values()
            .map(|b| {
                if b.name.is_empty() {
                    "?".into()
                } else {
                    b.name
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

// ------------------------------------------------------------ the event shape

/// One event, flattened.
///
/// The wire names the event twice — an SSE `event:` line and the payload's own
/// `type` — and this reads the payload's, so a gateway that drops the `event:`
/// line loses nothing. One struct rather than an enum per type because the
/// fields do not collide: `delta` is a block delta on one event and the
/// message's stop reason on another, and both fit the same optional shape.
#[derive(Debug, Deserialize)]
struct WireEvent {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    index: u32,
    #[serde(default)]
    message: Option<WireMessage>,
    #[serde(default)]
    content_block: Option<WireBlock>,
    #[serde(default)]
    delta: Option<WireDelta>,
    #[serde(default)]
    usage: Option<convert::WireUsage>,
    #[serde(default)]
    error: Option<WireError>,
}

#[derive(Debug, Deserialize)]
struct WireMessage {
    #[serde(default)]
    usage: Option<convert::WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireBlock {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
}

#[derive(Debug, Deserialize)]
struct WireDelta {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    partial_json: String,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireError {
    #[serde(default)]
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Finish;

    fn reader() -> Reader {
        Reader {
            frames: sse::Frames::over(Box::pin(futures_util::stream::empty())),
            done: false,
            stopped: false,
            pending: VecDeque::new(),
            blocks: BTreeMap::new(),
            usage: Usage::default(),
        }
    }

    fn events(r: &mut Reader, data: &[&str]) -> Vec<Chunk> {
        for d in data {
            r.event(d).unwrap();
        }
        r.pending.drain(..).collect()
    }

    #[test]
    fn a_text_turn_streams_its_deltas_and_ends_with_the_whole_bill() {
        let mut r = reader();
        let out = events(
            &mut r,
            &[
                r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant",
                   "content":[],"usage":{"input_tokens":100,"output_tokens":0,
                   "cache_read_input_tokens":64,"cache_creation_input_tokens":8}}}"#,
                r#"{"type":"content_block_start","index":0,
                   "content_block":{"type":"text","text":""}}"#,
                r#"{"type":"content_block_delta","index":0,
                   "delta":{"type":"text_delta","text":"Hello "}}"#,
                r#"{"type":"ping"}"#,
                r#"{"type":"content_block_delta","index":0,
                   "delta":{"type":"text_delta","text":"world"}}"#,
                r#"{"type":"content_block_stop","index":0}"#,
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},
                   "usage":{"output_tokens":8}}"#,
                r#"{"type":"message_stop"}"#,
            ],
        );
        assert!(r.stopped);

        // The input side is known from the first event.
        let first = out[0].usage.expect("message_start carries the input side");
        assert_eq!((first.input, first.output), (100, 0));
        assert_eq!((first.cached, first.cache_creation), (64, 8));

        let text: String = out.iter().map(|c| c.delta.as_str()).collect();
        assert_eq!(text, "Hello world");

        // And the finish carries BOTH halves: `message_delta` reports only the
        // output tokens, so a reader that overwrote would report input 0.
        let last = out.last().unwrap();
        assert_eq!(last.finish, Some(Finish::Stop));
        let usage = last.usage.expect("the finish carries the whole bill");
        assert_eq!((usage.input, usage.output), (100, 8));
        assert_eq!((usage.cached, usage.cache_creation), (64, 8));

        // And it is the LAST usage, not the sum of them: a caller that adds
        // the chunks up bills this turn for 200 input tokens, because the
        // number on each chunk is the running total and not an increment.
        let summed: u32 = out.iter().filter_map(|c| c.usage).map(|u| u.input).sum();
        assert_eq!(summed, 200, "the trap `Chunk::usage`'s doc names");
        assert_eq!(usage.input, 100, "the turn cost 100");
    }

    #[test]
    fn a_tool_call_assembles_from_its_fragments_and_rides_the_finish() {
        let mut r = reader();
        let out = events(
            &mut r,
            &[
                r#"{"type":"message_start","message":{"usage":{"input_tokens":42}}}"#,
                r#"{"type":"content_block_start","index":0,
                   "content_block":{"type":"text","text":""}}"#,
                r#"{"type":"content_block_delta","index":0,
                   "delta":{"type":"text_delta","text":"Looking."}}"#,
                r#"{"type":"content_block_stop","index":0}"#,
                r#"{"type":"content_block_start","index":1,
                   "content_block":{"type":"tool_use","id":"toolu_a","name":"search","input":{}}}"#,
                r#"{"type":"content_block_delta","index":1,
                   "delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}"#,
                r#"{"type":"content_block_delta","index":1,
                   "delta":{"type":"input_json_delta","partial_json":"\"acme\"}"}}"#,
                r#"{"type":"content_block_stop","index":1}"#,
                // A second call with no arguments at all: this wire sends no
                // fragments for one.
                r#"{"type":"content_block_start","index":2,
                   "content_block":{"type":"tool_use","id":"toolu_b","name":"now","input":{}}}"#,
                r#"{"type":"content_block_stop","index":2}"#,
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},
                   "usage":{"output_tokens":30}}"#,
            ],
        );

        // The fragments come out in arrival order, each carrying the identity
        // of the call it belongs to.
        let deltas: Vec<(u32, &str, &str, &str)> = out
            .iter()
            .filter_map(|c| c.tool_call_delta.as_ref())
            .map(|d| (d.index, d.id.as_str(), d.name.as_str(), d.delta.as_str()))
            .collect();
        assert_eq!(
            deltas,
            vec![
                // 0, though it is the SECOND content block: the index is
                // which of the parallel calls, as on the other wire.
                (0, "toolu_a", "search", r#"{"query":"#),
                (0, "toolu_a", "search", r#""acme"}"#),
            ]
        );

        // Nothing is emitted at content_block_stop: the assembled calls ride
        // the finish, as they do on the other wire.
        let last = out.last().unwrap();
        assert_eq!(last.finish, Some(Finish::ToolCalls));
        assert_eq!(
            last.tool_calls,
            vec![
                ToolCall {
                    id: "toolu_a".into(),
                    name: "search".into(),
                    arguments: r#"{"query":"acme"}"#.into(),
                },
                ToolCall {
                    id: "toolu_b".into(),
                    name: "now".into(),
                    // Not an empty string: `{}` is what it was called with.
                    arguments: "{}".into(),
                },
            ],
            "assembled in block order, complete"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&last.tool_calls[0].arguments).unwrap(),
            serde_json::json!({"query": "acme"}),
            "and therefore parseable, which is the point"
        );
        let usage = last.usage.unwrap();
        assert_eq!((usage.input, usage.output), (42, 30));
    }

    #[test]
    fn an_error_event_ends_the_turn_with_the_reason_and_no_invented_status() {
        let mut r = reader();
        match r
            .event(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#)
        {
            Err(Error::Stream { message }) => assert_eq!(message, "Overloaded"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_body_that_ends_mid_tool_call_is_refused_rather_than_handed_on() {
        let mut r = reader();
        events(
            &mut r,
            &[
                r#"{"type":"content_block_start","index":0,
                   "content_block":{"type":"tool_use","id":"toolu_a","name":"search"}}"#,
                r#"{"type":"content_block_delta","index":0,
                   "delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}"#,
            ],
        );
        match r.ended() {
            Err(Error::Stream { message }) => {
                assert!(message.contains("ended mid tool call"), "{message}");
                assert!(message.contains("search"), "{message}");
            }
            other => panic!("{other:?}"),
        }

        // A turn whose calls reached the caller on a finish has nothing
        // outstanding, and is the benign shape a body that just closes makes.
        let mut r = reader();
        events(
            &mut r,
            &[
                r#"{"type":"content_block_start","index":0,
                   "content_block":{"type":"tool_use","id":"toolu_a","name":"search"}}"#,
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            ],
        );
        assert!(r.ended().is_ok());
    }

    #[test]
    fn a_fragment_for_a_block_that_never_started_is_not_a_panic() {
        // Adversarial, and Go's own test: a delta whose index has no block.
        let mut r = reader();
        let out = events(
            &mut r,
            &[r#"{"type":"content_block_delta","index":7,
                 "delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}"#],
        );
        assert!(out.is_empty(), "{out:?}");
        assert!(r.ended().is_ok());
    }
}
