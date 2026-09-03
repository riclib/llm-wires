//! Server-sent events, framed.
//!
//! The bytes half of a streamed turn, shared by both wires: a response body in,
//! one `data:` payload at a time out. What a payload *means* is the wire's
//! business and lives in its own `stream.rs` — this file knows only where an
//! event begins and ends, which is a fact about SSE and not about a provider.
//!
//! There is no `close` here for the same reason there is none on
//! [`crate::Provider::chat_stream`]: the body hangs off this struct, so
//! dropping it is what closes the connection.

use std::pin::Pin;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};

use crate::{Error, Result};

type Body = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// A response body, read as events.
pub(crate) struct Frames {
    body: Body,
    buf: Vec<u8>,
    eof: bool,
}

impl Frames {
    pub(crate) fn new(resp: reqwest::Response) -> Frames {
        Frames::over(Box::pin(resp.bytes_stream()))
    }

    pub(crate) fn over(body: Body) -> Frames {
        Frames {
            body,
            buf: Vec::new(),
            eof: false,
        }
    }

    /// The next `data:` payload, or `None` at the end of the body.
    ///
    /// A block with no `data:` field — a comment keepalive, an `event:` line
    /// on its own — is skipped rather than delivered as an empty frame.
    pub(crate) async fn next_data(&mut self) -> Result<Option<String>> {
        loop {
            while let Some(block) = self.take_block() {
                if let Some(data) = data_field(&block) {
                    return Ok(Some(data));
                }
            }
            if self.eof {
                return Ok(None);
            }
            match self.body.next().await {
                Some(Ok(bytes)) => self.buf.extend_from_slice(&bytes),
                Some(Err(e)) => return Err(Error::from(e)),
                None => self.eof = true,
            }
        }
    }

    /// One SSE event block — everything up to a blank line. At the end of the
    /// body a trailing block with no blank line after it still counts: a
    /// provider that closes right after its last frame has not lost it.
    fn take_block(&mut self) -> Option<String> {
        if let Some((end, next)) = boundary(&self.buf) {
            let block = String::from_utf8_lossy(&self.buf[..end]);
            let block = block.strip_suffix('\r').unwrap_or(&block).to_string();
            self.buf.drain(..next);
            return Some(block);
        }
        if self.eof && !self.buf.is_empty() {
            let block = String::from_utf8_lossy(&self.buf).into_owned();
            self.buf.clear();
            return Some(block);
        }
        None
    }
}

/// `(end of block, start of what follows)` for the first blank line, in
/// either spelling — a provider may end lines with `\n` or `\r\n` and both
/// are on the wire in practice.
fn boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while let Some(nl) = buf[i..].iter().position(|b| *b == b'\n') {
        let at = i + nl;
        match buf.get(at + 1) {
            Some(b'\n') => return Some((at, at + 2)),
            Some(b'\r') if buf.get(at + 2) == Some(&b'\n') => return Some((at, at + 3)),
            _ => i = at + 1,
        }
    }
    None
}

/// The block's `data:` field. Several `data:` lines in one event join with a
/// newline, per SSE; the wires send one, but a gateway that wraps is not a
/// reason to lose the frame. `event:` and `id:` lines are ignored — neither
/// wire needs the event name, because Anthropic repeats it as the payload's
/// own `type` and OpenAI names no event types at all.
fn data_field(block: &str) -> Option<String> {
    let mut out: Option<String> = None;
    for line in block.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        match &mut out {
            Some(acc) => {
                acc.push('\n');
                acc.push_str(rest);
            }
            None => out = Some(rest.to_string()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> Frames {
        Frames::over(Box::pin(futures_util::stream::empty()))
    }

    #[test]
    fn blocks_split_on_a_blank_line_in_either_spelling() {
        let mut r = empty();
        r.buf
            .extend_from_slice(b"data: a\n\ndata: b\r\n\r\ndata: c\n");
        assert_eq!(r.take_block().as_deref(), Some("data: a"));
        assert_eq!(r.take_block().as_deref(), Some("data: b"));
        // The tail has no blank line after it, so it waits for more bytes…
        assert_eq!(r.take_block(), None);
        // …and is delivered when the body ends rather than being dropped.
        r.eof = true;
        assert_eq!(r.take_block().as_deref(), Some("data: c\n"));
        assert_eq!(r.take_block(), None);
    }

    #[test]
    fn a_data_field_survives_both_spacings_and_a_wrapped_frame() {
        assert_eq!(data_field("data: {\"a\":1}").as_deref(), Some(r#"{"a":1}"#));
        assert_eq!(data_field("data:{\"a\":1}").as_deref(), Some(r#"{"a":1}"#));
        assert_eq!(data_field(": keepalive").as_deref(), None);
        // Anthropic names its events; the name is in the payload too, so the
        // line is skipped and the frame is still whole.
        assert_eq!(
            data_field("event: message_stop\ndata: {\"type\":\"message_stop\"}").as_deref(),
            Some(r#"{"type":"message_stop"}"#)
        );
        assert_eq!(
            data_field("event: message\ndata: {\"a\":\ndata: 1}").as_deref(),
            Some("{\"a\":\n1}")
        );
    }
}
