//! What a caller hands an [`Embed`](crate::Embed) client and what it gets
//! back.
//!
//! Batched, because every wire that embeds takes a batch and one round trip
//! per row is the difference between a corpus that indexes in a minute and one
//! that indexes in an hour. One vector per input, **in the order the inputs
//! went out**: a caller pairs the vectors with its own rows by position, so
//! the wire puts the provider's answer back in order rather than leaving every
//! caller to do it — and a provider that answers a row twice, or one it was
//! not asked for, is [`Error::Decode`](crate::Error::Decode) rather than a
//! vector silently against the wrong text.
//!
//! There is no `model` here, for the reason there is none on
//! [`ChatRequest`](crate::ChatRequest) or [`Judgement`](crate::Judgement): the
//! model is part of the [`Wire`](crate::Wire) a client was built from. It
//! matters more here than anywhere else — a vector is only comparable with
//! vectors from the same model, so a stored corpus is one model's, and a
//! request field would let a caller mix two widths into one index by typo.

use crate::Usage;

/// A batch to embed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbedRequest {
    /// The texts, in the order their vectors come back in. An empty batch, or
    /// an empty text in it, is refused before the socket: the server refuses
    /// both, and its 400 says less than the index of the row that is blank.
    pub inputs: Vec<String>,
    /// A shorter vector than the model's natural width, where the model
    /// supports one (OpenAI's `text-embedding-3-*` are trained so a prefix is
    /// still a usable vector). `None` omits the field entirely, so a gateway
    /// that has never heard of it is not handed it, and the model's own width
    /// is what comes back.
    pub dimensions: Option<u32>,
}

impl EmbedRequest {
    /// A batch, in order.
    pub fn of<I, S>(inputs: I) -> EmbedRequest
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        EmbedRequest {
            inputs: inputs.into_iter().map(Into::into).collect(),
            dimensions: None,
        }
    }

    /// One text — a query, usually, where the corpus went through
    /// [`EmbedRequest::of`].
    pub fn one(text: impl Into<String>) -> EmbedRequest {
        EmbedRequest::of([text])
    }

    /// Ask for this width. A query and its corpus must ask for the same one.
    pub fn with_dimensions(mut self, dimensions: u32) -> EmbedRequest {
        self.dimensions = Some(dimensions);
        self
    }
}

/// The vectors, in the order the inputs went out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbedResponse {
    /// One per input, same order, all of one width.
    pub vectors: Vec<Vec<f32>>,
    /// What the batch cost, where the wire reported it. Embeddings bill the
    /// input side only: `output`, `cached`, `cache_creation` and `reasoning`
    /// stay 0, and a sum over calls is still a sum of what was billed.
    pub usage: Usage,
}

impl EmbedResponse {
    /// The width of the vectors, or `None` for an answer with none in it.
    ///
    /// The first speaks for all: a ragged answer never gets this far — the
    /// wire refuses it — because what stores these is fixed-width, and a row
    /// of the wrong width fails at the writer with nothing left saying which
    /// call produced it.
    pub fn dimensions(&self) -> Option<usize> {
        self.vectors.first().map(Vec::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_keeps_the_order_it_was_written_in() {
        let req = EmbedRequest::of(["one", "two", "three"]).with_dimensions(256);
        assert_eq!(req.inputs, ["one", "two", "three"]);
        assert_eq!(req.dimensions, Some(256));
        // A query is a batch of one, so the corpus and the question go out
        // through the same door.
        assert_eq!(EmbedRequest::one("q").inputs, ["q"]);
        assert_eq!(EmbedRequest::one("q").dimensions, None);
    }

    #[test]
    fn the_width_is_the_first_vectors() {
        assert_eq!(EmbedResponse::default().dimensions(), None);
        let resp = EmbedResponse {
            vectors: vec![vec![0.0; 3], vec![1.0; 3]],
            usage: Usage::default(),
        };
        assert_eq!(resp.dimensions(), Some(3));
    }
}
