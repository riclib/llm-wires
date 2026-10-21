//! An image, as every wire that takes one takes it: embedded, never fetched.
//!
//! One type for Clairic's `images`, TypeSafe's `assets` and a chat turn's
//! image parts, because a caller that holds an upload should not have to
//! know which wire is about to read it. What differs is spelled at each
//! wire: Clairic takes the data URL or the two halves, TypeSafe raw base64
//! and a format, Anthropic a `base64` source with its media type, OpenAI a
//! data URL.
//!
//! The base64 here is hand-rolled, as is the header reading below it: a
//! wire that must refuse an image TypeSafe would refuse needs its width and
//! height, and that is thirty lines of PNG and JPEG headers rather than an
//! image library in every binary that links this crate.

use std::borrow::Cow;

/// An image, embedded.
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

impl Image {
    /// The file's bytes, encoded here. `content_type` is the media type,
    /// `image/png`; TypeSafe reads the format from the bytes, and the chat
    /// wires send it as given.
    pub fn from_bytes(content_type: impl Into<String>, bytes: &[u8]) -> Image {
        Image::Base64 {
            content_type: content_type.into(),
            base64: encode(bytes),
        }
    }

    /// The media type and the base64, when the image is base64 at all: a
    /// data URL that is not (`data:image/svg+xml,<svg…`) is `None`.
    pub(crate) fn base64_parts(&self) -> Option<(&str, &str)> {
        match self {
            Image::Base64 {
                content_type,
                base64,
            } => Some((content_type, base64)),
            Image::DataUrl(url) => {
                let rest = url
                    .get(..5)
                    .filter(|s| s.eq_ignore_ascii_case("data:"))
                    .map(|_| &url[5..])?;
                let (meta, data) = rest.split_once(',')?;
                let media = meta
                    .strip_suffix(";base64")
                    .or_else(|| meta.strip_suffix(";BASE64"))?;
                Some((media, data))
            }
        }
    }

    /// The whole thing as a data URL, which is how OpenAI's wire wants it.
    pub(crate) fn data_url(&self) -> Cow<'_, str> {
        match self {
            Image::DataUrl(url) => Cow::Borrowed(url),
            Image::Base64 {
                content_type,
                base64,
            } => Cow::Owned(format!("data:{content_type};base64,{base64}")),
        }
    }
}

/// What the bytes are, read from them and not from a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    Png,
    Jpeg,
}

/// The format and the width and height, from the file's own header. `None`
/// when the bytes are neither a PNG nor a JPEG with a frame header, which is
/// what a wire that only takes those two refuses.
pub(crate) fn dimensions(bytes: &[u8]) -> Option<(Format, u32, u32)> {
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";
    if bytes.starts_with(PNG) {
        // The first chunk is IHDR: width and height, big-endian, at 16 and 20.
        if bytes.get(12..16)? != b"IHDR" {
            return None;
        }
        let be = |at: usize| Some(u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
        return Some((Format::Png, be(16)?, be(20)?));
    }
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    // A JPEG is segments, each `FF marker` and, for most, a two-byte length
    // that counts itself. The first start-of-frame holds the size: precision,
    // then height, then width.
    let mut at = 2;
    loop {
        while *bytes.get(at)? != 0xFF {
            at += 1;
        }
        while *bytes.get(at)? == 0xFF {
            at += 1;
        }
        let marker = *bytes.get(at)?;
        at += 1;
        match marker {
            0xD0..=0xD9 | 0x01 => continue,
            0xC0..=0xCF if !matches!(marker, 0xC4 | 0xC8 | 0xCC) => {
                let be = |i: usize| -> Option<u32> {
                    Some(u16::from_be_bytes(bytes.get(i..i + 2)?.try_into().ok()?).into())
                };
                return Some((Format::Jpeg, be(at + 5)?, be(at + 3)?));
            }
            _ => {
                let len = usize::from(u16::from_be_bytes(bytes.get(at..at + 2)?.try_into().ok()?));
                at += len;
            }
        }
    }
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with `=` padding and no line breaks — what every wire
/// here documents.
pub(crate) fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, b)| n | u32::from(*b) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The strict inverse of [`encode`]: the standard alphabet, padded, no
/// whitespace. Anything else is `None`, because it is what TypeSafe refuses.
pub(crate) fn decode(text: &str) -> Option<Vec<u8>> {
    let text = text.as_bytes();
    if text.len() % 4 != 0 {
        return None;
    }
    let value = |c: u8| ALPHABET.iter().position(|a| *a == c).map(|v| v as u32);
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let chunks = text.len() / 4;
    for (k, chunk) in text.chunks(4).enumerate() {
        let pad = chunk.iter().rev().take_while(|c| **c == b'=').count();
        if pad > 2 || (pad > 0 && k + 1 != chunks) {
            return None;
        }
        let mut n = 0u32;
        for (i, c) in chunk[..4 - pad].iter().enumerate() {
            n |= value(*c)? << (18 - 6 * i);
        }
        for i in 0..3 - pad {
            out.push((n >> (16 - 8 * i)) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A PNG's signature and IHDR, which is all [`dimensions`] reads.
    pub(crate) fn png_header(width: u32, height: u32) -> Vec<u8> {
        let mut b = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        b.extend(width.to_be_bytes());
        b.extend(height.to_be_bytes());
        b.extend([8, 2, 0, 0, 0]);
        b
    }

    #[test]
    fn base64_round_trips_and_refuses_what_typesafe_refuses() {
        for bytes in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"] {
            assert_eq!(decode(&encode(bytes)).as_deref(), Some(bytes));
        }
        assert_eq!(encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(encode(b"fo"), "Zm8=");
        let all: Vec<u8> = (0..=255).collect();
        assert_eq!(decode(&encode(&all)), Some(all));
        for bad in ["Zm8", "Zm8=\n", "Zm 8=", "Z===", "Zm8=Zm8=", "Zm-_"] {
            assert_eq!(decode(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_png_and_a_jpeg_give_up_their_size() {
        assert_eq!(
            dimensions(&png_header(300, 281)),
            Some((Format::Png, 300, 281))
        );
        // SOI, an APP0 segment to skip, a DHT that is not a frame, then SOF0:
        // precision 8, height 0x0118 = 280, width 0x0640 = 1600.
        let jpeg = [
            &[0xFF, 0xD8][..],
            &[0xFF, 0xE0, 0x00, 0x04, 0xAA, 0xBB],
            &[0xFF, 0xC4, 0x00, 0x03, 0x00],
            &[0xFF, 0xC0, 0x00, 0x11, 0x08, 0x01, 0x18, 0x06, 0x40, 0x03],
        ]
        .concat();
        assert_eq!(dimensions(&jpeg), Some((Format::Jpeg, 1600, 280)));
        // A GIF, a truncated PNG, a JPEG with no frame.
        assert_eq!(dimensions(b"GIF89a\x01\x00\x01\x00"), None);
        assert_eq!(dimensions(&png_header(1, 1)[..18]), None);
        assert_eq!(dimensions(&[0xFF, 0xD8, 0xFF, 0xD9]), None);
    }

    #[test]
    fn a_data_url_splits_into_its_media_type_and_base64() {
        let url = Image::DataUrl("data:image/png;base64,AAAA".into());
        assert_eq!(url.base64_parts(), Some(("image/png", "AAAA")));
        assert_eq!(url.data_url(), "data:image/png;base64,AAAA");
        let upper = Image::DataUrl("DATA:image/jpeg;base64,BBBB".into());
        assert_eq!(upper.base64_parts(), Some(("image/jpeg", "BBBB")));
        let svg = Image::DataUrl("data:image/svg+xml,<svg/>".into());
        assert_eq!(svg.base64_parts(), None);
        let raw = Image::from_bytes("image/png", b"foo");
        assert_eq!(raw.base64_parts(), Some(("image/png", "Zm9v")));
        assert_eq!(raw.data_url(), "data:image/png;base64,Zm9v");
    }
}
