//! A secret that does not print itself.
//!
//! Extracted verbatim from solid-rust's `vault` crate so that anything wanting
//! only this type does not have to link a keystore to get it. The behaviour is
//! unchanged; the one edit is the error type on [`Secret::expose_str`], which
//! was `vault::Error` and is now this crate's own.

use std::fmt;

use zeroize::Zeroizing;

/// The body was asked for as text and is not UTF-8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotUtf8;

impl fmt::Display for NotUtf8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("secret body is not valid UTF-8")
    }
}

impl std::error::Error for NotUtf8 {}

/// A decrypted secret body.
///
/// `Debug` and `Display` print `<secret>`, so a body reaches a log, a trace
/// field or an error message only when someone writes `expose()` to put it
/// there. The buffer is zeroed on drop.
///
/// There is deliberately no `PartialEq`. A derived one compares byte by byte
/// and returns at the first difference, so `==` would leak the length of a
/// matching prefix through timing — in exactly the use the type invites,
/// checking a supplied token against a stored one. When that day comes it
/// gets a constant-time comparison, not a derive.
#[derive(Clone)]
pub struct Secret(Zeroizing<Vec<u8>>);

impl Secret {
    pub fn new(body: Vec<u8>) -> Secret {
        Secret(Zeroizing::new(body))
    }

    /// The body. Naming the call is the point: a grep for `expose` finds every
    /// place a secret leaves the vault.
    pub fn expose(&self) -> &[u8] {
        &self.0
    }

    /// The body as text — a connection string, an API key.
    pub fn expose_str(&self) -> Result<&str, NotUtf8> {
        std::str::from_utf8(&self.0).map_err(|_| NotUtf8)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<&str> for Secret {
    fn from(s: &str) -> Secret {
        Secret::new(s.as_bytes().to_vec())
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<secret>")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<secret>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_secret_does_not_print_itself() {
        let s = Secret::from("sk-live-do-not-log-me");
        assert_eq!(format!("{s}"), "<secret>");
        assert_eq!(format!("{s:?}"), "<secret>");
        assert_eq!(format!("{:?}", Some(s.clone())), "Some(<secret>)");
        assert_eq!(s.expose_str().unwrap(), "sk-live-do-not-log-me");
    }
}
