//! The client's TLS, named once.

use std::sync::{Arc, OnceLock};

use rustls::RootCertStore;

use crate::{Error, Result};

/// `ring`, named rather than inherited.
///
/// It is the one C dependency the binary carries (`CLAUDE.md`), and both the
/// listener (`solid/src/serve.rs`) and the agent (`solidagent/src/tls.rs`)
/// already spell it. reqwest is taken with `rustls-no-provider` and handed
/// this config through `use_preconfigured_tls`, so `aws-lc-rs` never enters
/// the tree and no crate linked later gets to vote by installing a
/// process-wide default.
pub(crate) fn client_config() -> Result<rustls::ClientConfig> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut cfg = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|_| {
            Error::Tls("the ring provider does not have the default protocol versions".into())
        })?
        .with_root_certificates(roots()?)
        .with_no_client_auth();
    // What a provider endpoint is offered, in order — and this line is
    // load-bearing, not decoration: reqwest passes a *preconfigured* rustls
    // config through untouched, unlike its own branch, which fills the ALPN
    // list in for you. Delete it and there is no h2 at all.
    cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(cfg)
}

/// The box's trust store, read once per process.
///
/// A client is built per use — a credential rotates without a restart, so
/// nothing about a provider is cached — and reading and parsing the system
/// bundle on every `build` would put tens of milliseconds in front of every
/// call. The roots do not rotate on that timescale; the key does.
fn roots() -> Result<Arc<RootCertStore>> {
    static ROOTS: OnceLock<std::result::Result<Arc<RootCertStore>, String>> = OnceLock::new();
    ROOTS
        .get_or_init(|| {
            let found = rustls_native_certs::load_native_certs();
            let mut store = RootCertStore::empty();
            let (added, _) = store.add_parsable_certificates(found.certs);
            if added == 0 {
                let why = found
                    .errors
                    .first()
                    .map_or_else(|| "no certificates in it".to_string(), ToString::to_string);
                return Err(format!("the box has no usable trust store ({why})"));
            }
            Ok(Arc::new(store))
        })
        .clone()
        .map_err(Error::Tls)
}
