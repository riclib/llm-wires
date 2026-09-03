//! One turn over real TLS.
//!
//! Every other pin is `http://`, which leaves `llm_wires::tls` exercised only
//! in that it must not return `Err`: a wrong root store, a dropped ALPN list
//! or a swapped crypto provider would pass all of them. This one drives a
//! chat through a rustls listener holding a certificate minted here, and
//! reads back what the client offered in its ClientHello.
//!
//! **One test in the file on purpose.** `llm_wires::tls` reads the box's trust
//! store once per process, so the `SSL_CERT_FILE` this points at the test CA
//! has to be set before anything builds a client — which is only safe with
//! nothing else running beside it. A second test here would race the
//! `OnceLock`, and on the losing side would try to verify the test leaf
//! against the operator's real roots.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use llm_wires::{ChatRequest, Wire};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_rustls::TlsAcceptor;
use wire_secret::Secret;

#[tokio::test]
async fn a_turn_goes_over_tls_with_the_alpn_list_this_crate_offers() {
    // A CA and a leaf for `localhost`, both minted here. A CA rather than a
    // bare self-signed leaf because that is the shape a trust store holds,
    // and it is the shape an operator's box will hand the client.
    let ca_key = rcgen::KeyPair::generate().unwrap();
    let mut ca_params = rcgen::CertificateParams::new(Vec::new()).unwrap();
    ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    ca_params
        .distinguished_name
        .push(rcgen::DnType::CommonName, "solid providers test CA");
    let ca = ca_params.self_signed(&ca_key).unwrap();

    let leaf_key = rcgen::KeyPair::generate().unwrap();
    let leaf = rcgen::CertificateParams::new(vec!["localhost".to_string()])
        .unwrap()
        .signed_by(&leaf_key, &ca, &ca_key)
        .unwrap();

    // The client reads the box's trust store, and `rustls-native-certs`
    // honours SSL_CERT_FILE. Set before the first client is built, which is
    // why this file holds one test.
    let dir = tempfile::tempdir().unwrap();
    let roots = dir.path().join("ca.pem");
    std::fs::write(&roots, ca.pem()).unwrap();
    unsafe {
        std::env::set_var("SSL_CERT_FILE", &roots);
    }

    let heard = Arc::new(Mutex::new(None));
    let addr = listen(
        leaf.der().clone(),
        leaf_key.serialize_der(),
        Arc::clone(&heard),
    )
    .await;

    let client = llm_wires::build(
        // `localhost`, not the address: the leaf's SAN is what is being
        // verified, so a client that skipped hostname checking would pass.
        Wire::openai(format!("https://localhost:{}/v1", addr.port()), "gpt-4o"),
        Some(Secret::from("sk-test")),
    )
    .unwrap();

    let answer = client
        .chat(ChatRequest::ask("You are terse.", "Say hi"))
        .await
        .expect("the handshake and the turn both complete");
    assert_eq!(answer.message.content, "hi");

    // What the ClientHello offered, which is `tls.rs`'s list and nothing the
    // HTTP client filled in for us: reqwest passes a preconfigured rustls
    // config through untouched, so deleting that line loses h2 silently.
    let offered = heard.lock().unwrap().clone().expect("a ClientHello");
    assert_eq!(offered, vec!["h2".to_string(), "http/1.1".to_string()]);
}

/// A rustls listener that answers exactly one request.
///
/// `ring` and the default protocol versions, the same spelling the estate's
/// own listener uses; ALPN is `http/1.1` alone so the negotiated protocol is
/// the one the fixture below can read, while the client's full offer is still
/// recorded off the ClientHello.
async fn listen(
    cert: CertificateDer<'static>,
    key: Vec<u8>,
    heard: Arc<Mutex<Option<Vec<String>>>>,
) -> SocketAddr {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let signing = rustls::crypto::ring::sign::any_supported_type(&PrivateKeyDer::Pkcs8(
        PrivatePkcs8KeyDer::from(key),
    ))
    .unwrap();
    let resolver = Records {
        key: Arc::new(CertifiedKey::new(vec![cert], signing)),
        heard,
    };

    let mut config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_cert_resolver(Arc::new(resolver));
    config.alpn_protocols = vec![b"http/1.1".to_vec()];

    let acceptor = TlsAcceptor::from(Arc::new(config));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut tls = acceptor.accept(tcp).await.expect("the handshake completes");

        // Read past the head, then answer. Nothing about the request is
        // pinned here — `openai.rs` does that over cleartext.
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = tls.read(&mut tmp).await.unwrap();
            if n == 0 {
                return;
            }
            buf.extend_from_slice(&tmp[..n]);
            let head = buf
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|at| at + 4);
            if let Some(head) = head {
                let want: usize = String::from_utf8_lossy(&buf[..head])
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")?
                            .trim()
                            .parse()
                            .ok()
                    })
                    .unwrap_or(0);
                if buf.len() >= head + want {
                    break;
                }
            }
        }

        let body = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#;
        let head = format!(
            "HTTP/1.1 200 S\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        tls.write_all(head.as_bytes()).await.unwrap();
        tls.write_all(body.as_bytes()).await.unwrap();
        tls.flush().await.unwrap();
        // Hold the connection until the client has read and hung up.
        let mut sink = Vec::new();
        let _ = tokio::time::timeout(Duration::from_secs(5), tls.read_to_end(&mut sink)).await;
    });

    addr
}

/// Serves the leaf, and remembers what the ClientHello offered on the way in.
#[derive(Debug)]
struct Records {
    key: Arc<CertifiedKey>,
    heard: Arc<Mutex<Option<Vec<String>>>>,
}

impl ResolvesServerCert for Records {
    fn resolve(&self, hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        *self.heard.lock().unwrap() = Some(
            hello
                .alpn()
                .map(|it| {
                    it.map(|p| String::from_utf8_lossy(p).into_owned())
                        .collect()
                })
                .unwrap_or_default(),
        );
        Some(Arc::clone(&self.key))
    }
}
