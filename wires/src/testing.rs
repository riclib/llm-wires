//! A listener both wires' pins are held against, and the ones above them.
//!
//! Behind the `test-support` feature, so nothing in a release build links it:
//! `providers`' own tests turn it on through a dev-dependency on itself, and
//! `domains` turns it on in its dev-dependencies to run an llm card
//! end-to-end — a credential, a secret body and a real socket, with no
//! step of the chain replaced by a double.
//!
//! It is thirty lines of tokio rather than a mock because what the pins are
//! about is the **bytes** — which path, which query, which headers, which
//! JSON — and a fake client can only ever agree with the code that built it.
//! One listener answers one request and closes, so each test gets its own
//! port and nothing is shared.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;

/// What the listener answers with.
pub enum Reply {
    /// A 200 and a JSON body.
    Json(&'static str),
    /// A status and a JSON body.
    Status(u16, &'static str),
    /// A status, extra response headers, and a JSON body — for what a wire
    /// reads off the head of a failure: a request id, a `retry-after`.
    Headed(u16, &'static [(&'static str, &'static str)], &'static str),
    /// A chunked `text/event-stream`, one HTTP chunk per event, then the end.
    Sse(&'static [&'static str]),
    /// The same event forever, until the client goes away.
    SseForever(&'static str),
}

/// What the listener saw.
#[derive(Clone, Debug, Default)]
pub struct Seen {
    pub method: String,
    pub path: String,
    pub query: String,
    /// Lowercased names, in the order they arrived.
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Seen {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).expect("the request body is JSON")
    }
}

/// One listener, on a port the operating system chose.
pub struct Fixture {
    addr: SocketAddr,
    seen: Arc<Mutex<Option<Seen>>>,
    /// Fired when the listener notices the client has gone.
    pub gone: Arc<Notify>,
}

impl Fixture {
    /// The endpoint an operator would type for the OpenAI wire, pointed at
    /// this listener.
    pub fn endpoint(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// The same for Anthropic, whose `/v1` belongs to the wire and not to
    /// the operator's endpoint.
    pub fn root(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// The same, with a path prefix in front — a corporate gateway.
    pub fn endpoint_at(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    pub fn request(&self) -> Seen {
        self.seen
            .lock()
            .unwrap()
            .clone()
            .expect("the listener saw a request")
    }
}

/// An endpoint on a port nothing is listening on.
///
/// Bound and dropped, so the operating system has just told us the port is
/// free and the connect that follows is refused rather than merely slow. A
/// reserved port number guessed by hand is a different pin on a box that has
/// something bound there.
pub async fn closed_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}/v1")
}

/// Bind, answer one request with `reply`, and close.
pub async fn fixture(reply: Reply) -> Fixture {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(None));
    let gone = Arc::new(Notify::new());
    let (heard, left) = (Arc::clone(&seen), Arc::clone(&gone));

    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        *heard.lock().unwrap() = Some(read_request(&mut sock).await);
        match reply {
            Reply::Json(body) => reply_with(&mut sock, 200, body).await,
            Reply::Status(code, body) => reply_with(&mut sock, code, body).await,
            Reply::Headed(code, headers, body) => {
                reply_headed(&mut sock, code, headers, body).await;
            }
            Reply::Sse(events) => {
                sock.write_all(SSE_HEAD.as_bytes()).await.unwrap();
                for e in events {
                    sock.write_all(http_chunk(e).as_bytes()).await.unwrap();
                    sock.flush().await.unwrap();
                }
                sock.write_all(b"0\r\n\r\n").await.unwrap();
                drain(&mut sock).await;
            }
            Reply::SseForever(event) => {
                sock.write_all(SSE_HEAD.as_bytes()).await.unwrap();
                // Keep pushing until the write fails, which is what a peer
                // that closed its end looks like from here.
                while sock.write_all(http_chunk(event).as_bytes()).await.is_ok()
                    && sock.flush().await.is_ok()
                {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                left.notify_one();
            }
        }
    });

    Fixture { addr, seen, gone }
}

const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\n\
     content-type: text/event-stream\r\n\
     transfer-encoding: chunked\r\n\
     connection: close\r\n\r\n";

fn http_chunk(body: &str) -> String {
    format!("{:X}\r\n{body}\r\n", body.len())
}

async fn reply_with(sock: &mut TcpStream, status: u16, body: &str) {
    reply_headed(sock, status, &[], body).await;
}

async fn reply_headed(sock: &mut TcpStream, status: u16, headers: &[(&str, &str)], body: &str) {
    let mut head = format!(
        "HTTP/1.1 {status} S\r\n\
         content-type: application/json\r\n\
         content-length: {}\r\n\
         connection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    sock.write_all(head.as_bytes()).await.unwrap();
    sock.write_all(body.as_bytes()).await.unwrap();
    sock.flush().await.unwrap();
    drain(sock).await;
}

/// Hold the connection open until the client hangs up, so the reply is not
/// cut short by the socket dropping under it.
async fn drain(sock: &mut TcpStream) {
    let mut sink = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), sock.read_to_end(&mut sink)).await;
}

async fn read_request(sock: &mut TcpStream) -> Seen {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let head_end = loop {
        if let Some(at) = find(&buf, b"\r\n\r\n") {
            break at;
        }
        let n = sock.read(&mut tmp).await.unwrap();
        assert!(n > 0, "the client closed before it sent a request");
        buf.extend_from_slice(&tmp[..n]);
    };

    let head = String::from_utf8(buf[..head_end].to_vec()).unwrap();
    let mut lines = head.split("\r\n");
    let mut start = lines.next().unwrap().split(' ');
    let method = start.next().unwrap().to_string();
    let target = start.next().unwrap();
    let (path, query) = target.split_once('?').unwrap_or((target, ""));

    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let want: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .map_or(0, |(_, v)| v.parse().unwrap());

    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < want {
        let n = sock.read(&mut tmp).await.unwrap();
        assert!(n > 0, "the client closed mid-body");
        body.extend_from_slice(&tmp[..n]);
    }

    Seen {
        method,
        path: path.to_string(),
        query: query.to_string(),
        headers,
        body: String::from_utf8(body).unwrap(),
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
