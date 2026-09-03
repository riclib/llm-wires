# llm-wires

One `Provider` trait over the **Anthropic** and **OpenAI** HTTP shapes — tools,
streaming, and a key that cannot accidentally be printed.

```rust
use llm_wires::{ChatRequest, Wire};

let client = llm_wires::build(
    Wire::anthropic("https://api.anthropic.com", "claude-haiku-4-5"),
    Some(wire_secret::Secret::from("sk-ant-…")),
)?;
let answer = client.chat(ChatRequest::ask("You are terse.", "Say hi")).await?;
println!("{}", answer.message.content);
```

Point the same code at OpenAI, or at anything that speaks its shape:

```rust
Wire::openai("https://api.openai.com/v1", "gpt-4o-mini")
Wire::openai("http://localhost:11434/v1", "llama3")   // ollama is just the OpenAI wire
```

## What it is

A wire, not a framework. It knows how to turn a `ChatRequest` into an HTTP
request for either shape and turn the answer back into a `ChatResponse` — and
nothing above that. No agent loop, no memory, no notion of what a "model card"
or a "deployment" is; those belong to whoever is calling.

- **Two shapes, one type.** `ChatRequest` carries `messages`, `tools`,
  `tool_choice`, `system`, `temperature`, `max_tokens` and
  `cache_stable_prefix`. Anthropic gets a `system` field and `tool_use` content
  blocks; OpenAI gets a leading system message and `tool_calls`. The caller
  writes it once.
- **Tools on both.** A `ToolCall` keeps its arguments as *the JSON text the
  model produced*, not a parsed `Value` — a stream assembles it from fragments,
  and a model that emits invalid JSON is a fact the caller should see rather
  than an error thrown three layers below where it can be reported.
- **Streaming with no `close`.** `chat_stream` hands back a `Stream`, so a
  caller `select!`s on it against a cancellation like any future, and
  **dropping the stream drops the response body** — which is what closes the
  connection. There is nothing to remember to call.
- **Prefix caching where it exists.** `cache_stable_prefix` marks the system
  prompt and tool definitions. Anthropic acts on it; OpenAI caches prefixes
  server-side with no flag, so on that wire it is read and deliberately
  ignored, and the request is byte-identical either way.
- **`ring`, not `aws-lc-rs`.** TLS is `rustls` with `rustls-no-provider` and a
  preconfigured config, so a binary that already ships `ring` does not end up
  linking two crypto backends.

## `wire-secret`

A key is a [`Secret`](secret/), not a `String`:

```rust
let key = wire_secret::Secret::from("sk-ant-…");
println!("{key}");        // <secret>
println!("{key:?}");      // <secret>
key.expose_str()?         // the only way out, and it is greppable
```

`Debug` and `Display` both print `<secret>`, so a key reaches a log line, a
trace field or an error message only when someone wrote `expose`. The buffer is
zeroed on drop. There is deliberately no `PartialEq` — a derived one compares
byte by byte and returns at the first difference, leaking the length of a
matching prefix through timing, in exactly the use the type invites.

It is a separate crate so that something wanting only this does not have to
link a keystore to get it.

## Testing

`--features test-support` exposes a fixture listener that stands a provider up
on a real socket, so callers can test their own code end to end against a real
HTTP round trip rather than a mock of one.

## Provenance

Both crates were extracted from the `providers` and `vault` crates of a private
codebase, where they have been in use. The extraction is byte-identical apart
from where `Secret` is imported from and two module doc-comments that referred
to the parent project. The upstream will migrate onto these published crates
rather than keeping a fork.

## License

MIT.
