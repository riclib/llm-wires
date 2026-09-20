# llm-wires

One `Provider` trait over the **Anthropic** and **OpenAI** HTTP shapes — tools,
streaming, and a key that cannot accidentally be printed — and beside it a
`Judge` trait over **TypeSafe**'s System One, for typed questions with
calibrated answers, and an `Embed` trait over the OpenAI embeddings shape, for
text turned into vectors.

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

## `Judge`: the TypeSafe wire

TypeSafe does not chat. A request is one `state` plus a map of typed questions,
and each answer is a distribution the model calibrated, not text we parsed. So
it is a second trait beside `Provider`, not extra methods on it — a deployment
that judges cannot chat, and one interface would force a method that always
errors. The `Wire` enum is still one, so a caller keeps one switch.

```rust
use llm_wires::{Answer, Judgement, Question, Wire};

let judge = llm_wires::build_judge(
    Wire::typesafe("https://api.typesafe.ai", "jev-latest"),
    Some(wire_secret::Secret::from("ts-…")),
)?;
let verdict = judge.judge(
    Judgement::of("Help! My payouts have been failing for 3 days.")
        .ask("is_urgent", Question::noul("Does this convey urgency?"))
        .ask("department", Question::choice("Which team should handle this?", [
            ("billing", "Payments, invoicing, refunds"),
            ("technical", "Bugs, outages, integrations"),
        ]))
        .ask("frustration", Question::score("How frustrated is the customer?",
            ["Calm", "Frustrated", "Very angry"])),
).await?;

match &verdict.answers["department"] {
    Answer::Choice { choice, confidence, .. } if *confidence > 0.8 => act(choice),
    Answer::Choice { .. } => review(),
    _ => unreachable!("a choice question gets a choice answer, or an Error::Decode"),
}
```

- **Three question kinds, three answer kinds.** `noul` (yes/no, answered as a
  probability), `choice` (labelled options, answered with the pick and the
  whole distribution), `score` (an ordered rubric, answered as a weighted
  level between the ends). Every leaf — the state, the instructions, a
  description — is a `serde_json::Value`, because the wire takes text, an
  object, an array or `null` at each of them.
- **A 200 that is not what we asked is an error.** An answer missing for a
  question, or of the wrong kind, is `Error::Decode`, not a verdict with a
  hole a caller could read as a no.
- **No retries, no thresholds.** A 429's `retry-after` is surfaced on
  `Error::Api` and not acted on; what confidence is enough to act on is the
  caller's number. The provider's request id rides the same error, on every
  wire.
- Read from the published SDK (`@typesafe-ai/sdk` 0.6.0) rather than the
  docs' prose, where the two differ.

## `Embed`: vectors

An embedding deployment does not chat either, so it is a third trait behind the
same `Wire` switch. A batch of texts goes out and one vector per text comes
back, **in the order the texts went out** — the wire obeys the `index` each row
carries rather than trusting the array's order, because the caller is about to
pair these with its own rows by position and store them in a fixed-width
column, where an answer that is nearly right is worse than no answer. A row
missing, a row twice, a row for an input that was not sent, or a width that
disagrees with the rest is an `Error::Decode` that names which. The model is
the client's, as everywhere else in this crate, and here it matters most: a
vector is only comparable with vectors from the same model, so an index belongs
to one model and a request field would let a typo mix two of them.

```rust
use llm_wires::{EmbedRequest, Wire};

let embed = llm_wires::build_embed(
    Wire::openai("https://api.openai.com/v1", "text-embedding-3-small"),
    Some(wire_secret::Secret::from("sk-…")),
)?;
let answer = embed.embed(EmbedRequest::of(["the first row", "the second"])).await?;
assert_eq!(answer.vectors.len(), 2);          // one per input, same order
assert_eq!(answer.usage.output, 0);           // embeddings bill the input side
```

`dimensions` is optional and omitted from the body when nothing asked for it,
so a gateway that has never heard of the field is not handed it; the request
names `encoding_format: "float"` because the vendor SDKs ask for base64 and
this wire decodes floats. An empty batch, an empty text in one, or a width of
zero is refused before the socket, naming the row — a server's 400 about a
batch of two thousand is not a sentence anybody can act on. Azure embeddings
are the same wire at Azure's URL, the deployment in the path.

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
