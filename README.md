# llm-stream-guard

[![CI](https://github.com/airouter-dev/llm-stream-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/airouter-dev/llm-stream-guard/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/llm-stream-guard.svg)](https://crates.io/crates/llm-stream-guard)
[![docs.rs](https://docs.rs/llm-stream-guard/badge.svg)](https://docs.rs/llm-stream-guard)

Conservative replay-boundary tracking for streamed LLM responses in Rust.

`llm-stream-guard` wraps an existing `futures_core::Stream`, classifies each item before forwarding that same item, and exposes a small snapshot that answers one narrow question: has enough evidence been collected to authorize whole-request replay?

It is runtime-neutral and performs zero prefetching. It does **not** open a connection, parse an SSE byte stream, sleep, log, buffer a response, or retry a request. The caller continues to own transport behavior, idempotency, retry budgets, and side-effect safety.

## Why this boundary exists

A transport failure can happen after an HTTP 200 and after output has reached a consumer. Retrying the complete generation at that point can duplicate text, billing, or tool effects. Conversely, seeing no output does not prove that the server never accepted the request.

This failure class appears in public Rust project reports:

- [`nearai/cloud-api#889`](https://github.com/nearai/cloud-api/issues/889) documents mid-stream transport cuts that can leave partial output, omit an actionable terminal signal, or end without `[DONE]`.
- [`adelie-ai/desktop-assistant#799`](https://github.com/adelie-ai/desktop-assistant/issues/799) documents provider error events being ignored and a truncated response being returned as success.

Those reports are demand evidence, not adoption claims. Neither project is affiliated with or known to use this crate.

## Install

```toml
[dependencies]
llm-stream-guard = "0.1"
```

The default feature set contains only the generic stream guard. Enable the bounded OpenAI-compatible JSON classifier explicitly when needed:

```toml
[dependencies]
llm-stream-guard = { version = "0.1", features = ["openai-json"] }
```

The minimum supported Rust version (MSRV) is 1.75.0. The crate requires `std`; “runtime-neutral” means that it is not tied to Tokio, async-std, smol, an HTTP client, or an SSE parser.

## The only allowing decision

`Snapshot::verdict()` fails closed. It returns `ReplayVerdict::Allow` for exactly one combination:

| Caller contract | Observed output boundary | Terminal evidence | Poll/classification in progress | Verdict |
| --- | --- | --- | --- | --- |
| `KnownSafe` | `Uncrossed` | `Failed(Transient)` | No | `Allow` |
| Any other combination | Any | Any | Any | `Deny` |

`KnownSafe` is an assertion supplied by your application from the request contract. It cannot be inferred from an idempotency-key header, an empty stream, or this crate's state. If that assertion would be a guess, use `ReplayContract::Unknown`.

The verdict is an authorization signal, not a retry loop. Even an `Allow` result still needs an application-owned attempt limit, elapsed-time budget, cancellation policy, and backoff policy.

## Wrap an existing stream

The generic API accepts any item type and a stateful classifier. The example uses a made-up application event; substitute the type already emitted by your transport or parser.

The example's `.next()` helper comes from `futures-util`; the guard itself depends only on the `futures-core` trait and does not require that extension crate.

```rust,ignore
use futures_util::StreamExt;
use llm_stream_guard::{
    FailureKind, GuardedStream, Observation, OutputKind, ReplayContract,
    ReplayVerdict,
};

enum AppEvent {
    Metadata,
    Text(String),
    ToolArguments(String),
    Complete,
    TransportFailure { transient: bool },
}

let guarded = GuardedStream::new(
    upstream,
    |event: &AppEvent| match event {
        AppEvent::Metadata => Observation::Neutral,
        AppEvent::Text(_) => Observation::Output(OutputKind::Text),
        // A partial JSON fragment is already output. Do not wait for valid JSON.
        AppEvent::ToolArguments(_) => Observation::Output(OutputKind::ToolCall),
        AppEvent::Complete => Observation::Completed,
        AppEvent::TransportFailure { transient: true } => {
            Observation::Failed(FailureKind::Transient)
        }
        AppEvent::TransportFailure { transient: false } => {
            Observation::Failed(FailureKind::Permanent)
        }
    },
    ReplayContract::KnownSafe,
);

let probe = guarded.probe();
let mut guarded = Box::pin(guarded);

while let Some(event) = guarded.next().await {
    consume(event).await;
}

if probe.snapshot().verdict() == ReplayVerdict::Allow {
    // Your application may schedule a budgeted replay. This crate never does it.
}
```

Classification is committed before the item becomes visible downstream. Each downstream `poll_next` polls the upstream stream at most once; after upstream returns `None`, the wrapper is fused and does not poll it again.

The repository includes a compiled [`eventsource-stream` integration
test](tests/eventsource_stream.rs). It passes already-framed `Event` values to
the optional classifier and covers partial-output EOF, explicit `[DONE]`, and
transport-error paths without making an SSE parser a normal dependency.

## State transitions

Every item maps to one `Observation`:

| Observation | Boundary effect | Terminal effect |
| --- | --- | --- |
| `Neutral` | No change | No change |
| `Output(kind)` | Crosses the boundary and retains the first known `OutputKind` | No change |
| `Uncertain` | Changes `Uncrossed` to `Uncertain`; never erases a crossed boundary | No change |
| `Completed` | No change | Records explicit clean completion |
| `Failed(kind)` | No change | Records failure and can override an earlier completion |

The output boundary is monotonic. Failure evidence also only becomes stricter: `Unknown` outranks all other failure kinds, and `Permanent` outranks `Transient`. A completed stream never authorizes replay.

If the wrapper is dropped while active, termination becomes `Exited(Dropped)`. If upstream returns `None` before an explicit terminal observation, it becomes `Exited(EndOfStream)`. If upstream polling or classification panics, a snapshot taken during unwinding denies replay and dropping the wrapper records `Exited(Panicked)`. All three exit paths deny replay.

## Classify already-framed OpenAI-compatible events

The `openai-json` feature enables a convenience module that classifies one complete, already-framed SSE event. It supports common Chat Completions and Responses event shapes without taking ownership of event data.

```rust
use llm_stream_guard::openai::{
    classify_openai_event, ClassifierLimits, SseEventRef,
};
use llm_stream_guard::{Observation, OutputKind};

let event = SseEventRef::named(
    "response.function_call_arguments.delta",
    br#"{"type":"response.function_call_arguments.delta","delta":"{\"city\":"}"#,
);

assert_eq!(
    classify_openai_event(event, ClassifierLimits::default()),
    Observation::Output(OutputKind::ToolCall),
);
```

`SseEventRef::data` must already contain the joined `data:` fields for one event. The module does not split chunks, decode an EventSource stream, join multi-line fields, reconnect, or protect an upstream parser from buffering too much data.

### Built-in classification

| Evidence | Observation |
| --- | --- |
| Chat Completions `content`, `refusal`, reasoning, audio, `tool_calls`, or `function_call` data | The matching `OutputKind` |
| Responses text, refusal, reasoning, audio, function/tool-call, or partial-image event | The matching `OutputKind` |
| `[DONE]` or `response.completed` | `Completed` |
| `error`, `response.failed`, or `response.incomplete` | `Failed(Unknown)` |
| Oversized input, malformed JSON, duplicate object keys, conflicting signals, or an unknown semantic shape | `Uncertain` |
| Known metadata with no output | `Neutral` |

The built-in classifier deliberately maps in-stream failures to `FailureKind::Unknown`. A response event can prove that a failure occurred, but usually cannot prove that replaying the complete request is transient-safe. Consequently, the built-in classifier alone does not turn an in-stream provider error into `ReplayVerdict::Allow`.

### Missing `[DONE]`

For Chat Completions, `[DONE]` is explicit completion evidence. If it never arrives and the parsed stream simply ends, the wrapper records `Exited(EndOfStream)`, not `Completed`. For Responses streams, `response.completed` is also explicit completion evidence.

This distinction prevents silent EOF from being treated as success. It does not detect a stalled socket; read deadlines remain the transport's responsibility.

### Partial tool-call arguments

The first non-empty tool-call or function-call fragment crosses the boundary immediately. The fragment does not need to be valid JSON. Retrying after `{"city":` can duplicate a call, repeat billing, or cause a downstream assembler to combine fragments from different attempts.

This crate tracks the boundary; it does not execute tools, deduplicate calls, validate argument schemas, or make side effects idempotent.

## Resource and privacy boundaries

`GuardedStream` retains the replay contract, first output kind, terminal state, and an internal poll-in-progress flag behind an `Arc<Mutex<_>>`. It forwards each original item unchanged and does not store response bodies.

The OpenAI-compatible classifier temporarily parses one event as JSON. Defaults and hard ceilings are:

| Input | Default | Hard ceiling |
| --- | ---: | ---: |
| Event name | 256 bytes | 4 KiB |
| Joined `data` payload | 64 KiB | 1 MiB |
| Chat Completions choices inspected | 128 | 1,024 |

Caller-supplied values are clamped to the hard ceilings. Inputs over the effective limit become `Observation::Uncertain`.

These bounds apply only after your code has framed an event. They do not bound socket buffers, HTTP bodies, an external SSE parser, downstream consumers, or the original event item. The crate does not provide complete memory protection or data-loss prevention. Avoid logging prompts, output, tool arguments, or raw provider errors.

## When to use it

Use this crate when:

- an application already has a Rust `Stream` and needs one conservative replay boundary across transports or providers;
- items may contain text, refusal, reasoning, audio, image, or partial tool-call output;
- clean completion, provider failure, EOF, and consumer drop must remain distinct;
- another task needs a consistent snapshot without retaining payloads.

Do not use it when:

- you need an SSE parser, EventSource reconnection, an HTTP client, or a retry/backoff implementation;
- your SDK already gives your application a tested equivalent state machine;
- you need token-level resume or continuation rather than whole-request replay;
- you cannot define the request's independent replay contract;
- you expect the crate to make arbitrary tool or upstream side effects safe.

## Public API

Core types:

```text
Classify<Item>
GuardedStream<S, C>
Probe
Snapshot
Observation
OutputKind
FailureKind
ReplayBoundary
Termination
ExitReason
ReplayContract
ReplayVerdict
```

OpenAI-compatible event helpers (`openai-json` feature):

```text
openai::SseEventRef
openai::ClassifierLimits
openai::OpenAiClassifier
openai::classify_openai_event
```

See [docs.rs](https://docs.rs/llm-stream-guard) for method-level documentation.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo test --all-targets --no-default-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
cargo package --locked
```

CI runs the test suite on Linux, macOS, and Windows, verifies MSRV 1.75.0,
checks the no-default-features build, validates documentation, creates the
publishable package, and audits advisories, licenses, duplicate versions,
wildcard requirements, and dependency sources.

## Security

Report replay-boundary under-classification, resource-limit bypasses, synchronization defects, or release-workflow concerns through [GitHub private vulnerability reporting](https://github.com/airouter-dev/llm-stream-guard/security/advisories/new). Use only synthetic fixtures; do not submit real prompts, model output, credentials, tool arguments, or customer data.

## License and naming

Licensed under either Apache-2.0 or MIT, at your option.

This provider-neutral project is maintained by [airouter.dev](https://ai-router.dev/) contributors. It is independent and is not affiliated with or endorsed by OpenAI. “OpenAI” is used only to describe compatible event shapes.
