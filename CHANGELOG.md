# Changelog

All notable changes to this project will be documented in this file. The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-29

### Added

- Runtime-neutral `GuardedStream` wrapper over `futures_core::Stream` with zero prefetch and unchanged item forwarding.
- Cloneable `Probe` and consistent `Snapshot` for observing caller contract, output boundary, termination, and an in-progress poll guard together.
- Conservative replay verdict: only known-safe contract, uncrossed output boundary, and explicit transient failure can authorize replay.
- Monotonic output tracking for text, refusal, reasoning, audio, tool calls, images, and other semantic output.
- Distinct terminal states for clean completion, typed failure, upstream EOF without a terminal marker, consumer drop, and panic during upstream polling or classification.
- Fail-closed panic handling that prevents stale transient-failure evidence from authorizing replay during unwinding.
- Optional `openai-json` feature with a bounded classifier for already-framed OpenAI-compatible Chat Completions and Responses SSE events.
- Compiled `eventsource-stream` integration coverage for partial-output EOF, explicit `[DONE]`, and transport failure.
- Fail-closed handling for malformed or oversized JSON, duplicate object keys, conflicting event signals, and unknown semantic event shapes.
- CI coverage for stable Rust and MSRV 1.75.0, documentation, publishable package contents, and dependency-policy checks.

### Scope limits

- No HTTP client, SSE framing parser, network I/O, prefetch, sleep, logging, automatic retry, tool execution, or idempotency mechanism.
- The OpenAI-compatible classifier stores no payload after classification, but its limits do not bound memory already used by an upstream parser or downstream consumer.
