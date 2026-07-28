# Contributing

Contributions should make the replay boundary more accurate, make ambiguous evidence fail closed, or add a synthetic regression fixture for a real stream failure class. New provider-name lists, keyword-only changes, and unverified compatibility claims are not useful by themselves.

## Development setup

Install stable Rust and Rust 1.75.0. The stable toolchain runs formatting, Clippy, documentation, and packaging checks; 1.75.0 verifies the declared MSRV.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo test --all-targets --no-default-features --locked
cargo test --doc --all-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
cargo package --locked
```

If `cargo-deny` is installed, also run:

```bash
cargo deny check --all-features
```

## Invariants to preserve

- `GuardedStream` performs zero prefetch: one downstream poll causes at most one upstream poll.
- Classification is committed before the same item is returned downstream.
- Items are forwarded unchanged and payloads are not retained by the guard state.
- `ReplayBoundary` is monotonic; uncertain or crossed evidence must never return to `Uncrossed`.
- Only `KnownSafe` + `Uncrossed` + `Failed(Transient)` may produce `ReplayVerdict::Allow`.
- A snapshot taken while upstream polling or classification is in progress must deny replay; panic unwinding must not expose stale `Allow` evidence.
- Seeing no output is not proof that the request or an upstream side effect is replay-safe.
- EOF, consumer drop, and panic without explicit semantic completion remain distinct from clean completion.
- A partial tool/function-call argument crosses the boundary before it becomes valid JSON.
- Unknown, malformed, conflicting, duplicated-key, or over-limit OpenAI-compatible events fail closed.
- Built-in in-stream provider failures remain `FailureKind::Unknown` unless independent, typed evidence establishes something stronger.
- Runtime neutrality must not be replaced by an implicit Tokio, HTTP-client, or EventSource dependency.

Changes to a public enum or verdict rule need focused tests and a short compatibility note. Because public enums may be non-exhaustive, additions can still change downstream behavior and should be justified by a concrete semantic shape.

## Fixtures and compatibility claims

Use invented text, fake request IDs, synthetic credentials, and minimal JSON. Do not copy production headers, prompts, completions, error messages, tool arguments, traces, or customer data into source, issues, or pull requests.

A compatibility claim must identify the exact event shape and how it was verified. A public issue can motivate a regression test, but do not imply that the reporting project uses or endorses this crate.

## Dependencies and MSRV

Keep dependencies small and registry-sourced. New dependencies require a clear risk reduction or substantial maintenance benefit, compatible licensing, `cargo-deny` approval, and Rust 1.75.0 support. Do not add wildcard version requirements or git dependencies for convenience.

If raising MSRV is necessary, document the reason in the changelog and make the change in a minor release, not an unannounced patch.

## Pull requests

Describe the replay boundary or failure class being changed, the new evidence, and the exact commands run. Include tests for the safe path, the ambiguous path, and the failure path where applicable. Keep release credentials and registry publishing out of contributor pull requests.

Unless explicitly stated otherwise, contributions intentionally submitted for inclusion are licensed under either Apache-2.0 or MIT, at the contributor's option, without additional terms.
