# Security Policy

## Supported versions

Before the first crates.io release, security work targets the current `main` branch. After publication, the latest `0.1.x` patch line will receive fixes until a newer minor line replaces it.

## Report privately

Use [GitHub private vulnerability reporting](https://github.com/airouter-dev/llm-stream-guard/security/advisories/new) for suspected vulnerabilities. If private reporting is unavailable, open a public issue asking maintainers to enable a private channel; do not include exploit details or sensitive data.

Useful reports include the affected revision or version, Rust version, a minimal synthetic reproducer, observed state, expected state, and why the difference can change a replay decision.

Please report issues such as:

- semantic output classified as `Neutral` or left `Uncrossed`;
- a state combination other than known-safe + uncrossed + transient failure producing `Allow`;
- malformed, ambiguous, or oversized input bypassing fail-closed classification;
- resource use exceeding the documented classifier ceilings;
- inconsistent snapshots, panic/unwind handling, or synchronization defects;
- unsafe release-workflow, package-content, or dependency-source behavior.

Never submit real API keys, authorization headers, prompts, model output, tool arguments, provider error bodies, customer records, or production traces. Replace them with synthetic canaries and the smallest invented fixture that reproduces the issue.

## Security boundaries

`llm-stream-guard` records semantic replay evidence. It does not provide transport security, authenticate providers, parse a raw SSE byte stream, enforce read deadlines, prevent an upstream parser from buffering too much data, execute or deduplicate tools, anonymize arbitrary payloads, or perform a retry.

The OpenAI-compatible classifier temporarily parses one bounded event and retains no event payload after returning. This is not a promise that the caller, upstream parser, allocator, logger, or downstream consumer retains no data. Treat prompts, output, tool arguments, and raw errors as sensitive before they reach this crate.

An `Allow` verdict is only one input to an application-owned retry policy. A snapshot taken while upstream polling or classification is in progress denies replay; this includes the unwind window after a panic. Attempt limits, elapsed-time budgets, cancellation, backoff, billing implications, and side-effect safety remain the application's responsibility.
