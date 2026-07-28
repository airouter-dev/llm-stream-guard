# Releasing

Published versions are immutable. Prepare and review the complete crate before
creating a tag; do not publish placeholder versions to reserve a name.

## Release checklist

1. Update `CHANGELOG.md` and remove the `Unreleased` marker for the target
   version and date.
2. Confirm `Cargo.toml`, the changelog, and the intended `vX.Y.Z` tag agree.
3. Run the commands in `CONTRIBUTING.md`, including both feature sets and
   `cargo package --locked --list`.
4. Inspect the generated `.crate`; it must not contain credentials, `.env`
   files, production traces, prompts, responses, tool arguments, or marketing
   working files.
5. Compile a clean downstream fixture from the generated `.crate`.
6. Push the reviewed commit and wait for every required CI job on `main`.
7. Create the immutable tag and GitHub release from that exact commit.

## First crates.io release

crates.io Trusted Publishing requires the crate to exist first. Version 0.1.0
therefore uses a bootstrap API token restricted to `publish-new`, the exact
crate name, and the shortest practical expiry. Inject it only through the
current process environment as `CARGO_REGISTRY_TOKEN`; do not run `cargo login`,
put it in a file, add it to a workflow, or send it through chat.

Re-run `cargo publish --locked --dry-run`, publish once, unset the environment
variable, and revoke the bootstrap token immediately. Then download the
registry artifact, compare its checksum and contents, compile a fresh registry
consumer, and verify the docs.rs build.

## Later releases

After 0.1.0 exists, configure the exact repository, `release.yml`, and
`crates-io` environment as a crates.io Trusted Publisher. Later releases use
the manually dispatched release workflow, which checks the existing tag,
version, main-branch ancestry, absent registry version, tests, documentation,
and dry run before requesting a short-lived OIDC token.

Never add a long-lived token fallback to `release.yml`. Do not publish a patch
version solely to test authentication; use the next real, reviewed change.
