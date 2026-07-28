//! Runtime-neutral replay boundaries for streamed LLM responses.
//!
//! This crate observes items as they pass through a [`GuardedStream`]. It does
//! not perform I/O, parse a transport, retain response bodies, or retry a
//! request. A retry is authorized only when the caller declares a
//! [`ReplayContract::KnownSafe`], no output boundary was crossed, and the
//! classifier observed an explicit transient failure.
//!
//! # Example
//!
//! The guard observes an item before returning that same item. A transient
//! failure after text has already appeared therefore cannot authorize replay:
//!
//! ```
//! use futures_util::{FutureExt, StreamExt};
//! use llm_stream_guard::{
//!     FailureKind, GuardedStream, Observation, OutputKind, ReplayBoundary,
//!     ReplayContract, ReplayVerdict,
//! };
//!
//! #[derive(Clone, Copy, Debug, Eq, PartialEq)]
//! enum AppEvent {
//!     Text,
//!     TransientFailure,
//! }
//!
//! let upstream = futures_util::stream::iter([
//!     AppEvent::Text,
//!     AppEvent::TransientFailure,
//! ]);
//! let guarded = GuardedStream::new(
//!     upstream,
//!     |event: &AppEvent| match event {
//!         AppEvent::Text => Observation::Output(OutputKind::Text),
//!         AppEvent::TransientFailure => {
//!             Observation::Failed(FailureKind::Transient)
//!         }
//!     },
//!     ReplayContract::KnownSafe,
//! );
//! let probe = guarded.probe();
//! futures_util::pin_mut!(guarded);
//!
//! assert_eq!(guarded.next().now_or_never(), Some(Some(AppEvent::Text)));
//! assert_eq!(
//!     probe.snapshot().boundary(),
//!     ReplayBoundary::Crossed(OutputKind::Text),
//! );
//! assert_eq!(
//!     guarded.next().now_or_never(),
//!     Some(Some(AppEvent::TransientFailure)),
//! );
//! assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Deny);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod core;
#[cfg(feature = "openai-json")]
pub mod openai;

pub use crate::core::{
    Classify, ExitReason, FailureKind, GuardedStream, Observation, OutputKind, Probe,
    ReplayBoundary, ReplayContract, ReplayVerdict, Snapshot, Termination,
};
