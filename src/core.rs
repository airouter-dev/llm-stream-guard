use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use futures_core::stream::{FusedStream, Stream};
use pin_project_lite::pin_project;

/// Classifies one item emitted by an upstream stream.
///
/// Implementations may be stateful. Classification happens after the item is
/// produced by the upstream stream and before that same item is returned to the
/// downstream consumer.
pub trait Classify<Item> {
    /// Describes the replay-relevant meaning of `item`.
    fn classify(&mut self, item: &Item) -> Observation;
}

impl<Item, F> Classify<Item> for F
where
    F: FnMut(&Item) -> Observation,
{
    fn classify(&mut self, item: &Item) -> Observation {
        self(item)
    }
}

/// A semantic kind of model output that makes whole-request replay unsafe.
///
/// Values carry no response text, arguments, audio, or other body data.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum OutputKind {
    /// User-visible generated text.
    Text,
    /// A model refusal.
    Refusal,
    /// Reasoning or analysis output.
    Reasoning,
    /// Audio output or an audio transcript.
    Audio,
    /// A tool call, function call, or partial call arguments.
    ToolCall,
    /// Generated image data or an image generation event.
    Image,
    /// Output that is known to exist but does not fit another kind.
    Other,
}

/// How confidently a classified failure may be treated as transient.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum FailureKind {
    /// The classifier explicitly identified a transient failure.
    Transient,
    /// The classifier explicitly identified a permanent failure.
    Permanent,
    /// The classifier could not establish whether the failure is transient.
    Unknown,
}

/// A replay-relevant observation made from one stream item.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Observation {
    /// The item is known not to change the replay boundary or termination.
    Neutral,
    /// The item contains semantic model output.
    Output(OutputKind),
    /// The item establishes clean semantic completion.
    Completed,
    /// The item establishes an explicit failure.
    Failed(FailureKind),
    /// The item might contain output, so an uncrossed boundary cannot be proven.
    Uncertain,
}

/// Whether semantic output has crossed the whole-request replay boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ReplayBoundary {
    /// Every classified item so far was proven not to contain output.
    Uncrossed,
    /// Output was observed; the value records the first known output kind.
    Crossed(OutputKind),
    /// An item could not be proven free of output.
    Uncertain,
}

/// Why observation of the stream ended without a semantic terminal item.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ExitReason {
    /// The upstream stream returned `None` without an explicit terminal item.
    EndOfStream,
    /// The wrapper was dropped before a terminal outcome was established.
    Dropped,
    /// An upstream poll or item classifier unwound with a panic.
    ///
    /// This state is recorded when the wrapper is dropped during unwinding, or
    /// when a caller catches the panic and polls the same wrapper again.
    Panicked,
}

/// The terminal state observed for a stream.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Termination {
    /// The stream has not established a terminal outcome.
    Active,
    /// The classifier observed explicit clean completion.
    Completed,
    /// The classifier observed an explicit failure.
    Failed(FailureKind),
    /// Observation ended without an explicit semantic outcome.
    Exited(ExitReason),
}

/// The caller's independent guarantee about replaying the whole request.
///
/// This is deliberately separate from the response stream. Seeing no output
/// cannot prove that a request, tool, or upstream side effect is replay-safe.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ReplayContract {
    /// The caller has independently established that whole-request replay is safe.
    KnownSafe,
    /// The caller cannot establish whether replay is safe.
    Unknown,
    /// The caller knows that replay can duplicate a side effect.
    KnownUnsafe,
}

/// The conservative whole-request replay decision.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ReplayVerdict {
    /// All required evidence is present, so replay is authorized.
    Allow,
    /// Replay is not authorized.
    Deny,
}

/// A consistent, point-in-time view of replay evidence.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Snapshot {
    contract: ReplayContract,
    boundary: ReplayBoundary,
    termination: Termination,
    poll_in_progress: bool,
}

impl Snapshot {
    /// Returns the caller-provided replay contract.
    pub const fn contract(self) -> ReplayContract {
        self.contract
    }

    /// Returns the observed replay boundary.
    pub const fn boundary(self) -> ReplayBoundary {
        self.boundary
    }

    /// Returns the observed termination state.
    pub const fn termination(self) -> Termination {
        self.termination
    }

    /// Returns whether an upstream poll or item classification was in progress.
    ///
    /// A snapshot taken during this small window never authorizes replay. If a
    /// panic is caught by the caller, the flag remains set until the wrapper is
    /// dropped or polled again, preventing stale transient-failure evidence from
    /// authorizing replay.
    pub const fn poll_in_progress(self) -> bool {
        self.poll_in_progress
    }

    /// Computes a conservative replay verdict from this exact snapshot.
    ///
    /// The only allowing combination is a known-safe request, an uncrossed
    /// output boundary, and an explicitly transient failure.
    pub const fn verdict(self) -> ReplayVerdict {
        if self.poll_in_progress {
            return ReplayVerdict::Deny;
        }
        match (self.contract, self.boundary, self.termination) {
            (
                ReplayContract::KnownSafe,
                ReplayBoundary::Uncrossed,
                Termination::Failed(FailureKind::Transient),
            ) => ReplayVerdict::Allow,
            _ => ReplayVerdict::Deny,
        }
    }
}

#[derive(Debug)]
struct State {
    contract: ReplayContract,
    boundary: ReplayBoundary,
    termination: Termination,
    poll_in_progress: bool,
}

impl State {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            contract: self.contract,
            boundary: self.boundary,
            termination: self.termination,
            poll_in_progress: self.poll_in_progress,
        }
    }

    fn begin_poll(&mut self) {
        if self.poll_in_progress {
            self.record_panic();
        }
        self.poll_in_progress = true;
    }

    fn finish_poll(&mut self) {
        self.poll_in_progress = false;
    }

    fn wrapper_dropped(&mut self) {
        if self.poll_in_progress {
            self.record_panic();
        } else {
            self.exit(ExitReason::Dropped);
        }
    }

    fn record_panic(&mut self) {
        if matches!(self.boundary, ReplayBoundary::Uncrossed) {
            self.boundary = ReplayBoundary::Uncertain;
        }
        self.termination = Termination::Exited(ExitReason::Panicked);
        self.poll_in_progress = false;
    }

    fn observe(&mut self, observation: Observation) {
        match observation {
            Observation::Neutral => {}
            Observation::Output(kind) => {
                if !matches!(self.boundary, ReplayBoundary::Crossed(_)) {
                    self.boundary = ReplayBoundary::Crossed(kind);
                }
            }
            Observation::Uncertain => {
                if matches!(self.boundary, ReplayBoundary::Uncrossed) {
                    self.boundary = ReplayBoundary::Uncertain;
                }
            }
            Observation::Completed => {
                if matches!(self.termination, Termination::Active) {
                    self.termination = Termination::Completed;
                }
            }
            Observation::Failed(kind) => self.observe_failure(kind),
        }
    }

    fn observe_failure(&mut self, kind: FailureKind) {
        self.termination = match self.termination {
            Termination::Active | Termination::Completed => Termination::Failed(kind),
            Termination::Failed(previous) => Termination::Failed(merge_failures(previous, kind)),
            Termination::Exited(reason) => Termination::Exited(reason),
        };
    }

    fn exit(&mut self, reason: ExitReason) {
        if matches!(self.termination, Termination::Active) {
            self.termination = Termination::Exited(reason);
        }
    }
}

const fn merge_failures(left: FailureKind, right: FailureKind) -> FailureKind {
    match (left, right) {
        (FailureKind::Unknown, _) | (_, FailureKind::Unknown) => FailureKind::Unknown,
        (FailureKind::Permanent, _) | (_, FailureKind::Permanent) => FailureKind::Permanent,
        (FailureKind::Transient, FailureKind::Transient) => FailureKind::Transient,
    }
}

fn lock_state(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A cloneable handle for reading consistent stream snapshots.
///
/// Cloning a probe is cheap. Each call to [`Probe::snapshot`] acquires one
/// mutex and copies the complete state, so the boundary and termination always
/// describe the same point in time.
#[derive(Clone, Debug)]
pub struct Probe {
    state: Arc<Mutex<State>>,
}

impl Probe {
    /// Returns a consistent point-in-time snapshot.
    pub fn snapshot(&self) -> Snapshot {
        lock_state(&self.state).snapshot()
    }
}

#[derive(Debug)]
struct DropGuard {
    state: Arc<Mutex<State>>,
    armed: bool,
}

impl DropGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for DropGuard {
    fn drop(&mut self) {
        if self.armed {
            lock_state(&self.state).wrapper_dropped();
        }
    }
}

pin_project! {
    /// A transparent stream wrapper that records replay-relevant observations.
    ///
    /// The wrapper performs no prefetching. Each downstream `poll_next` polls
    /// the upstream stream at most once, and every item is returned unchanged.
    /// The replay state is committed before the item becomes visible to the
    /// downstream consumer.
    pub struct GuardedStream<S, C> {
        #[pin]
        inner: S,
        classifier: C,
        probe: Probe,
        drop_guard: DropGuard,
        ended: bool,
    }
}

impl<S, C> GuardedStream<S, C> {
    /// Wraps `inner` with `classifier` and an independent replay contract.
    pub fn new(inner: S, classifier: C, contract: ReplayContract) -> Self {
        let state = Arc::new(Mutex::new(State {
            contract,
            boundary: ReplayBoundary::Uncrossed,
            termination: Termination::Active,
            poll_in_progress: false,
        }));

        Self {
            inner,
            classifier,
            probe: Probe {
                state: Arc::clone(&state),
            },
            drop_guard: DropGuard { state, armed: true },
            ended: false,
        }
    }

    /// Returns a cloneable probe for this stream.
    pub fn probe(&self) -> Probe {
        self.probe.clone()
    }
}

impl<S, C> Stream for GuardedStream<S, C>
where
    S: Stream,
    C: Classify<S::Item>,
{
    type Item = S::Item;

    fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.ended {
            return Poll::Ready(None);
        }

        lock_state(&this.probe.state).begin_poll();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Pending => {
                lock_state(&this.probe.state).finish_poll();
                Poll::Pending
            }
            Poll::Ready(Some(item)) => {
                let observation = this.classifier.classify(&item);
                let mut state = lock_state(&this.probe.state);
                state.observe(observation);
                state.finish_poll();
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                *this.ended = true;
                let mut state = lock_state(&this.probe.state);
                state.finish_poll();
                state.exit(ExitReason::EndOfStream);
                this.drop_guard.disarm();
                Poll::Ready(None)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.ended {
            (0, Some(0))
        } else {
            self.inner.size_hint()
        }
    }
}

impl<S, C> FusedStream for GuardedStream<S, C>
where
    S: Stream,
    C: Classify<S::Item>,
{
    fn is_terminated(&self) -> bool {
        self.ended
    }
}
