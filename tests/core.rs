use std::collections::VecDeque;
use std::marker::PhantomPinned;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use futures_core::stream::{FusedStream, Stream};
use llm_stream_guard::{
    ExitReason, FailureKind, GuardedStream, Observation, OutputKind, ReplayBoundary,
    ReplayContract, ReplayVerdict, Termination,
};
use pin_project_lite::pin_project;

struct NoopWake;

impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

fn context() -> Context<'static> {
    let waker = Waker::from(Arc::new(NoopWake));
    Context::from_waker(Box::leak(Box::new(waker)))
}

pin_project! {
    struct TestStream<T> {
        items: VecDeque<Poll<Option<T>>>,
        polls: Arc<AtomicUsize>,
        #[pin]
        _not_unpin: PhantomPinned,
    }
}

impl<T> TestStream<T> {
    fn new(items: impl IntoIterator<Item = Poll<Option<T>>>, polls: Arc<AtomicUsize>) -> Self {
        Self {
            items: items.into_iter().collect(),
            polls,
            _not_unpin: PhantomPinned,
        }
    }
}

impl<T> Stream for TestStream<T> {
    type Item = T;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.polls.fetch_add(1, Ordering::SeqCst);
        this.items.pop_front().unwrap_or(Poll::Ready(None))
    }
}

#[test]
fn forwards_items_without_prefetch_and_supports_not_unpin() {
    let polls = Arc::new(AtomicUsize::new(0));
    let source = TestStream::new(
        [
            Poll::Pending,
            Poll::Ready(Some(String::from("first"))),
            Poll::Ready(Some(String::from("second"))),
            Poll::Ready(None),
        ],
        Arc::clone(&polls),
    );
    let guarded = GuardedStream::new(
        source,
        |_item: &String| Observation::Neutral,
        ReplayContract::Unknown,
    );
    let mut guarded = std::pin::pin!(guarded);
    let mut cx = context();

    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(guarded.as_mut().poll_next(&mut cx), Poll::Pending);
    assert_eq!(polls.load(Ordering::SeqCst), 1);
    assert_eq!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(String::from("first")))
    );
    assert_eq!(polls.load(Ordering::SeqCst), 2);
    assert_eq!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(String::from("second")))
    );
    assert_eq!(polls.load(Ordering::SeqCst), 3);
    assert_eq!(guarded.as_mut().poll_next(&mut cx), Poll::Ready(None));
    assert_eq!(polls.load(Ordering::SeqCst), 4);
    assert!(guarded.is_terminated());

    // The wrapper is fused and does not poll upstream after its first `None`.
    assert_eq!(guarded.as_mut().poll_next(&mut cx), Poll::Ready(None));
    assert_eq!(polls.load(Ordering::SeqCst), 4);
}

#[test]
fn commits_observation_before_returning_the_same_item() {
    let polls = Arc::new(AtomicUsize::new(0));
    let source = TestStream::new(
        [Poll::Ready(Some(7_u8)), Poll::Ready(None)],
        Arc::clone(&polls),
    );
    let guarded = GuardedStream::new(
        source,
        |_item: &u8| Observation::Output(OutputKind::Text),
        ReplayContract::KnownSafe,
    );
    let probe = guarded.probe();
    let mut guarded = std::pin::pin!(guarded);
    let mut cx = context();

    assert_eq!(guarded.as_mut().poll_next(&mut cx), Poll::Ready(Some(7)));
    assert_eq!(
        probe.snapshot().boundary(),
        ReplayBoundary::Crossed(OutputKind::Text)
    );
}

#[test]
fn fn_mut_classifier_can_keep_state() {
    let polls = Arc::new(AtomicUsize::new(0));
    let source = TestStream::new(
        [
            Poll::Ready(Some(1_u8)),
            Poll::Ready(Some(2)),
            Poll::Ready(None),
        ],
        polls,
    );
    let mut seen = 0_u8;
    let guarded = GuardedStream::new(
        source,
        move |_item: &u8| {
            seen += 1;
            if seen == 2 {
                Observation::Output(OutputKind::ToolCall)
            } else {
                Observation::Neutral
            }
        },
        ReplayContract::KnownSafe,
    );
    let probe = guarded.probe();
    let mut guarded = std::pin::pin!(guarded);
    let mut cx = context();

    assert!(matches!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(1))
    ));
    assert_eq!(probe.snapshot().boundary(), ReplayBoundary::Uncrossed);
    assert!(matches!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(2))
    ));
    assert_eq!(
        probe.snapshot().boundary(),
        ReplayBoundary::Crossed(OutputKind::ToolCall)
    );
}

fn snapshot_for(contract: ReplayContract, observation: Observation) -> llm_stream_guard::Snapshot {
    let polls = Arc::new(AtomicUsize::new(0));
    let source = TestStream::new([Poll::Ready(Some(()))], polls);
    let guarded = GuardedStream::new(source, move |_: &()| observation, contract);
    let probe = guarded.probe();
    let mut guarded = std::pin::pin!(guarded);
    let mut cx = context();
    assert!(matches!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(()))
    ));
    probe.snapshot()
}

#[test]
fn only_known_safe_uncrossed_transient_failure_allows_replay() {
    let allowed = snapshot_for(
        ReplayContract::KnownSafe,
        Observation::Failed(FailureKind::Transient),
    );
    assert_eq!(allowed.boundary(), ReplayBoundary::Uncrossed);
    assert_eq!(
        allowed.termination(),
        Termination::Failed(FailureKind::Transient)
    );
    assert_eq!(allowed.verdict(), ReplayVerdict::Allow);

    for contract in [ReplayContract::Unknown, ReplayContract::KnownUnsafe] {
        assert_eq!(
            snapshot_for(contract, Observation::Failed(FailureKind::Transient)).verdict(),
            ReplayVerdict::Deny
        );
    }
    for failure in [FailureKind::Permanent, FailureKind::Unknown] {
        assert_eq!(
            snapshot_for(ReplayContract::KnownSafe, Observation::Failed(failure)).verdict(),
            ReplayVerdict::Deny
        );
    }

    let crossed = {
        let polls = Arc::new(AtomicUsize::new(0));
        let source = TestStream::new([Poll::Ready(Some(1_u8)), Poll::Ready(Some(2_u8))], polls);
        let guarded = GuardedStream::new(
            source,
            |item: &u8| match item {
                1 => Observation::Output(OutputKind::Text),
                _ => Observation::Failed(FailureKind::Transient),
            },
            ReplayContract::KnownSafe,
        );
        let probe = guarded.probe();
        let mut guarded = std::pin::pin!(guarded);
        let mut cx = context();
        let _ = guarded.as_mut().poll_next(&mut cx);
        let _ = guarded.as_mut().poll_next(&mut cx);
        probe.snapshot()
    };
    assert_eq!(crossed.verdict(), ReplayVerdict::Deny);

    let uncertain = {
        let polls = Arc::new(AtomicUsize::new(0));
        let source = TestStream::new([Poll::Ready(Some(1_u8)), Poll::Ready(Some(2_u8))], polls);
        let guarded = GuardedStream::new(
            source,
            |item: &u8| match item {
                1 => Observation::Uncertain,
                _ => Observation::Failed(FailureKind::Transient),
            },
            ReplayContract::KnownSafe,
        );
        let probe = guarded.probe();
        let mut guarded = std::pin::pin!(guarded);
        let mut cx = context();
        let _ = guarded.as_mut().poll_next(&mut cx);
        let _ = guarded.as_mut().poll_next(&mut cx);
        probe.snapshot()
    };
    assert_eq!(uncertain.boundary(), ReplayBoundary::Uncertain);
    assert_eq!(uncertain.verdict(), ReplayVerdict::Deny);
}

#[test]
fn dropped_and_unmarked_eof_never_authorize_replay() {
    let dropped_probe = {
        let polls = Arc::new(AtomicUsize::new(0));
        let source = TestStream::<()>::new([Poll::Pending], polls);
        let guarded = GuardedStream::new(
            source,
            |_: &()| Observation::Neutral,
            ReplayContract::KnownSafe,
        );
        let probe = guarded.probe();
        drop(guarded);
        probe
    };
    assert_eq!(
        dropped_probe.snapshot().termination(),
        Termination::Exited(ExitReason::Dropped)
    );
    assert_eq!(dropped_probe.snapshot().verdict(), ReplayVerdict::Deny);

    let eof_probe = {
        let polls = Arc::new(AtomicUsize::new(0));
        let source = TestStream::<()>::new([Poll::Ready(None)], polls);
        let guarded = GuardedStream::new(
            source,
            |_: &()| Observation::Neutral,
            ReplayContract::KnownSafe,
        );
        let probe = guarded.probe();
        let mut guarded = std::pin::pin!(guarded);
        let mut cx = context();
        assert_eq!(guarded.as_mut().poll_next(&mut cx), Poll::Ready(None));
        probe
    };
    assert_eq!(
        eof_probe.snapshot().termination(),
        Termination::Exited(ExitReason::EndOfStream)
    );
    assert_eq!(eof_probe.snapshot().verdict(), ReplayVerdict::Deny);
}

#[test]
fn explicit_terminal_observations_survive_wrapper_drop() {
    let failed = snapshot_for(
        ReplayContract::KnownSafe,
        Observation::Failed(FailureKind::Transient),
    );
    assert_eq!(
        failed.termination(),
        Termination::Failed(FailureKind::Transient)
    );
    assert_eq!(failed.verdict(), ReplayVerdict::Allow);

    let completed = snapshot_for(ReplayContract::KnownSafe, Observation::Completed);
    assert_eq!(completed.termination(), Termination::Completed);
    assert_eq!(completed.verdict(), ReplayVerdict::Deny);
}

#[test]
fn later_failure_outranks_completion_and_failure_evidence_only_gets_stricter() {
    let polls = Arc::new(AtomicUsize::new(0));
    let source = TestStream::new(
        [
            Poll::Ready(Some(0_u8)),
            Poll::Ready(Some(1_u8)),
            Poll::Ready(Some(2_u8)),
            Poll::Ready(None),
        ],
        polls,
    );
    let guarded = GuardedStream::new(
        source,
        |item: &u8| match item {
            0 => Observation::Completed,
            1 => Observation::Failed(FailureKind::Transient),
            _ => Observation::Failed(FailureKind::Unknown),
        },
        ReplayContract::KnownSafe,
    );
    let probe = guarded.probe();
    let mut guarded = std::pin::pin!(guarded);
    let mut cx = context();

    let _ = guarded.as_mut().poll_next(&mut cx);
    assert_eq!(probe.snapshot().termination(), Termination::Completed);
    let _ = guarded.as_mut().poll_next(&mut cx);
    assert_eq!(
        probe.snapshot().termination(),
        Termination::Failed(FailureKind::Transient)
    );
    let _ = guarded.as_mut().poll_next(&mut cx);
    assert_eq!(
        probe.snapshot().termination(),
        Termination::Failed(FailureKind::Unknown)
    );
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Deny);
}

#[test]
fn classifier_panic_cannot_leave_stale_allow_evidence() {
    let polls = Arc::new(AtomicUsize::new(0));
    let source = TestStream::new([Poll::Ready(Some(1_u8)), Poll::Ready(Some(2_u8))], polls);
    let guarded = GuardedStream::new(
        source,
        |item: &u8| match item {
            1 => Observation::Failed(FailureKind::Transient),
            _ => panic!("synthetic classifier panic"),
        },
        ReplayContract::KnownSafe,
    );
    let probe = guarded.probe();
    let mut guarded = Box::pin(guarded);
    let mut cx = context();

    assert!(matches!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(1))
    ));
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Allow);

    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = guarded.as_mut().poll_next(&mut cx);
    }));
    assert!(panic.is_err());
    assert!(probe.snapshot().poll_in_progress());
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Deny);

    drop(guarded);
    assert_eq!(
        probe.snapshot().termination(),
        Termination::Exited(ExitReason::Panicked)
    );
    assert_eq!(probe.snapshot().boundary(), ReplayBoundary::Uncertain);
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Deny);
}

struct PanickingStream;

impl Stream for PanickingStream {
    type Item = ();

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        panic!("synthetic upstream panic");
    }
}

#[test]
fn upstream_panic_is_recorded_when_the_wrapper_is_dropped() {
    let guarded = GuardedStream::new(
        PanickingStream,
        |_: &()| Observation::Neutral,
        ReplayContract::KnownSafe,
    );
    let probe = guarded.probe();
    let mut guarded = Box::pin(guarded);
    let mut cx = context();

    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = guarded.as_mut().poll_next(&mut cx);
    }));
    assert!(panic.is_err());
    assert!(probe.snapshot().poll_in_progress());
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Deny);

    drop(guarded);
    assert_eq!(
        probe.snapshot().termination(),
        Termination::Exited(ExitReason::Panicked)
    );
    assert_eq!(probe.snapshot().boundary(), ReplayBoundary::Uncertain);
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Deny);
}
