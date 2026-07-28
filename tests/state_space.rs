use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use futures_core::Stream;
use futures_util::stream;
use llm_stream_guard::{
    FailureKind, GuardedStream, Observation, OutputKind, ReplayBoundary, ReplayContract,
    ReplayVerdict, Termination,
};

const OBSERVATIONS: [Observation; 7] = [
    Observation::Neutral,
    Observation::Output(OutputKind::Text),
    Observation::Uncertain,
    Observation::Completed,
    Observation::Failed(FailureKind::Transient),
    Observation::Failed(FailureKind::Permanent),
    Observation::Failed(FailureKind::Unknown),
];

struct NoopWake;

impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

fn context() -> Context<'static> {
    let waker = Waker::from(Arc::new(NoopWake));
    Context::from_waker(Box::leak(Box::new(waker)))
}

fn visit_sequences(
    remaining: usize,
    prefix: &mut Vec<Observation>,
    visitor: &mut impl FnMut(&[Observation]),
) {
    if remaining == 0 {
        visitor(prefix);
        return;
    }

    for observation in OBSERVATIONS {
        prefix.push(observation);
        visit_sequences(remaining - 1, prefix, visitor);
        prefix.pop();
    }
}

#[test]
fn exhaustive_short_sequences_preserve_fail_closed_invariants() {
    let mut checked = 0_usize;

    for length in 0..=4 {
        visit_sequences(length, &mut Vec::new(), &mut |sequence| {
            for contract in [
                ReplayContract::KnownSafe,
                ReplayContract::Unknown,
                ReplayContract::KnownUnsafe,
            ] {
                checked += 1;
                let upstream = stream::iter(sequence.iter().copied());
                let guarded = GuardedStream::new(
                    upstream,
                    |observation: &Observation| *observation,
                    contract,
                );
                let probe = guarded.probe();
                let mut guarded = Box::pin(guarded);
                let mut cx = context();
                let mut first_output = None;
                let mut uncertain_without_output = false;

                for observation in sequence {
                    assert_eq!(
                        guarded.as_mut().poll_next(&mut cx),
                        Poll::Ready(Some(*observation))
                    );
                    match observation {
                        Observation::Output(kind) => {
                            first_output.get_or_insert(*kind);
                        }
                        Observation::Uncertain if first_output.is_none() => {
                            uncertain_without_output = true;
                        }
                        _ => {}
                    }

                    let snapshot = probe.snapshot();
                    assert!(!snapshot.poll_in_progress());
                    match first_output {
                        Some(kind) => {
                            assert_eq!(snapshot.boundary(), ReplayBoundary::Crossed(kind));
                            assert_eq!(snapshot.verdict(), ReplayVerdict::Deny);
                        }
                        None if uncertain_without_output => {
                            assert_eq!(snapshot.boundary(), ReplayBoundary::Uncertain);
                            assert_eq!(snapshot.verdict(), ReplayVerdict::Deny);
                        }
                        None => assert_eq!(snapshot.boundary(), ReplayBoundary::Uncrossed),
                    }

                    if snapshot.verdict() == ReplayVerdict::Allow {
                        assert_eq!(contract, ReplayContract::KnownSafe);
                        assert_eq!(snapshot.boundary(), ReplayBoundary::Uncrossed);
                        assert_eq!(
                            snapshot.termination(),
                            Termination::Failed(FailureKind::Transient)
                        );
                    }
                }

                assert_eq!(guarded.as_mut().poll_next(&mut cx), Poll::Ready(None));
                let final_snapshot = probe.snapshot();
                if final_snapshot.verdict() == ReplayVerdict::Allow {
                    assert_eq!(contract, ReplayContract::KnownSafe);
                    assert_eq!(final_snapshot.boundary(), ReplayBoundary::Uncrossed);
                    assert_eq!(
                        final_snapshot.termination(),
                        Termination::Failed(FailureKind::Transient)
                    );
                }
            }
        });
    }

    assert_eq!(checked, 8_403);
}
