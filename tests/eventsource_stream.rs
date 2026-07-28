#![cfg(feature = "openai-json")]

use std::io;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use eventsource_stream::{Event, EventStreamError, Eventsource};
use futures_core::Stream;
use futures_util::stream;
use llm_stream_guard::openai::{classify_openai_event, ClassifierLimits, SseEventRef};
use llm_stream_guard::{
    ExitReason, FailureKind, GuardedStream, Observation, OutputKind, ReplayBoundary,
    ReplayContract, ReplayVerdict, Termination,
};

struct NoopWake;

impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

fn context() -> Context<'static> {
    let waker = Waker::from(Arc::new(NoopWake));
    Context::from_waker(Box::leak(Box::new(waker)))
}

fn classify_event(item: &Result<Event, EventStreamError<io::Error>>) -> Observation {
    match item {
        Ok(event) => classify_openai_event(
            SseEventRef::named(&event.event, event.data.as_bytes()),
            ClassifierLimits::default(),
        ),
        Err(EventStreamError::Transport(_)) => Observation::Failed(FailureKind::Transient),
        Err(EventStreamError::Utf8(_) | EventStreamError::Parser(_)) => Observation::Uncertain,
    }
}

#[test]
fn parsed_partial_output_then_eof_is_not_clean_completion() {
    let source = stream::iter([Ok::<_, io::Error>(
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
    )]);
    let events = source.eventsource();
    let guarded = GuardedStream::new(events, classify_event, ReplayContract::KnownSafe);
    let probe = guarded.probe();
    let mut guarded = Box::pin(guarded);
    let mut cx = context();

    assert!(matches!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(Ok(_)))
    ));
    assert_eq!(
        probe.snapshot().boundary(),
        ReplayBoundary::Crossed(OutputKind::Text)
    );
    assert!(matches!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(None)
    ));
    assert_eq!(
        probe.snapshot().termination(),
        Termination::Exited(ExitReason::EndOfStream)
    );
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Deny);
}

#[test]
fn parsed_done_event_is_explicit_completion() {
    let source = stream::iter([Ok::<_, io::Error>("data: [DONE]\n\n")]);
    let events = source.eventsource();
    let guarded = GuardedStream::new(events, classify_event, ReplayContract::KnownSafe);
    let probe = guarded.probe();
    let mut guarded = Box::pin(guarded);
    let mut cx = context();

    assert!(matches!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(Ok(_)))
    ));
    assert_eq!(probe.snapshot().termination(), Termination::Completed);
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Deny);
}

#[test]
fn parser_transport_error_is_visible_to_the_application_classifier() {
    let source = stream::iter([Err::<&'static str, _>(io::Error::new(
        io::ErrorKind::ConnectionReset,
        "synthetic reset",
    ))]);
    let events = source.eventsource();
    let guarded = GuardedStream::new(events, classify_event, ReplayContract::KnownSafe);
    let probe = guarded.probe();
    let mut guarded = Box::pin(guarded);
    let mut cx = context();

    assert!(matches!(
        guarded.as_mut().poll_next(&mut cx),
        Poll::Ready(Some(Err(EventStreamError::Transport(_))))
    ));
    assert_eq!(
        probe.snapshot().termination(),
        Termination::Failed(FailureKind::Transient)
    );
    assert_eq!(probe.snapshot().boundary(), ReplayBoundary::Uncrossed);
    assert_eq!(probe.snapshot().verdict(), ReplayVerdict::Allow);
}
