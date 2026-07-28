#![cfg(feature = "openai-json")]

use llm_stream_guard::openai::{
    classify_openai_event, ClassifierLimits, OpenAiClassifier, SseEventRef,
};
use llm_stream_guard::{Classify, FailureKind, Observation, OutputKind};

fn classify(data: &[u8]) -> Observation {
    classify_openai_event(SseEventRef::new(data), ClassifierLimits::default())
}

fn classify_named(event: &str, data: &[u8]) -> Observation {
    classify_openai_event(SseEventRef::named(event, data), ClassifierLimits::default())
}

#[test]
fn borrowed_event_constructors_do_not_transform_input() {
    let unnamed = SseEventRef::new(b"payload");
    assert_eq!(unnamed.event, None);
    assert_eq!(unnamed.data, b"payload");

    let named = SseEventRef::named("response.created", b"{}");
    assert_eq!(named.event, Some("response.created"));
    assert_eq!(named.data, b"{}");
}

#[test]
fn limits_are_clamped_to_hard_bounds() {
    let minimum = ClassifierLimits::new(0)
        .with_max_event_name_bytes(0)
        .with_max_choices(0);
    assert_eq!(minimum.max_data_bytes(), 1);
    assert_eq!(minimum.max_event_name_bytes(), 1);
    assert_eq!(minimum.max_choices(), 1);

    let maximum = ClassifierLimits::new(usize::MAX)
        .with_max_event_name_bytes(usize::MAX)
        .with_max_choices(usize::MAX);
    assert_eq!(maximum.max_data_bytes(), 1024 * 1024);
    assert_eq!(maximum.max_event_name_bytes(), 4 * 1024);
    assert_eq!(maximum.max_choices(), 1024);

    let defaults = ClassifierLimits::default();
    assert_eq!(defaults.max_data_bytes(), 64 * 1024);
    assert_eq!(defaults.max_event_name_bytes(), 256);
    assert_eq!(defaults.max_choices(), 128);
}

#[test]
fn completion_sentinels_and_explicit_failures_have_conservative_precedence() {
    assert_eq!(classify(b" \r\n[DONE]\t"), Observation::Completed);
    assert_eq!(
        classify_named("response.completed", b"[DONE]"),
        Observation::Completed
    );

    for event in ["error", "response.failed", "response.incomplete"] {
        assert_eq!(
            classify_named(event, b"[DONE]"),
            Observation::Failed(FailureKind::Unknown),
            "event={event}"
        );
        assert_eq!(
            classify_named(event, b""),
            Observation::Failed(FailureKind::Unknown),
            "empty event={event}"
        );
    }

    assert_eq!(
        classify_named("response.output_text.delta", b"[DONE]"),
        Observation::Uncertain
    );
    assert_eq!(
        classify_named("vendor.done", b"[DONE]"),
        Observation::Uncertain
    );
}

#[test]
fn responses_terminal_payloads_are_recognized_without_retaining_errors() {
    for data in [
        br#"{"type":"response.failed","response":{"status":"failed"}}"#.as_slice(),
        br#"{"type":"response.incomplete","response":{"status":"incomplete"}}"#.as_slice(),
        br#"{"type":"error","error":{"message":"do not retain me"}}"#.as_slice(),
        br#"{"error":{"code":"rate_limit_exceeded","message":"secret"}}"#.as_slice(),
        br#"{"response":{"status":"failed"}}"#.as_slice(),
        br#"{"response":{"status":"incomplete"}}"#.as_slice(),
        br#"{"response":{"error":{"message":"secret"}}}"#.as_slice(),
    ] {
        assert_eq!(
            classify(data),
            Observation::Failed(FailureKind::Unknown),
            "payload={}",
            String::from_utf8_lossy(data)
        );
    }

    assert_eq!(
        classify(br#"{"type":"response.completed","response":{"status":"completed"}}"#),
        Observation::Completed
    );
    assert_eq!(
        classify_named("response.completed", br#"{}"#),
        Observation::Completed
    );
    assert_eq!(
        classify(br#"{"response":{"status":"completed"}}"#),
        Observation::Completed
    );
}

#[test]
fn responses_output_event_matrix_maps_to_semantic_kinds() {
    let cases = [
        ("response.output_text.delta", OutputKind::Text),
        ("response.output_text.done", OutputKind::Text),
        ("response.refusal.delta", OutputKind::Refusal),
        ("response.reasoning.delta", OutputKind::Reasoning),
        ("response.reasoning_text.delta", OutputKind::Reasoning),
        (
            "response.reasoning_summary_text.delta",
            OutputKind::Reasoning,
        ),
        ("response.audio.delta", OutputKind::Audio),
        ("response.output_audio.delta", OutputKind::Audio),
        ("response.audio_transcript.delta", OutputKind::Audio),
        (
            "response.function_call_arguments.delta",
            OutputKind::ToolCall,
        ),
        ("response.tool_call.delta", OutputKind::ToolCall),
        (
            "response.image_generation_call.partial_image",
            OutputKind::Image,
        ),
    ];

    for (event_type, kind) in cases {
        let payload = format!(r#"{{"type":"{event_type}","delta":"sensitive"}}"#);
        assert_eq!(
            classify(payload.as_bytes()),
            Observation::Output(kind),
            "type={event_type}"
        );
        assert_eq!(
            classify_named(event_type, br#"{}"#),
            Observation::Output(kind),
            "event={event_type}"
        );
    }
}

#[test]
fn responses_metadata_is_neutral_unless_it_already_contains_output() {
    for event_type in [
        "response.created",
        "response.in_progress",
        "response.queued",
    ] {
        let empty = format!(r#"{{"type":"{event_type}","response":{{"output":[]}}}}"#);
        assert_eq!(classify(empty.as_bytes()), Observation::Neutral);

        let populated =
            format!(r#"{{"type":"{event_type}","response":{{"output":[{{"type":"message"}}]}}}}"#);
        assert_eq!(
            classify(populated.as_bytes()),
            Observation::Output(OutputKind::Other)
        );
    }
}

#[test]
fn chat_completions_delta_matrix_maps_to_semantic_kinds() {
    let cases = [
        (r#""content":"hello""#, OutputKind::Text),
        (r#""refusal":"no""#, OutputKind::Refusal),
        (r#""reasoning":"step""#, OutputKind::Reasoning),
        (r#""reasoning_content":"step""#, OutputKind::Reasoning),
        (r#""audio":{"id":"a"}"#, OutputKind::Audio),
        (
            r#""tool_calls":[{"function":{"arguments":"{"}}]"#,
            OutputKind::ToolCall,
        ),
        (r#""function_call":{"arguments":"{"}"#, OutputKind::ToolCall),
    ];

    for (delta, kind) in cases {
        let payload = format!(
            r#"{{"object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{{delta}}},"finish_reason":null}}]}}"#
        );
        assert_eq!(
            classify(payload.as_bytes()),
            Observation::Output(kind),
            "delta={delta}"
        );
    }
}

#[test]
fn chat_completions_supports_typed_legacy_and_full_message_shapes() {
    assert_eq!(
        classify(br#"{"type":"chat.completion.chunk","choices":[{"delta":{"content":"x"}}]}"#),
        Observation::Output(OutputKind::Text)
    );
    assert_eq!(
        classify(br#"{"choices":[{"text":"legacy"}]}"#),
        Observation::Output(OutputKind::Text)
    );
    assert_eq!(
        classify(br#"{"choices":[{"message":{"tool_calls":[{"id":"call"}]}}]}"#),
        Observation::Output(OutputKind::ToolCall)
    );
    assert_eq!(
        classify(br#"{"choices":[{"message":{"content":"complete"}}]}"#),
        Observation::Output(OutputKind::Text)
    );
}

#[test]
fn chat_metadata_is_neutral_but_unknown_choice_content_is_uncertain() {
    for data in [
        br#"{"object":"chat.completion.chunk","choices":[]}"#.as_slice(),
        br#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#.as_slice(),
        br#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":2}}"#.as_slice(),
        br#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":null}]}"#.as_slice(),
    ] {
        assert_eq!(classify(data), Observation::Neutral);
    }

    for data in [
        br#"{"object":"chat.completion.chunk","choices":[7]}"#.as_slice(),
        br#"{"object":"chat.completion.chunk"}"#.as_slice(),
        br#"{"object":"chat.completion.chunk","choices":[{"delta":"text"}]}"#.as_slice(),
        br#"{"object":"chat.completion.chunk","choices":[{"delta":{"vendor_output":"visible"}}]}"#
            .as_slice(),
        br#"{"object":"chat.completion.chunk","choices":[{"vendor_output":"visible"}]}"#.as_slice(),
    ] {
        assert_eq!(classify(data), Observation::Uncertain);
    }
}

#[test]
fn malformed_nonstandard_and_duplicate_json_is_uncertain() {
    for data in [
        b"{not-json}".as_slice(),
        b"\xff".as_slice(),
        br#"{"type":"response.created"} trailing"#.as_slice(),
        br#"{"value":NaN}"#.as_slice(),
        br#"{"value":Infinity}"#.as_slice(),
        br#"{"value":1e999999}"#.as_slice(),
        br#"{"type":"response.created","x":1,"x":2}"#.as_slice(),
        br#"{"type":"response.created","x":1,"\u0078":2}"#.as_slice(),
        br#"{"type":"response.created","nested":{"x":1,"x":2}}"#.as_slice(),
    ] {
        assert_eq!(
            classify(data),
            Observation::Uncertain,
            "payload={}",
            String::from_utf8_lossy(data)
        );
    }
}

#[test]
fn scalar_array_unknown_and_empty_semantic_events_fail_closed() {
    for data in [
        br#""plain""#.as_slice(),
        br#"[]"#.as_slice(),
        br#"null"#.as_slice(),
        br#"{"vendor_output":"visible"}"#.as_slice(),
        br#"{"type":"response.future_delta","delta":"visible"}"#.as_slice(),
        br#"{"type":7}"#.as_slice(),
    ] {
        assert_eq!(classify(data), Observation::Uncertain);
    }

    assert_eq!(classify(b""), Observation::Neutral);
    assert_eq!(
        classify_named("response.created", b""),
        Observation::Neutral
    );
    assert_eq!(
        classify_named("response.output_text.delta", b""),
        Observation::Uncertain
    );
}

#[test]
fn conflicting_event_name_and_payload_type_is_uncertain() {
    assert_eq!(
        classify_named(
            "response.completed",
            br#"{"type":"response.output_text.delta","delta":"visible"}"#,
        ),
        Observation::Uncertain
    );
    assert_eq!(
        classify_named(
            "vendor.delta",
            br#"{"type":"response.output_text.delta","delta":"visible"}"#,
        ),
        Observation::Uncertain
    );
    assert_eq!(
        classify_named(
            "message",
            br#"{"type":"response.output_text.delta","delta":"visible"}"#,
        ),
        Observation::Output(OutputKind::Text)
    );

    // Explicit failure evidence wins over a conflicting completion signal.
    assert_eq!(
        classify_named("response.failed", br#"{"type":"response.completed"}"#),
        Observation::Failed(FailureKind::Unknown)
    );
}

#[test]
fn byte_and_choice_limits_fail_closed_before_unbounded_work() {
    let tiny_data = ClassifierLimits::new(2);
    assert_eq!(
        classify_openai_event(SseEventRef::new(br#"{} "#), tiny_data),
        Observation::Uncertain
    );

    let one_byte_name = ClassifierLimits::default().with_max_event_name_bytes(1);
    assert_eq!(
        classify_openai_event(SseEventRef::named("你", br#"{}"#), one_byte_name,),
        Observation::Uncertain
    );

    let one_choice = ClassifierLimits::default().with_max_choices(1);
    assert_eq!(
        classify_openai_event(
            SseEventRef::new(
                br#"{"choices":[{"delta":{"role":"assistant"}},{"delta":{"content":"not inspected"}}]}"#,
            ),
            one_choice,
        ),
        Observation::Uncertain
    );
}

#[test]
fn reusable_classifier_implements_the_core_trait_without_retaining_payloads() {
    let limits = ClassifierLimits::new(1024);
    let mut classifier = OpenAiClassifier::new(limits);
    assert_eq!(classifier.limits(), limits);

    let payload = br#"{"type":"response.output_text.delta","delta":"ephemeral"}"#.to_vec();
    let observation = classifier.classify(&SseEventRef::new(&payload));
    drop(payload);

    assert_eq!(observation, Observation::Output(OutputKind::Text));
    assert_eq!(classifier.limits(), limits);
}
