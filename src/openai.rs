//! Conservative classification of already-framed OpenAI-compatible SSE events.
//!
//! This module deliberately does not parse an SSE byte stream. Callers pass one
//! complete event after their transport/parser has joined its `data:` lines.
//! Classification temporarily parses bounded JSON, returns only a semantic
//! [`Observation`], and does not retain event names, response text, tool
//! arguments, error messages, or any other payload data.

use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::{Classify, FailureKind, Observation, OutputKind};

const DEFAULT_MAX_EVENT_NAME_BYTES: usize = 256;
const DEFAULT_MAX_DATA_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_CHOICES: usize = 128;
const HARD_MAX_EVENT_NAME_BYTES: usize = 4 * 1024;
const HARD_MAX_DATA_BYTES: usize = 1024 * 1024;
const HARD_MAX_CHOICES: usize = 1024;

/// A borrowed, already-framed Server-Sent Event.
///
/// `data` is the complete event payload after an SSE parser has removed field
/// prefixes and joined multiple `data:` lines. This type does not protect an
/// upstream parser from unbounded buffering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SseEventRef<'a> {
    /// The optional SSE `event` field, without the `event:` prefix.
    pub event: Option<&'a str>,
    /// The joined SSE data bytes, without any `data:` prefixes.
    pub data: &'a [u8],
}

impl<'a> SseEventRef<'a> {
    /// Creates an event that uses the default SSE message type.
    pub const fn new(data: &'a [u8]) -> Self {
        Self { event: None, data }
    }

    /// Creates an event with an explicit SSE event name.
    pub const fn named(event: &'a str, data: &'a [u8]) -> Self {
        Self {
            event: Some(event),
            data,
        }
    }
}

/// Hard-bounded resource limits for OpenAI-compatible JSON classification.
///
/// Values supplied by callers are clamped to crate-level hard ceilings. A
/// limit of zero is raised to one byte/item. These limits only bound this
/// classifier; they cannot retroactively bound memory used by an SSE parser.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ClassifierLimits {
    max_event_name_bytes: usize,
    max_data_bytes: usize,
    max_choices: usize,
}

impl ClassifierLimits {
    /// Creates limits with a caller-selected JSON payload limit.
    ///
    /// The event-name and Chat Completions choice limits retain their defaults.
    pub const fn new(max_data_bytes: usize) -> Self {
        Self {
            max_event_name_bytes: DEFAULT_MAX_EVENT_NAME_BYTES,
            max_data_bytes: bounded(max_data_bytes, HARD_MAX_DATA_BYTES),
            max_choices: DEFAULT_MAX_CHOICES,
        }
    }

    /// Sets the maximum UTF-8 byte length of the optional SSE event name.
    pub const fn with_max_event_name_bytes(mut self, value: usize) -> Self {
        self.max_event_name_bytes = bounded(value, HARD_MAX_EVENT_NAME_BYTES);
        self
    }

    /// Sets the maximum number of Chat Completions choices inspected.
    pub const fn with_max_choices(mut self, value: usize) -> Self {
        self.max_choices = bounded(value, HARD_MAX_CHOICES);
        self
    }

    /// Returns the effective event-name byte limit.
    pub const fn max_event_name_bytes(self) -> usize {
        self.max_event_name_bytes
    }

    /// Returns the effective JSON payload byte limit.
    pub const fn max_data_bytes(self) -> usize {
        self.max_data_bytes
    }

    /// Returns the effective Chat Completions choice limit.
    pub const fn max_choices(self) -> usize {
        self.max_choices
    }
}

impl Default for ClassifierLimits {
    fn default() -> Self {
        Self {
            max_event_name_bytes: DEFAULT_MAX_EVENT_NAME_BYTES,
            max_data_bytes: DEFAULT_MAX_DATA_BYTES,
            max_choices: DEFAULT_MAX_CHOICES,
        }
    }
}

const fn bounded(value: usize, hard_max: usize) -> usize {
    if value == 0 {
        1
    } else if value > hard_max {
        hard_max
    } else {
        value
    }
}

/// A reusable classifier for OpenAI-compatible, already-framed SSE events.
///
/// The classifier stores only limits. It never stores an event or parsed JSON
/// value after [`Classify::classify`] returns.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct OpenAiClassifier {
    limits: ClassifierLimits,
}

impl OpenAiClassifier {
    /// Creates a classifier with explicit bounded limits.
    pub const fn new(limits: ClassifierLimits) -> Self {
        Self { limits }
    }

    /// Returns this classifier's effective limits.
    pub const fn limits(self) -> ClassifierLimits {
        self.limits
    }
}

impl<'a> Classify<SseEventRef<'a>> for OpenAiClassifier {
    fn classify(&mut self, item: &SseEventRef<'a>) -> Observation {
        classify_openai_event(*item, self.limits)
    }
}

/// Classifies one bounded, already-framed OpenAI-compatible SSE event.
///
/// Known output deltas cross the replay boundary. Explicit completion, failed,
/// incomplete, and error events establish terminal observations. Invalid JSON,
/// duplicate object keys, conflicting event/type signals, oversized input, and
/// unknown semantic events all return [`Observation::Uncertain`]. Failures are
/// conservatively [`FailureKind::Unknown`]; a stream body alone is insufficient
/// evidence that replay is transient-safe.
///
/// ```
/// use llm_stream_guard::openai::{
///     classify_openai_event, ClassifierLimits, SseEventRef,
/// };
/// use llm_stream_guard::{Observation, OutputKind};
///
/// let event = SseEventRef::named(
///     "response.function_call_arguments.delta",
///     br#"{"type":"response.function_call_arguments.delta","delta":"{\"city\":"}"#,
/// );
/// assert_eq!(
///     classify_openai_event(event, ClassifierLimits::default()),
///     Observation::Output(OutputKind::ToolCall),
/// );
/// ```
pub fn classify_openai_event(event: SseEventRef<'_>, limits: ClassifierLimits) -> Observation {
    let event_name = match event.event {
        Some(name) if name.len() > limits.max_event_name_bytes => {
            return Observation::Uncertain;
        }
        Some(name) => Some(name),
        None => None,
    };

    if event.data.len() > limits.max_data_bytes {
        return Observation::Uncertain;
    }

    let data = trim_ascii(event.data);
    let named_signal = event_name.map(signal_for_type);

    if data == b"[DONE]" {
        return match named_signal {
            Some(Signal::Failed) => Observation::Failed(FailureKind::Unknown),
            Some(Signal::Completed) | Some(Signal::Envelope) | None => Observation::Completed,
            Some(Signal::Neutral) | Some(Signal::Output(_)) | Some(Signal::ChatChunk) => {
                Observation::Uncertain
            }
            Some(Signal::Unknown) => Observation::Uncertain,
        };
    }

    if data.is_empty() {
        return match named_signal {
            Some(Signal::Failed) => Observation::Failed(FailureKind::Unknown),
            Some(Signal::Completed) => Observation::Completed,
            Some(Signal::Neutral) | Some(Signal::Envelope) | None => Observation::Neutral,
            Some(Signal::Output(_)) | Some(Signal::ChatChunk) | Some(Signal::Unknown) => {
                Observation::Uncertain
            }
        };
    }

    let payload: Value = match serde_json::from_slice(data) {
        Ok(payload) => payload,
        Err(_) => return Observation::Uncertain,
    };
    if has_duplicate_object_key(data) {
        return Observation::Uncertain;
    }
    let object = match payload.as_object() {
        Some(object) => object,
        None => return Observation::Uncertain,
    };

    if is_failure_payload(object, named_signal) {
        return Observation::Failed(FailureKind::Unknown);
    }

    let payload_type = match object.get("type") {
        Some(Value::String(value)) => Some(value.as_str()),
        Some(_) => return Observation::Uncertain,
        None => None,
    };
    let typed_signal = payload_type.map(signal_for_type);

    if signals_conflict(named_signal, typed_signal) {
        return Observation::Uncertain;
    }

    let signal = prefer_typed_signal(named_signal, typed_signal);
    match signal {
        Some(Signal::Completed) => Observation::Completed,
        Some(Signal::Failed) => Observation::Failed(FailureKind::Unknown),
        Some(Signal::Output(kind)) => Observation::Output(kind),
        Some(Signal::ChatChunk) => classify_chat_choices(object, limits.max_choices),
        Some(Signal::Neutral) => classify_response_metadata(object),
        Some(Signal::Envelope) | None if object.contains_key("choices") => {
            classify_chat_choices(object, limits.max_choices)
        }
        Some(Signal::Envelope) | None if response_status(object) == Some("completed") => {
            Observation::Completed
        }
        Some(Signal::Envelope) | None => Observation::Uncertain,
        Some(Signal::Unknown) => Observation::Uncertain,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Signal {
    Envelope,
    Neutral,
    ChatChunk,
    Output(OutputKind),
    Completed,
    Failed,
    Unknown,
}

fn signal_for_type(value: &str) -> Signal {
    match value {
        "message" => Signal::Envelope,
        "chat.completion.chunk" => Signal::ChatChunk,
        "response.created" | "response.in_progress" | "response.queued" => Signal::Neutral,
        "response.output_text.delta" | "response.output_text.done" => {
            Signal::Output(OutputKind::Text)
        }
        "response.refusal.delta" | "response.refusal.done" => Signal::Output(OutputKind::Refusal),
        "response.reasoning.delta"
        | "response.reasoning.done"
        | "response.reasoning_text.delta"
        | "response.reasoning_text.done"
        | "response.reasoning_summary.delta"
        | "response.reasoning_summary.done"
        | "response.reasoning_summary_text.delta"
        | "response.reasoning_summary_text.done" => Signal::Output(OutputKind::Reasoning),
        "response.audio.delta"
        | "response.audio.done"
        | "response.audio_transcript.delta"
        | "response.audio_transcript.done"
        | "response.output_audio.delta"
        | "response.output_audio.done"
        | "response.output_audio_transcript.delta"
        | "response.output_audio_transcript.done" => Signal::Output(OutputKind::Audio),
        "response.function_call_arguments.delta"
        | "response.function_call_arguments.done"
        | "response.tool_call.delta"
        | "response.tool_call.done" => Signal::Output(OutputKind::ToolCall),
        "response.image_generation_call.partial_image" => Signal::Output(OutputKind::Image),
        "response.completed" => Signal::Completed,
        "error" | "response.failed" | "response.incomplete" => Signal::Failed,
        _ => Signal::Unknown,
    }
}

fn signals_conflict(named: Option<Signal>, typed: Option<Signal>) -> bool {
    match (named, typed) {
        (None | Some(Signal::Envelope), _) | (_, None | Some(Signal::Envelope)) => false,
        (Some(left), Some(right)) => left != right,
    }
}

fn prefer_typed_signal(named: Option<Signal>, typed: Option<Signal>) -> Option<Signal> {
    match typed {
        Some(Signal::Envelope) | None => named,
        Some(signal) => Some(signal),
    }
}

fn is_failure_payload(object: &Map<String, Value>, named: Option<Signal>) -> bool {
    if named == Some(Signal::Failed) {
        return true;
    }
    if object.get("error").is_some_and(json_value_present) {
        return true;
    }
    if matches!(
        object.get("type").and_then(Value::as_str),
        Some("error" | "response.failed" | "response.incomplete")
    ) {
        return true;
    }
    matches!(response_status(object), Some("failed" | "incomplete"))
        || object
            .get("response")
            .and_then(Value::as_object)
            .and_then(|response| response.get("error"))
            .is_some_and(json_value_present)
}

fn response_status(object: &Map<String, Value>) -> Option<&str> {
    object
        .get("response")
        .and_then(Value::as_object)
        .and_then(|response| response.get("status"))
        .and_then(Value::as_str)
}

fn classify_response_metadata(object: &Map<String, Value>) -> Observation {
    match object
        .get("response")
        .and_then(Value::as_object)
        .and_then(|response| response.get("output"))
    {
        Some(output) if json_value_present(output) => Observation::Output(OutputKind::Other),
        _ => Observation::Neutral,
    }
}

fn classify_chat_choices(object: &Map<String, Value>, max_choices: usize) -> Observation {
    let choices = match object.get("choices").and_then(Value::as_array) {
        Some(choices) => choices,
        None => return Observation::Uncertain,
    };
    if choices.len() > max_choices {
        return Observation::Uncertain;
    }

    let mut uncertain = false;
    for choice in choices {
        let choice = match choice.as_object() {
            Some(choice) => choice,
            None => {
                uncertain = true;
                continue;
            }
        };

        if let Some(observation) = classify_chat_choice(choice, &mut uncertain) {
            return observation;
        }
    }

    if uncertain {
        Observation::Uncertain
    } else {
        Observation::Neutral
    }
}

fn classify_chat_choice(choice: &Map<String, Value>, uncertain: &mut bool) -> Option<Observation> {
    if let Some(text) = choice.get("text") {
        if json_value_present(text) {
            return Some(Observation::Output(OutputKind::Text));
        }
    }

    if let Some(message) = choice.get("message") {
        match classify_message(message) {
            Some(observation) => return Some(observation),
            None if !message.is_null() => *uncertain = true,
            None => {}
        }
    }

    if let Some(delta) = choice.get("delta") {
        let delta = match delta.as_object() {
            Some(delta) => delta,
            None if delta.is_null() => return None,
            None => {
                *uncertain = true;
                return None;
            }
        };

        if let Some(observation) = classify_delta(delta) {
            return Some(observation);
        }

        const KNOWN_METADATA: &[&str] = &["role"];
        if delta.iter().any(|(key, value)| {
            !KNOWN_METADATA.contains(&key.as_str()) && json_value_present(value)
        }) {
            *uncertain = true;
        }
    }

    const KNOWN_CHOICE_FIELDS: &[&str] = &[
        "index",
        "delta",
        "finish_reason",
        "logprobs",
        "text",
        "message",
    ];
    if choice.iter().any(|(key, value)| {
        !KNOWN_CHOICE_FIELDS.contains(&key.as_str()) && json_value_present(value)
    }) {
        *uncertain = true;
    }

    None
}

fn classify_message(message: &Value) -> Option<Observation> {
    let message = message.as_object()?;
    classify_delta(message)
}

fn classify_delta(delta: &Map<String, Value>) -> Option<Observation> {
    const FIELDS: &[(&str, OutputKind)] = &[
        ("tool_calls", OutputKind::ToolCall),
        ("function_call", OutputKind::ToolCall),
        ("content", OutputKind::Text),
        ("refusal", OutputKind::Refusal),
        ("reasoning", OutputKind::Reasoning),
        ("reasoning_content", OutputKind::Reasoning),
        ("audio", OutputKind::Audio),
    ];

    FIELDS.iter().find_map(|(field, kind)| {
        delta
            .get(*field)
            .filter(|value| json_value_present(value))
            .map(|_| Observation::Output(*kind))
    })
}

fn json_value_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(_) | Value::Number(_) => true,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn trim_ascii(mut input: &[u8]) -> &[u8] {
    while input.first().is_some_and(u8::is_ascii_whitespace) {
        input = &input[1..];
    }
    while input.last().is_some_and(u8::is_ascii_whitespace) {
        input = &input[..input.len() - 1];
    }
    input
}

fn has_duplicate_object_key(input: &[u8]) -> bool {
    let mut cursor = JsonCursor { input, position: 0 };
    matches!(cursor.scan_value(0), Err(ScanError::Duplicate))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanError {
    Duplicate,
    Invalid,
}

struct JsonCursor<'a> {
    input: &'a [u8],
    position: usize,
}

impl JsonCursor<'_> {
    fn scan_value(&mut self, depth: usize) -> Result<(), ScanError> {
        if depth > 128 {
            return Err(ScanError::Invalid);
        }
        self.skip_whitespace();
        match self.current() {
            Some(b'{') => self.scan_object(depth + 1),
            Some(b'[') => self.scan_array(depth + 1),
            Some(b'"') => {
                self.scan_string()?;
                Ok(())
            }
            Some(_) => {
                self.scan_primitive();
                Ok(())
            }
            None => Err(ScanError::Invalid),
        }
    }

    fn scan_object(&mut self, depth: usize) -> Result<(), ScanError> {
        self.position += 1;
        self.skip_whitespace();
        if self.current() == Some(b'}') {
            self.position += 1;
            return Ok(());
        }

        let mut keys = HashSet::new();
        loop {
            self.skip_whitespace();
            if self.current() != Some(b'"') {
                return Err(ScanError::Invalid);
            }
            let key_start = self.position;
            self.scan_string()?;
            let key: String = serde_json::from_slice(&self.input[key_start..self.position])
                .map_err(|_| ScanError::Invalid)?;
            if !keys.insert(key) {
                return Err(ScanError::Duplicate);
            }

            self.skip_whitespace();
            if self.current() != Some(b':') {
                return Err(ScanError::Invalid);
            }
            self.position += 1;
            self.scan_value(depth)?;
            self.skip_whitespace();
            match self.current() {
                Some(b',') => self.position += 1,
                Some(b'}') => {
                    self.position += 1;
                    return Ok(());
                }
                _ => return Err(ScanError::Invalid),
            }
        }
    }

    fn scan_array(&mut self, depth: usize) -> Result<(), ScanError> {
        self.position += 1;
        self.skip_whitespace();
        if self.current() == Some(b']') {
            self.position += 1;
            return Ok(());
        }

        loop {
            self.scan_value(depth)?;
            self.skip_whitespace();
            match self.current() {
                Some(b',') => self.position += 1,
                Some(b']') => {
                    self.position += 1;
                    return Ok(());
                }
                _ => return Err(ScanError::Invalid),
            }
        }
    }

    fn scan_string(&mut self) -> Result<(), ScanError> {
        if self.current() != Some(b'"') {
            return Err(ScanError::Invalid);
        }
        self.position += 1;
        while let Some(byte) = self.current() {
            match byte {
                b'"' => {
                    self.position += 1;
                    return Ok(());
                }
                b'\\' => {
                    self.position += 1;
                    if self.current().is_none() {
                        return Err(ScanError::Invalid);
                    }
                    self.position += 1;
                }
                _ => self.position += 1,
            }
        }
        Err(ScanError::Invalid)
    }

    fn scan_primitive(&mut self) {
        while let Some(byte) = self.current() {
            if byte.is_ascii_whitespace() || matches!(byte, b',' | b']' | b'}') {
                return;
            }
            self.position += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while self
            .current()
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.position += 1;
        }
    }

    fn current(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }
}
