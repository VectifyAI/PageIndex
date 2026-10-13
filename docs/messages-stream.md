# Messages results and streaming

`client.chat(..., protocol="messages")` runs document tools through the
Anthropic SDK. One call returns one logical message. Its `content` contains
the visible blocks from all model turns, in order, followed by any client
tool calls that the runner did not execute. Calls with recorded tool results
are internal: they appear in the continuation history, not in this content.

## Streaming

```python
stream = client.chat("Read the document", protocol="messages",
                     model="anthropic/claude-sonnet-5-5", stream=True)
try:
    for event in stream:
        if event["type"] == "content_block_delta":
            delta = event["delta"]
            if delta["type"] == "text_delta":
                print(delta["text"], end="", flush=True)
    result = stream.get_final_result()
finally:
    stream.close()
```

The return type is `pageindex.MessagesStream`, a synchronous iterator of JSON
event dictionaries. It emits one `message_start`, content-block events with
indices into the logical message, one final `message_delta`, and one
`message_stop`. Text, thinking, citations, and server-tool blocks retain their
native event shapes. SDK convenience events such as `text` and `input_json`
are not wire events. Client `tool_use` blocks are held until their execution
state is known; unexecuted calls are emitted at the end.

An HTTP adapter can serialize each event without composing messages:

```python
import json

for event in stream:
    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
```

The official Anthropic `messages.stream()` consumer can assemble this stream
into one message. The terminal `message_stop` has a PageIndex extension,
`result`, containing the same dictionary as the non-streaming API. A raw SSE
consumer can read that extension; the official helper's `get_final_message()`
only assembles the native message fields. Python callers can use
`stream.get_final_result()` to drain the stream, or inspect `stream.result`
after exhaustion, failure, or explicit close. Before then, `result` is None.
The opening usage reports zero completed-turn input/output tokens; aggregate
counts arrive in the final delta. Optional cache fields start unset so a
missing total cannot leave a first-turn count in the official accumulator.

## Final result

Alongside the message fields, the result contains:

- `messages`: the new assistant/tool-result turns to append to the original
  input. It preserves model turns and executed tool results, including error
  results. Unexecuted tool calls and incomplete model turns are excluded from
  this appendable history; partial visible content remains in `content`.
- `run.status`: `completed`, `max_turns`, `failed`, or `cancelled`.
- `run.turns`: the number of model messages observed, including a partial
  message if the stream stopped during generation.
- `run.model_stop_reason`: the last model message's native stop reason.
- `run.tool_calls`: each client tool call, with its original `id`, `name`,
  `input`, a `status`, and the recorded `tool_result` in `result`, or null.
  Status is `completed`, `failed` (an error result), `not_executed`, or
  `unknown` (execution was interrupted before its result was recorded).
- `run.can_resume`: false if local tool execution is uncertain. In that case,
  reconcile the affected calls before retrying; the transcript alone cannot
  rule out duplicate effects.
- `run.usage_complete`: whether all exposed counters are known and the run
  ended normally or at its turn limit.
- `run.error`: the execution error text, or null.

`completed` means the runner stopped according to the model's stop reason.
The native reason still distinguishes `end_turn`, `max_tokens`, `refusal`,
and other model outcomes. At the iteration limit, `run.status` is `max_turns`
and the logical message's `stop_reason` is `pause_turn`: the agent has paused
after executing any tools the runner dispatched. The last model's
`tool_use` reason remains in `run.model_stop_reason`; it does not ask the
caller to execute those tools again.

```python
history = [{"role": "user", "content": "Read the document"}]
result = client.chat(history, protocol="messages", model=model, max_turns=1)
if result["run"]["can_resume"]:
    history += result["messages"]
    if result["run"]["status"] == "max_turns":
        result = client.chat(history, protocol="messages", model=model)
```

Keep the model, instructions, document scope, and folder scope consistent
when continuing. Append `messages`, not the logical message's `content`:
only the transcript includes tools that have already run.

## Usage

`usage` sums each model message's final input, output, cache-read, and
cache-creation counters. Within one message, successive usage deltas are
cumulative and replace earlier values. Cache-creation TTL buckets are also
summed when present. A missing counter in any observed turn makes that total
null, never zero. An interrupted message's output count is unknown.

After failure or cancellation, counters cover only the observed messages;
`run.usage_complete` is false even when some counts are available. A request
that failed before returning a message may have incurred additional usage.
These counters do not constitute the per-call measurement interface tracked
by [#557](https://github.com/VectifyAI/PageIndex/issues/557).

Anthropic `input_tokens` excludes cache reads and writes. Total input is
`input_tokens + cache_read_input_tokens + cache_creation_input_tokens` when
all three are known. Nulls are PageIndex's missing-measurement extension;
consumers requiring non-null native usage must check `run.usage_complete`.
The authoritative counters are in `result.usage`. Official helpers may retain
the opening zero for an unknown input total, and do not accumulate cache TTL
buckets from a delta; those quantities must be read from the result extension.

## Failure and close

Iteration raises on execution failure. `stream.result` retains the collected
message, transcript, tool states, and `run.status="failed"`. A
`PageIndexAPIError` raised during execution also carries this dictionary as
`error.result`, including for non-streaming calls. Setup errors before a
runner is constructed do not have a run result. Failed streams do not emit a
successful terminal message. An HTTP adapter may include `error.result` in
its protocol error response as an extension; successful `message_stop.result`
needs no adapter-specific processing.

Call `close()` on the consuming thread when abandoning a stream. It releases
the current response and prevents further local iterations. Closing before
iteration starts makes no model request. A run already finalized retains its
result; otherwise close records `cancelled` and a null message stop reason.
Breaking a loop alone does not close the iterator. An already dispatched
provider request or external tool side effect cannot be certified cancelled
by closing a Python iterator. This API does not add a whole-run deadline.

These are Messages semantics. Responses uses its own output-item lifecycle
and `items` transcript; Chat Completions exposes an answer without tool
history. Their stream and final-result contracts are unchanged.
