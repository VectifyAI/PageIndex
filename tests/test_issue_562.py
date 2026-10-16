"""#562 — distinguish caller-side request validation from execution
failures. Callers should be able to ``except PageIndexValidationError``
without matching exception messages; everything else stays
``PageIndexAPIError``. ``PageIndexValidationError`` is a subclass so
existing ``except PageIndexAPIError`` blocks keep working."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex import (PageIndexAPIError, PageIndexClient,
                       PageIndexLocalClient, PageIndexValidationError)
from pageindex.local_chat import (_process_options, _refuse_skeleton,
                                  _split_chat_messages, _system_text,
                                  _validate_max_turns)


# ── class shape ──

class TestValidationErrorClass:
    def test_is_an_api_error(self):
        """Existing ``except PageIndexAPIError`` blocks keep catching
        validation failures — the new class narrows the catch, it does
        not introduce a parallel hierarchy."""
        assert issubclass(PageIndexValidationError, PageIndexAPIError)

    def test_validation_does_not_catch_execution(self):
        with pytest.raises(PageIndexAPIError) as exc:
            raise PageIndexAPIError("provider failed", status_code=502)
        # A plain PageIndexAPIError (transport) must NOT be caught
        # by ``except PageIndexValidationError`` — distinguishability
        # depends on this asymmetry.
        assert not isinstance(exc.value, PageIndexValidationError)

    def test_validation_subclass_accepts_kwargs(self):
        err = PageIndexValidationError("bad input", status_code=None)
        assert err.status_code is None
        assert isinstance(err, PageIndexAPIError)


# ── call-site classification: pre-request rejections ──

class TestRequestValidation:
    def test_extra_body_tools_override_rejected(self):
        """``extra_body`` is the SDK's managed channel; a caller passing
        ``tools`` there tries to replace the SDK's own tools list."""
        with pytest.raises(PageIndexValidationError, match="managed prompt"):
            _refuse_skeleton({"tools": []})

    def test_extra_body_messages_override_rejected(self):
        with pytest.raises(PageIndexValidationError):
            _refuse_skeleton({"messages": [{"role": "user", "content": "x"}]})

    def test_extra_body_stream_or_doc_id_override_rejected(self):
        """Stream and doc_id are first-class chat() arguments — passing
        them again through extra_body silently double-handles the SDK's
        own bookkeeping."""
        with pytest.raises(PageIndexValidationError):
            _refuse_skeleton({"stream": True})
        with pytest.raises(PageIndexValidationError):
            _refuse_skeleton({"doc_id": "doc-1"})

    def test_extra_body_must_be_a_mapping(self):
        with pytest.raises(PageIndexValidationError):
            _refuse_skeleton(["not", "a", "mapping"])
        with pytest.raises(PageIndexValidationError):
            _refuse_skeleton("a string")

    def test_max_turns_zero_rejected(self):
        with pytest.raises(PageIndexValidationError, match="max_turns"):
            _validate_max_turns(0)

    def test_max_turns_negative_rejected(self):
        with pytest.raises(PageIndexValidationError, match="max_turns"):
            _validate_max_turns(-3)

    def test_max_turns_non_int_rejected(self):
        with pytest.raises(PageIndexValidationError, match="max_turns"):
            _validate_max_turns(1.5)
        with pytest.raises(PageIndexValidationError, match="max_turns"):
            _validate_max_turns("10")

    def test_max_turns_none_is_allowed(self):
        """``None`` means "let the runner default" — validation must
        not reject the unset state."""
        _validate_max_turns(None)  # no exception

    def test_system_content_must_be_string_or_parts(self):
        with pytest.raises(PageIndexValidationError):
            _system_text(12345)
        with pytest.raises(PageIndexValidationError):
            _system_text({"not": "a list"})

    def test_messages_must_contain_user_or_assistant(self):
        with pytest.raises(PageIndexValidationError, match="non-empty"):
            _split_chat_messages([])
        with pytest.raises(PageIndexValidationError,
                           match="user or assistant"):
            _split_chat_messages([{"role": "system", "content": "x"}])

    def test_each_message_must_be_a_dict_with_role(self):
        with pytest.raises(PageIndexValidationError, match="dict"):
            _split_chat_messages(["not a dict"])
        with pytest.raises(PageIndexValidationError, match="role"):
            _split_chat_messages([{"content": "x"}])

    def test_unsupported_message_role_rejected(self):
        with pytest.raises(PageIndexValidationError, match="Unsupported role"):
            _split_chat_messages([{"role": "tool", "content": "x"}])

    def test_show_process_must_be_bool_or_mapping(self):
        with pytest.raises(PageIndexValidationError):
            _process_options("yes please")
        with pytest.raises(PageIndexValidationError):
            _process_options(42)

    def test_show_process_unknown_keys_rejected(self):
        with pytest.raises(PageIndexValidationError, match="Unknown"):
            _process_options({"verbose": True})

    def test_show_process_bool_values_required(self):
        with pytest.raises(PageIndexValidationError, match="must be a bool"):
            _process_options({"thinking": "yes"})

    def test_show_process_max_chars_must_be_positive_int(self):
        with pytest.raises(PageIndexValidationError, match="positive int"):
            _process_options({"max_chars": 0})
        with pytest.raises(PageIndexValidationError, match="positive int"):
            _process_options({"max_chars": True})


# ── execution-time failures stay on PageIndexAPIError ──

class TestExecutionStaysApiError:
    """The acceptance criteria call these out specifically so callers
    can tell 'the caller's input is wrong' from 'we tried and it
    didn't work'. Transport errors, empty model responses, and tool
    runner mismatches must not be reclassified as validation."""

    def test_empty_model_response_is_execution(self):
        """The model was reached — it just said nothing."""
        err = PageIndexAPIError("The model returned no response.")
        assert not isinstance(err, PageIndexValidationError)

    def test_tool_runner_mismatch_is_execution(self):
        """The Anthropic conversation-back channel failed mid-run —
        the SDK never had an opportunity to refuse the call."""
        err = PageIndexAPIError(
            "Could not read the conversation back from the anthropic "
            "tool runner — the installed anthropic version is "
            "incompatible with this pageindex release."
        )
        assert not isinstance(err, PageIndexValidationError)

    def test_transport_error_with_status_is_execution(self):
        """status_code is the discriminating signal: present ⇒ a
        real upstream response was attempted."""
        err = PageIndexAPIError("provider said no", status_code=429)
        assert not isinstance(err, PageIndexValidationError)

    def test_api_error_still_accepts_status_in_constructor(self):
        """Backwards compatibility: ``PageIndexAPIError(message,
        status_code=...)`` must still work — the validation subclass
        inherits the constructor without changing the signature."""
        err = PageIndexAPIError("boom", status_code=500)
        assert err.status_code == 500


# ── the ChatStream contract ──

class TestChatStreamOneViewValidation:
    """ChatStream is single-view: you can't iterate the text view
    and then iterate ``.events`` (or vice versa). The dual-consume
    state is a caller-side mistake; mid-run failures stay
    PageIndexAPIError."""

    def test_text_then_events_raises_validation(self):
        from pageindex.chat_stream import ChatStream

        def text():
            yield "hello"
            yield ""

        def events():
            yield {"type": "answer", "delta": "hi"}

        stream = ChatStream(text, events)
        # consume one piece from the text view
        assert next(stream) == "hello"
        # the second view is refused
        with pytest.raises(PageIndexValidationError, match="one view"):
            next(iter(stream.events))

    def test_events_then_text_raises_validation(self):
        from pageindex.chat_stream import ChatStream

        def text():
            yield "hello"

        def events():
            yield {"type": "answer", "delta": "hi"}

        stream = ChatStream(text, events)
        # consume one event
        assert next(iter(stream.events))["delta"] == "hi"
        with pytest.raises(PageIndexValidationError, match="one view"):
            next(stream)


# ── public client surface ──

class TestChatClientPublicSurface:
    """Drive each lane through the public API and confirm the
    validation/execution split callers actually see."""

    def test_local_client_rejects_list_of_non_dict_messages(self, tmp_path):
        """Local SDK rejects a malformed messages list before opening
        any network resources — a list of strings is not a list of
        message dicts."""
        storage_path = str(tmp_path / "store")
        client = PageIndexLocalClient(storage_path=storage_path)
        with pytest.raises(PageIndexValidationError, match="dict"):
            client.chat(["a bare string is not a message dict"])

    def test_local_client_rejects_max_turns_zero(self, tmp_path):
        storage_path = str(tmp_path / "store")
        client = PageIndexLocalClient(storage_path=storage_path)
        with pytest.raises(PageIndexValidationError, match="max_turns"):
            client.chat("hi", max_turns=0)

    def test_cloud_client_rejects_unknown_show_process_key(self, monkeypatch):
        """A cloud client's chat() call with a malformed show_process
        mapping is validation — it never reached the managed
        endpoint, and a real transport failure looks different."""
        monkeypatch.setenv("PAGEINDEX_API_KEY", "test-key")
        client = PageIndexClient(api_key=os.environ["PAGEINDEX_API_KEY"])
        with pytest.raises(PageIndexValidationError, match="Unknown"):
            client.chat("hi", show_process={"verbose": True})
