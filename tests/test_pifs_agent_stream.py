import io
import os
import unittest
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict

from pageindex.filesystem.agent import (
    PIFSAgentStreamObserver,
    build_agent_model_settings,
    normalize_agent_stream_mode,
    normalize_reasoning_effort,
    normalize_reasoning_summary,
    pifs_agent_raw_reasoning_enabled,
    serialize_agent_final_output,
    should_disable_pifs_agent_tracing,
    should_use_openai_compatible_chat_model,
)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    document_ids: list[str]


class PIFSAgentStreamTest(unittest.TestCase):
    def raw_event(self, event_type, delta):
        return SimpleNamespace(
            type="raw_response_event",
            data=SimpleNamespace(type=event_type, delta=delta),
        )

    def test_model_stream_prints_output_and_think_deltas(self):
        output = io.StringIO()
        stream_log = []
        observer = PIFSAgentStreamObserver("model", stream_log=stream_log, output=output)

        observer.handle_event(self.raw_event("response.reasoning_summary_text.delta", "look up folder"))
        observer.handle_event(self.raw_event("response.output_text.delta", '{"answer":'))
        observer.handle_event(self.raw_event("response.output_text.delta", '"done"}'))
        observer.finish()

        printed = output.getvalue()
        self.assertIn("[llm reasoning summary stream]", printed)
        self.assertIn("look up folder", printed)
        self.assertIn("[llm final output stream]", printed)
        self.assertIn('{"answer":"done"}', printed.replace("\n", ""))
        self.assertEqual(
            stream_log,
            [
                {"kind": "output", "text": '{"answer":"done"}'},
                {"kind": "think_summary", "text": "look up folder"},
            ],
        )

    def test_tools_mode_does_not_print_model_text(self):
        output = io.StringIO()
        stream_log = []
        observer = PIFSAgentStreamObserver("tools", stream_log=stream_log, output=output)

        observer.handle_event(self.raw_event("response.output_text.delta", "hidden from tools mode"))
        observer.handle_event(self.raw_event("response.function_call_arguments.delta", '{"command":"ls /"}'))
        observer.emit_tool_call("ls /")
        observer.emit_tool_result(ok=True, output='{"ok": true}', seconds=0.001)
        observer.finish()

        printed = output.getvalue()
        self.assertNotIn("hidden from tools mode", printed)
        self.assertIn("[llm -> pifs command]", printed)
        self.assertIn("ls /", printed)
        self.assertIn("[pifs -> llm result preview]", printed)
        self.assertIn('{"ok": true}', printed)
        self.assertEqual(stream_log[0], {"kind": "tool_call", "command": "ls /"})
        self.assertEqual(stream_log[1]["kind"], "tool_result")
        self.assertEqual(stream_log[2], {"kind": "tool_args", "text": '{"command":"ls /"}'})

    def test_tool_result_preview_compacts_large_outputs(self):
        output = io.StringIO()
        observer = PIFSAgentStreamObserver("tools", output=output)

        observer.emit_tool_result(
            ok=True,
            output="\n".join(f"line {index}" for index in range(50)),
            seconds=0.001,
        )

        printed = output.getvalue()
        self.assertIn("[large PIFS result", printed)
        self.assertIn("line 0", printed)
        self.assertIn("more lines omitted from preview", printed)
        self.assertNotIn("line 49", printed)

    def test_raw_reasoning_is_not_logged_by_default_but_summary_is(self):
        output = io.StringIO()
        stream_log = []
        previous = os.environ.pop("PAGEINDEX_PIFS_AGENT_RAW_REASONING", None)
        try:
            observer = PIFSAgentStreamObserver("model", stream_log=stream_log, output=output)
            observer.handle_event(self.raw_event("response.reasoning_text.delta", "private chain"))
            observer.handle_event(
                self.raw_event("response.reasoning_summary_text.delta", "visible summary")
            )
            observer.finish()
        finally:
            if previous is not None:
                os.environ["PAGEINDEX_PIFS_AGENT_RAW_REASONING"] = previous

        printed = output.getvalue()
        self.assertNotIn("private chain", printed)
        self.assertIn("visible summary", printed)
        self.assertEqual(stream_log, [{"kind": "think_summary", "text": "visible summary"}])

    def test_raw_reasoning_requires_debug_env_flag(self):
        self.assertFalse(pifs_agent_raw_reasoning_enabled({}))
        self.assertTrue(
            pifs_agent_raw_reasoning_enabled({"PAGEINDEX_PIFS_AGENT_RAW_REASONING": "on"})
        )
        self.assertTrue(
            pifs_agent_raw_reasoning_enabled({"PAGEINDEX_PIFS_AGENT_RAW_REASONING": "TRUE"})
        )
        self.assertFalse(
            pifs_agent_raw_reasoning_enabled({"PAGEINDEX_PIFS_AGENT_RAW_REASONING": "0"})
        )

    def test_stream_mode_aliases(self):
        self.assertEqual(normalize_agent_stream_mode("think"), "model")
        self.assertEqual(normalize_agent_stream_mode("debug"), "all")
        self.assertEqual(normalize_agent_stream_mode(""), "off")
        with self.assertRaises(ValueError):
            normalize_agent_stream_mode("nope")

    def test_reasoning_settings_enable_effort_and_summary(self):
        settings = build_agent_model_settings(
            reasoning_effort="medium",
            reasoning_summary="detailed",
        )

        self.assertIsNotNone(settings)
        self.assertEqual(settings.reasoning.effort, "medium")
        self.assertEqual(settings.reasoning.summary, "detailed")
        self.assertEqual(settings.verbosity, "low")

    def test_reasoning_effort_defaults_to_visible_summary(self):
        settings = build_agent_model_settings(reasoning_effort="low")

        self.assertIsNotNone(settings)
        self.assertEqual(settings.reasoning.effort, "low")
        self.assertEqual(settings.reasoning.summary, "auto")

    def test_reasoning_and_base_url_normalization(self):
        self.assertEqual(normalize_reasoning_effort("xhigh"), "xhigh")
        self.assertIsNone(normalize_reasoning_summary("none"))
        self.assertFalse(should_use_openai_compatible_chat_model(None))
        self.assertFalse(should_use_openai_compatible_chat_model("https://api.openai.com/v1/"))
        self.assertTrue(should_use_openai_compatible_chat_model("https://example.test/v1"))
        with self.assertRaises(ValueError):
            normalize_reasoning_effort("maximum")

    def test_tracing_is_disabled_by_default_unless_env_enables_it(self):
        self.assertTrue(should_disable_pifs_agent_tracing({}))
        self.assertFalse(
            should_disable_pifs_agent_tracing({"PAGEINDEX_PIFS_AGENT_TRACING": "1"})
        )
        self.assertFalse(
            should_disable_pifs_agent_tracing({"PAGEINDEX_PIFS_AGENT_TRACING": "true"})
        )
        self.assertFalse(
            should_disable_pifs_agent_tracing({"PAGEINDEX_PIFS_AGENT_TRACING": "on"})
        )
        self.assertTrue(
            should_disable_pifs_agent_tracing({"PAGEINDEX_PIFS_AGENT_TRACING": "0"})
        )

    def test_structured_agent_output_serializes_to_json(self):
        output = serialize_agent_final_output(
            StructuredAnswer(answer="done", document_ids=["dsid_1"])
        )

        self.assertEqual(output, '{"answer":"done","document_ids":["dsid_1"]}')


if __name__ == "__main__":
    unittest.main()
