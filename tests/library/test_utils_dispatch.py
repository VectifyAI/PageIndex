import asyncio

from pageindex import backends
from pageindex.backends.base import CliBackend, CliResult
from pageindex import utils


class Recording(CliBackend):
    name = "rec"

    def build_command(self, workdir):
        return ["true"]

    def parse_output(self, result, workdir):
        return result.stdout


def make(reply):
    prompts = []

    async def run(cmd, stdin_text, timeout):
        prompts.append(stdin_text)
        return CliResult(0, reply, "")
    return Recording("x", run=run), prompts


def test_llm_acompletion_uses_cli_backend():
    backend, prompts = make("async reply")
    backends.register("test-cli/a", backend)
    assert asyncio.run(utils.llm_acompletion("test-cli/a", "hi")) == "async reply"
    assert prompts == ["hi"]


def test_llm_completion_uses_cli_backend_and_flattens_history():
    backend, prompts = make("sync reply")
    backends.register("test-cli/b", backend)
    out = utils.llm_completion("test-cli/b", "now",
                               chat_history=[{"role": "user", "content": "earlier"},
                                             {"role": "assistant", "content": "ok"}])
    assert out == "sync reply"
    assert prompts[0] == "user: earlier\n\nassistant: ok\n\nuser: now"


def test_llm_completion_finish_reason_for_cli():
    backend, _ = make("r")
    backends.register("test-cli/c", backend)
    assert utils.llm_completion("test-cli/c", "p", return_finish_reason=True) == ("r", "finished")


def test_non_cli_model_still_goes_to_litellm(monkeypatch):
    import litellm
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)

        class R:
            class choices:
                pass
        msg = type("M", (), {"content": "lite"})()
        choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
        return type("Resp", (), {"choices": [choice]})()
    monkeypatch.setattr(litellm, "completion", fake_completion)
    assert utils.llm_completion("gpt-5.6-luna", "p") == "lite"
    assert seen["model"] == "openai/gpt-5.6-luna"
