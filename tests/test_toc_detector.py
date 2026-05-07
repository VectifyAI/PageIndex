import importlib


page_index = importlib.import_module("pageindex.page_index")


def test_single_page_toc_prompt_distinguishes_content_from_toc(monkeypatch):
    captured = {}

    def fake_completion(model, prompt):
        captured["model"] = model
        captured["prompt"] = prompt
        return '{"thinking": "structured content, not a toc", "toc_detected": "no"}'

    monkeypatch.setattr(page_index, "llm_completion", fake_completion)

    result = page_index.toc_detector_single_page(
        "1. Scope\nThis policy applies to all staff.\n\n"
        "2. Requirements\nUsers must rotate secrets.\n\n"
        "3. Exceptions\nExceptions require approval.",
        model="test-model",
    )

    assert result == "no"
    assert captured["model"] == "test-model"

    prompt = captured["prompt"].lower()
    assert "single-page" in prompt
    assert "actual document content" in prompt
    assert "numbered sections" in prompt
    assert "true table of contents" in prompt
    assert "references to content elsewhere" in prompt
