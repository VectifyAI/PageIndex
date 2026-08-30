import pytest

from pageindex.library.config import LibraryConfig
from pageindex.library import prompts


def test_config_defaults_from_env_home(home):
    cfg = LibraryConfig.load()
    assert cfg.home == home
    assert cfg.storage_path == home / ".pageindex"
    assert cfg.digests_dir == home / "digests"
    assert cfg.index_model == "claude-cli/sonnet"
    assert cfg.digest_model == "claude-cli/sonnet"
    assert cfg.profile == "nonfiction"
    assert cfg.max_leaf_pages == 40


def test_config_reads_library_yaml(home):
    (home / "library.yaml").write_text(
        "index_model: codex-cli/gpt-5.6-luna\ndigest_model: claude-cli/opus\nprofile: diary\n")
    cfg = LibraryConfig.load()
    assert cfg.index_model == "codex-cli/gpt-5.6-luna"
    assert cfg.digest_model == "claude-cli/opus"
    assert cfg.profile == "diary"


def test_config_rejects_unknown_keys(home):
    (home / "library.yaml").write_text("modle: x\n")
    with pytest.raises(ValueError, match="modle"):
        LibraryConfig.load()


def test_explicit_home_wins_over_env(home, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    assert LibraryConfig.load(str(other)).home == other


def test_profiles_exist():
    assert set(prompts.PROFILES) >= {"nonfiction", "diary"}


@pytest.mark.parametrize("profile", ["nonfiction", "diary"])
@pytest.mark.parametrize("tier", ["summary", "digest"])
def test_every_profile_tier_renders_leaf_and_parent(profile, tier):
    leaf = prompts.render(profile, tier, "leaf", book="B", title="T", text="TEXT")
    parent = prompts.render(profile, tier, "parent", book="B", title="T", intro="I",
                            children="[]")
    assert "TEXT" in leaf and "T" in leaf
    assert "[]" in parent
    assert str(prompts.max_words(profile, tier)) in leaf


def test_summary_tier_asks_for_json_and_digest_for_markdown():
    assert '"summary"' in prompts.render("nonfiction", "summary", "leaf", book="B", title="T", text="x")
    assert "Markdown" in prompts.render("nonfiction", "digest", "leaf", book="B", title="T", text="x")


def test_description_template():
    out = prompts.render("nonfiction", "description", "doc", book="B", structure="{...}")
    assert "{...}" in out


def test_diary_falls_back_to_nonfiction_for_missing_keys():
    # diary overrides summary.leaf only; description must still render
    assert prompts.render("diary", "description", "doc", book="B", structure="S")


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        prompts.render("poetry", "summary", "leaf", book="B", title="T", text="x")
