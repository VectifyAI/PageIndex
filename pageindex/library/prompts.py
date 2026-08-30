"""Prompt templates for the summary/digest tiers, loaded from prompts.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

FALLBACK_PROFILE = "nonfiction"


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(Path(__file__).with_name("prompts.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["profiles"]


PROFILES = tuple(_load().keys())


def _tier(profile: str, tier: str) -> dict:
    profiles = _load()
    if profile not in profiles:
        raise KeyError(f"Unknown profile {profile!r}; known: {sorted(profiles)}")
    block = profiles[profile].get(tier) or profiles[FALLBACK_PROFILE][tier]
    fallback = profiles[FALLBACK_PROFILE][tier]
    return {**fallback, **block}


def max_words(profile: str, tier: str) -> int:
    return int(_tier(profile, tier)["max_words"])


def render(profile: str, tier: str, kind: str, **fields) -> str:
    block = _tier(profile, tier)
    template = block[kind]
    return template.format(max_words=block["max_words"], **fields)
