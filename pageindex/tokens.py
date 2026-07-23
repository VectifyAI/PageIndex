# pageindex/tokens.py
# Leaf utility so both the parser and index layers can count tokens without
# the parser reaching back into pageindex.index (a reverse/horizontal
# dependency). Depends only on litellm (imported lazily — it's several
# seconds and a network fetch, and cloud-only paths never need it).


def count_tokens(text, model=None):
    if not text:
        return 0
    import litellm
    return litellm.token_counter(model=model, text=text)
