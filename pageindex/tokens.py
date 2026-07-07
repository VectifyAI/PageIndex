# pageindex/tokens.py
# Leaf utility so both the parser and index layers can count tokens without
# the parser reaching back into pageindex.index (a reverse/horizontal
# dependency). Depends only on litellm.
import litellm


def count_tokens(text, model=None):
    if not text:
        return 0
    return litellm.token_counter(model=model, text=text)
