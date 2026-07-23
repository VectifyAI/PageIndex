# pageindex/tokens.py
# Shared by parser and index layers (avoids a reverse dependency).


def count_tokens(text, model=None):
    if not text:
        return 0
    import litellm
    return litellm.token_counter(model=model, text=text)
