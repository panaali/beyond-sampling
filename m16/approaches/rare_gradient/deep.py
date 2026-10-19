"""Shim so the unchanged truthc_copy.py imports (its models() is not used by T372)."""


def models():
    import k3src
    return k3src.models()
