def remove_sqlite_extension(s: str) -> str:
    """Remove the .sqlite extension from a string."""
    if s.endswith(".sqlite"):
        return s[:-7]
    return s
