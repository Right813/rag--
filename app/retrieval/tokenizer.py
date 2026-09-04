import re


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.findall(text or ""):
        value = match.casefold()
        if all("\u4e00" <= char <= "\u9fff" for char in value):
            tokens.extend(value)
        else:
            tokens.append(value)
    return tokens
