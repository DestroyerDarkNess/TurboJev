#!/usr/bin/env python3
"""Offline Rust source sanity checks.

This is intentionally not a compiler replacement. It catches merge markers,
unbalanced delimiters, unterminated comments/strings and accidental empty
source files when a Rust toolchain is unavailable.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {")": "(", "]": "[", "}": "{"}
OPEN = set(PAIRS.values())


def scan(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if not text.strip():
        return [f"{path}: empty Rust source"]
    for marker in ("<<<<<<<", "=======", ">>>>>>>"):
        if marker in text:
            errors.append(f"{path}: merge-conflict marker {marker!r}")

    stack: list[tuple[str, int]] = []
    i = 0
    line = 1
    block_depth = 0
    state = "code"
    raw_hashes = 0

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch == "\n":
            line += 1

        if state == "line_comment":
            if ch == "\n":
                state = "code"
            i += 1
            continue

        if state == "block_comment":
            if ch == "/" and nxt == "*":
                block_depth += 1
                i += 2
                continue
            if ch == "*" and nxt == "/":
                block_depth -= 1
                i += 2
                if block_depth == 0:
                    state = "code"
                continue
            i += 1
            continue

        if state == "string":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "code"
            i += 1
            continue

        if state == "char":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "code"
            i += 1
            continue

        if state == "raw":
            if ch == '"':
                suffix = "#" * raw_hashes
                if text.startswith('"' + suffix, i):
                    i += 1 + raw_hashes
                    state = "code"
                    continue
            i += 1
            continue

        # code
        if ch == "/" and nxt == "/":
            state = "line_comment"
            i += 2
            continue
        if ch == "/" and nxt == "*":
            state = "block_comment"
            block_depth = 1
            i += 2
            continue
        if ch == '"':
            state = "string"
            i += 1
            continue
        if ch == "'":
            # Lifetime syntax such as 'a is not a char literal. Treat as a char
            # only when a closing quote is nearby or it is an escaped char.
            j = i + 1
            if j < len(text) and text[j] == "\\":
                state = "char"
            elif j + 1 < len(text) and text[j + 1] == "'":
                state = "char"
            i += 1
            continue
        if ch == "r":
            j = i + 1
            hashes = 0
            while j < len(text) and text[j] == "#":
                hashes += 1
                j += 1
            if j < len(text) and text[j] == '"':
                state = "raw"
                raw_hashes = hashes
                i = j + 1
                continue

        if ch in OPEN:
            stack.append((ch, line))
        elif ch in PAIRS:
            if not stack or stack[-1][0] != PAIRS[ch]:
                errors.append(f"{path}:{line}: unmatched {ch!r}")
            else:
                stack.pop()
        i += 1

    if state == "block_comment":
        errors.append(f"{path}: unterminated block comment")
    elif state in {"string", "raw"}:
        errors.append(f"{path}: unterminated string literal")
    for token, token_line in reversed(stack):
        errors.append(f"{path}:{token_line}: unclosed {token!r}")
    return errors


def main() -> int:
    files = sorted((ROOT / "crates").rglob("*.rs"))
    errors = [error for path in files for error in scan(path)]
    if errors:
        print("Rust source sanity check failed:")
        for error in errors:
            print(" -", error)
        return 1
    print(f"OK: {len(files)} Rust source files pass offline delimiter/conflict sanity checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
