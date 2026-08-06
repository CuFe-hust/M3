"""Shared policy for withholding configured secrets from persisted evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


REDACTED_ARGUMENT = "<redacted-argument>"
_SENSITIVE_WORDS = frozenset(
    {
        "auth",
        "authorization",
        "credential",
        "credentials",
        "key",
        "password",
        "passwd",
        "private",
        "secret",
        "token",
    }
)
_COMPOUND_SENSITIVE_NAMES = frozenset(
    {"apikey", "accesskey", "clientsecret", "privatekey", "signingkey"}
)
_ASSIGNMENT = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_.-]*)\s*(?P<separator>[:=])\s*(?P<value>[^\s]+)"
)


def is_secret_name(name: str) -> bool:
    """Classify delimiter-separated secret-like names without substring false positives."""
    words = [word for word in re.split(r"[^a-z0-9]+", name.casefold()) if word]
    compact = "".join(words)
    return bool(set(words) & _SENSITIVE_WORDS) or compact in _COMPOUND_SENSITIVE_NAMES


def redact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return only non-sensitive entries, excluding secret names as well as values."""
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        text_key = str(key)
        if is_secret_name(text_key):
            continue
        if isinstance(value, Mapping):
            result[text_key] = redact_mapping(value)
        elif isinstance(value, list):
            result[text_key] = [redact_value(item) for item in value]
        elif isinstance(value, tuple):
            result[text_key] = tuple(redact_value(item) for item in value)
        else:
            result[text_key] = value
    return result


def redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value


def secret_argument_indexes(argv: Iterable[str], explicit_positions: Iterable[int] = ()) -> set[int]:
    """Identify sensitive option/value pairs and explicitly configured value positions."""
    arguments = list(argv)
    indexes = {index for index in explicit_positions if 0 <= index < len(arguments)}
    for index, argument in enumerate(arguments):
        option, separator, _ = argument.partition("=")
        if option.startswith("-") and is_secret_name(option.lstrip("-")):
            indexes.add(index)
            if not separator and index + 1 < len(arguments):
                indexes.add(index + 1)
        elif ":" in argument and is_secret_name(argument.split(":", 1)[0]):
            indexes.add(index)
    return indexes


def redact_argv(argv: Iterable[str], explicit_positions: Iterable[int] = ()) -> list[str]:
    arguments = list(argv)
    indexes = secret_argument_indexes(arguments, explicit_positions)
    return [REDACTED_ARGUMENT if index in indexes else argument for index, argument in enumerate(arguments)]


@dataclass(frozen=True)
class TextRedactor:
    """Redacts known secret names/values before text is committed to a log file."""

    values: tuple[str, ...]
    names: tuple[str, ...]

    @classmethod
    def from_execution(
        cls,
        environment: Mapping[str, str],
        argv: Iterable[str],
        explicit_positions: Iterable[int] = (),
    ) -> "TextRedactor":
        arguments = list(argv)
        argument_indexes = secret_argument_indexes(arguments, explicit_positions)
        values = [value for key, value in environment.items() if is_secret_name(key) and value]
        for index in argument_indexes:
            argument = arguments[index]
            if argument:
                values.append(argument)
                option, separator, value = argument.partition("=")
                if separator and option.startswith("-") and value:
                    values.append(value)
        names = [key for key in environment if is_secret_name(key)]
        names.extend(arguments[index] for index in argument_indexes if arguments[index].startswith("-"))
        return cls(
            values=tuple(sorted(set(values), key=len, reverse=True)),
            names=tuple(sorted(set(names), key=len, reverse=True)),
        )

    def redact(self, text: str) -> str:
        redacted = text
        for value in self.values:
            redacted = redacted.replace(value, REDACTED_ARGUMENT)
        for name in self.names:
            redacted = re.sub(re.escape(name), "<redacted-name>", redacted, flags=re.IGNORECASE)
        return _ASSIGNMENT.sub(_redact_secret_assignment, redacted)


def _redact_secret_assignment(match: re.Match[str]) -> str:
    if not is_secret_name(match.group("name")):
        return match.group(0)
    return f"<redacted-name>{match.group('separator')}{REDACTED_ARGUMENT}"
