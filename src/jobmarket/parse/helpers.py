"""Shared parsing helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {key}")
    return value.strip()


def optional_str(data: dict[str, object] | None, key: str) -> str | None:
    if data is None:
        return None
    value = data.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def optional_number(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def parse_unix_epoch(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        timestamp = int(str(value))
    except ValueError:
        return None
    return datetime.fromtimestamp(timestamp, tz=UTC)
