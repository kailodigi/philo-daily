#!/usr/bin/env python3
"""Validate brief JSON before any HTML or index/archive files are generated."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


SOCIAL_SHORTFALL_NOTICE = "今日可靠社媒趋势不足"
UNKNOWN_SOURCES = {
    "unknown",
    "unknown source",
    "n/a",
    "na",
    "none",
    "未知",
    "未知来源",
    "不详",
}
CATEGORY_FIELDS = (
    ("金融", "global_finance", 5),
    ("AI", "ai_industry", 5),
    ("半导体", "semiconductors", 2),
    ("社媒", "social_trends", 0),
)


def normalize_url(value: str) -> str:
    parts = urlsplit(value.strip())
    path = re.sub(r"/+$", "", parts.path) or "/"
    host = parts.netloc.lower()
    for prefix in ("www.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    return urlunsplit(("https", host, path, "", ""))


def source_urls_from_candidates(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        if isinstance(payload.get("candidates"), list):
            payload = payload["candidates"]
        elif isinstance(payload.get("search_results"), list):
            payload = payload["search_results"]
    if not isinstance(payload, list):
        raise ValueError("verified source file must contain a list of candidates/results")
    urls = {
        normalize_url(str(item.get("source_url") or item.get("url") or ""))
        for item in payload
        if isinstance(item, dict) and (item.get("source_url") or item.get("url"))
    }
    return {url for url in urls if urlsplit(url).netloc}


def _field(item: dict[str, Any], primary: str, alias: str) -> Any:
    value = item.get(primary)
    return value if value not in (None, "") else item.get(alias)


def _iter_items(brief: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for label, field_name, _minimum in CATEGORY_FIELDS:
        values = brief.get(field_name, [])
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    yield label, item


def validate_brief_payload(
    brief: dict[str, Any],
    *,
    brief_date: str | None = None,
    verified_source_urls: set[str] | None = None,
) -> list[str]:
    """Return all quality errors; an empty list means the payload may be rendered."""
    errors: list[str] = []
    requested_date = brief_date or str(brief.get("date", ""))
    try:
        reference_day = date.fromisoformat(requested_date)
    except ValueError:
        return ["brief date must use YYYY-MM-DD"]
    if brief.get("date") != requested_date:
        errors.append("brief date does not match the requested date")

    for label, field_name, minimum in CATEGORY_FIELDS:
        values = brief.get(field_name)
        if not isinstance(values, list):
            errors.append(f"{label}: {field_name} must be a list")
            continue
        if len(values) < minimum:
            errors.append(f"{label}: expected at least {minimum} items, found {len(values)}")

    social = brief.get("social_trends", [])
    social_count = len(social) if isinstance(social, list) else 0
    notice = brief.get("social_limit_notice")
    if social_count > 5:
        errors.append(f"社媒: expected at most 5 items, found {social_count}")
    elif social_count < 3 and notice != SOCIAL_SHORTFALL_NOTICE:
        errors.append(f"社媒: fewer than 3 reliable items requires notice '{SOCIAL_SHORTFALL_NOTICE}'")
    elif 3 <= social_count <= 5 and notice not in (None, ""):
        errors.append("社媒: shortfall notice must be empty when 3–5 reliable items exist")

    normalized_verified = (
        {normalize_url(url) for url in verified_source_urls}
        if verified_source_urls is not None
        else None
    )
    for label, item in _iter_items(brief):
        title = str(_field(item, "headline", "title") or "").strip()
        status = str(_field(item, "status", "event_type") or "").strip()
        summary = str(_field(item, "what_happened", "summary") or "").strip()
        importance = str(_field(item, "why_important", "importance") or "").strip()
        source_name = str(item.get("source_name") or item.get("source") or "").strip()
        source_url = str(item.get("source_url") or "").strip()
        published_time = str(_field(item, "published_at", "published_time") or "").strip()
        prefix = f"{label}/{title or '<无标题>'}"

        for field_label, value in (
            ("title", title),
            ("event_type", status),
            ("summary", summary),
            ("importance", importance),
            ("source", source_name),
            ("published_time", published_time),
        ):
            if not value:
                errors.append(f"{prefix}: missing {field_label}")
        if status not in {"新增", "延续"}:
            errors.append(f"{prefix}: status must be 新增 or 延续")
        if source_name.casefold() in UNKNOWN_SOURCES:
            errors.append(f"{prefix}: unknown source is forbidden")
        parsed_url = urlsplit(source_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            errors.append(f"{prefix}: source_url must be an absolute HTTP(S) URL")
        elif normalized_verified is not None and normalize_url(source_url) not in normalized_verified:
            errors.append(f"{prefix}: source_url is absent from verified search candidates")

        match = re.search(r"\d{4}-\d{2}-\d{2}", published_time)
        if not match:
            errors.append(f"{prefix}: published_time lacks an ISO date")
            continue
        published_day = date.fromisoformat(match.group(0))
        age_days = (reference_day - published_day).days
        if age_days < 0:
            errors.append(f"{prefix}: published_time is in the future")
        elif age_days > 6:
            errors.append(f"{prefix}: published_time is outside the 7-day window")
        elif age_days >= 2 and status != "延续":
            errors.append(f"{prefix}: news at or beyond 48h must be marked 延续")
    return errors


def assert_brief_quality(
    brief: dict[str, Any],
    *,
    brief_date: str | None = None,
    verified_source_urls: set[str] | None = None,
) -> None:
    errors = validate_brief_payload(
        brief,
        brief_date=brief_date,
        verified_source_urls=verified_source_urls,
    )
    if errors:
        raise ValueError("Quality gate failed:\n- " + "\n- ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("--date", dest="brief_date")
    parser.add_argument("--verified-sources", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        brief = json.loads(args.brief_json.read_text(encoding="utf-8"))
        verified_urls = None
        if args.verified_sources:
            source_payload = json.loads(args.verified_sources.read_text(encoding="utf-8"))
            verified_urls = source_urls_from_candidates(source_payload)
        assert_brief_quality(
            brief,
            brief_date=args.brief_date,
            verified_source_urls=verified_urls,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"OK quality gate: {args.brief_json} "
        f"(verified_sources={len(verified_urls) if verified_urls is not None else 'not supplied'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
