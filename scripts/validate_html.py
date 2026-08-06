#!/usr/bin/env python3
"""Fail closed when generated HTML is incomplete, invalid, or internally broken."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup


DAILY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.html$")
PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|PLACEHOLDER|LOREM IPSUM|INSERT HERE)\b|待填(?:写|充)?|示例内容|\{\{[^}]+\}\}",
    re.IGNORECASE,
)
REQUIRED_DAILY_SECTIONS = {
    "core",
    "signals",
    "finance",
    "ai",
    "semiconductors",
    "social",
    "insight",
    "tomorrow",
    "tracking",
}
SOCIAL_SHORTFALL_NOTICE = "今日可靠社媒趋势不足"


def section_shortfall_notice(count: int, label: str) -> str:
    return (
        f"今日公开来源中仅筛选出 {count} 条符合时效、来源与去重要求的"
        f"高可信{label}资讯，未使用旧闻或低可信来源补足数量。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--site-root", type=Path, default=Path("."))
    parser.add_argument(
        "--public-path",
        help="Public relative path for a staged candidate, e.g. 2026-08-04.html",
    )
    return parser.parse_args()


def check_internal_links(
    soup: BeautifulSoup, source: Path, site_root: Path, public_path: str | None
) -> list[str]:
    errors: list[str] = []
    base_relative = Path(public_path) if public_path else source.resolve().relative_to(site_root)
    base_dir = base_relative.parent

    for element in soup.select("a[href], link[href]"):
        href = element.get("href", "").strip()
        if not href:
            errors.append("empty href")
            continue
        parsed = urlsplit(href)
        if parsed.scheme in {"http", "https", "mailto", "tel"} or href.startswith("//"):
            continue
        if parsed.scheme or href.lower().startswith("javascript:"):
            errors.append(f"unsupported link: {href}")
            continue
        if not parsed.path:
            if parsed.fragment and not soup.find(id=unquote(parsed.fragment)):
                errors.append(f"missing local anchor: {href}")
            continue
        if parsed.path.startswith("/"):
            errors.append(f"root-absolute link is unsafe for project Pages: {href}")
            continue
        target = (site_root / base_dir / unquote(parsed.path)).resolve()
        try:
            target.relative_to(site_root)
        except ValueError:
            errors.append(f"link escapes site root: {href}")
            continue
        if not target.exists():
            errors.append(f"missing internal target: {href}")
    return errors


def check_daily(soup: BeautifulSoup) -> list[str]:
    errors: list[str] = []
    sections = {node.get("data-section") for node in soup.select("[data-section]")}
    missing_sections = REQUIRED_DAILY_SECTIONS - sections
    if missing_sections:
        errors.append("missing daily sections: " + ", ".join(sorted(missing_sections)))

    finance = soup.select('[data-section="finance"] [data-news-item]')
    ai_items = soup.select('[data-section="ai"] [data-news-item]')
    for section_id, items, label in (
        ("finance", finance, "金融"),
        ("ai", ai_items, " AI "),
    ):
        count = len(items)
        if not 3 <= count <= 5:
            errors.append(f"expected 3–5 {section_id} items, found {count}")
        notice = soup.select_one(f'[data-section="{section_id}"] .section-notice')
        if 3 <= count < 5:
            notice_text = notice.get_text(" ", strip=True) if notice else ""
            if notice_text != section_shortfall_notice(count, label):
                errors.append(
                    f"{section_id} shortfall requires the verified-source notice"
                )
        elif count == 5 and notice:
            errors.append(f"unexpected {section_id} shortfall notice with 5 items")
    social_items = soup.select('[data-section="social"] [data-news-item]')
    social_notice = soup.select_one('[data-section="social"] .section-notice')
    if len(social_items) > 5:
        errors.append(f"expected at most 5 social items, found {len(social_items)}")
    elif len(social_items) < 3:
        notice_text = social_notice.get_text(" ", strip=True) if social_notice else ""
        if notice_text != SOCIAL_SHORTFALL_NOTICE:
            errors.append("fewer than 3 social items requires the reliability notice")
    elif social_notice:
        errors.append("unexpected social reliability notice with 3–5 items")

    news_items = soup.select("[data-news-item]")
    for index, item in enumerate(news_items, start=1):
        if item.get("data-status") not in {"新增", "延续"}:
            errors.append(f"news item {index}: invalid or missing status")
        if not item.select_one(".what-happened"):
            errors.append(f"news item {index}: missing what happened")
        if not item.select_one(".why-important"):
            errors.append(f"news item {index}: missing why important")
        source = item.select_one("a.source-link[href]")
        if not source or not source.get("href", "").startswith(("https://", "http://")):
            errors.append(f"news item {index}: missing clickable source")
        published = item.select_one("time.published-at[datetime]")
        if not published or len(published.get_text(strip=True)) < 10:
            errors.append(f"news item {index}: missing published time")
        if not item.select_one(".confidence"):
            errors.append(f"news item {index}: missing confidence")

    lead_cards = soup.select('[data-section="core"] .lead-card')
    if len(lead_cards) != 3:
        errors.append("expected exactly 3 lead cards")
    elif any(len(card.get_text(" ", strip=True)) < 20 for card in lead_cards):
        errors.append("one or more lead cards are incomplete")
    return errors


def validate(path: Path, site_root: Path, public_path: str | None) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return ["file does not exist"]
    if path.stat().st_size <= 0:
        return ["file is empty"]
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return [f"not valid UTF-8: {exc}"]

    lowered = text.lower()
    for tag in ("<html", "<head", "<body"):
        if tag not in lowered:
            errors.append(f"missing {tag} element")
    if PLACEHOLDER.search(text):
        errors.append("obvious placeholder text detected")

    soup = BeautifulSoup(text, "html.parser")
    if soup.html is None or soup.head is None or soup.body is None:
        errors.append("HTML parser could not find html/head/body")
    errors.extend(check_internal_links(soup, path, site_root, public_path))
    if DAILY_NAME.match(Path(public_path or path.name).name):
        errors.extend(check_daily(soup))
    return errors


def main() -> int:
    args = parse_args()
    site_root = args.site_root.resolve()
    any_errors = False
    for path in args.paths:
        resolved = path.resolve()
        public_path = args.public_path if len(args.paths) == 1 else None
        errors = validate(resolved, site_root, public_path)
        if errors:
            any_errors = True
            print(f"ERROR {path}", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"OK {path} ({resolved.stat().st_size} bytes)")
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
