#!/usr/bin/env python3
"""Generate a web-grounded Philo Daily Brief candidate without touching live pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


BEIJING = ZoneInfo("Asia/Shanghai")
LOG = logging.getLogger("philo-daily")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewsItem(StrictModel):
    status: Literal["新增", "延续"]
    headline: str = Field(min_length=6, max_length=42)
    what_happened: str = Field(min_length=20, max_length=95)
    why_important: str = Field(min_length=20, max_length=95)
    source_name: str = Field(min_length=2, max_length=60)
    source_url: HttpUrl
    published_at: str = Field(min_length=10, max_length=48)
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None


class Signal(StrictModel):
    label: str = Field(min_length=2, max_length=16)
    value: str = Field(min_length=2, max_length=36)
    tone: Literal["积极", "中性", "警惕"]


class WatchItem(StrictModel):
    timeframe: str = Field(min_length=2, max_length=18)
    item: str = Field(min_length=10, max_length=80)


class ImpactItem(StrictModel):
    direction: str = Field(min_length=2, max_length=24)
    impact: str = Field(min_length=12, max_length=80)
    watch: str = Field(min_length=8, max_length=64)


class DailyBrief(StrictModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    title: str = Field(min_length=4, max_length=40)
    summary: str = Field(min_length=30, max_length=150)
    core_conclusions: list[Annotated[str, Field(min_length=20, max_length=100)]] = Field(
        min_length=3, max_length=3
    )
    signal_board: list[Signal] = Field(min_length=4, max_length=6)
    global_finance: list[NewsItem] = Field(min_length=5, max_length=5)
    ai_industry: list[NewsItem] = Field(min_length=5, max_length=5)
    semiconductors: list[NewsItem] = Field(min_length=3, max_length=3)
    social_trends: list[NewsItem] = Field(min_length=2, max_length=2)
    philo_insight: str = Field(min_length=80, max_length=420)
    tomorrow_watch: list[WatchItem] = Field(min_length=4, max_length=5)
    tracking_impacts: list[ImpactItem] = Field(min_length=3, max_length=5)
    data_limitations: list[Annotated[str, Field(min_length=12, max_length=120)]] = Field(
        min_length=1, max_length=4
    )


CATEGORY_FIELDS = (
    ("全球金融", "global_finance"),
    ("AI行业", "ai_industry"),
    ("半导体重点", "semiconductors"),
    ("社媒趋势", "social_trends"),
)


SYSTEM_PROMPT = """你是 Philo Daily Brief 的研究编辑。你必须使用 web_search 搜索当天信息，输出严格符合给定结构的中文日报数据。

编辑规则：
1. 事实与分析分开。不得虚构数字、来源、发布时间、链接或社媒热度。
2. 优先官方公告、公司投资者关系页面、监管机构、央行、交易所、Reuters、AP、FT、Bloomberg 等可靠来源。
3. 每条资讯必须绑定一个实际检索到的来源链接；source_url 必须原样来自搜索结果。
4. 全球金融与 AI 行业各恰好 5 条；半导体恰好 3 条；社媒趋势恰好 2 条。
5. 同一事件不得跨栏目重复。标题、发生了什么、为什么重要都要紧凑，全文适合 5 分钟阅读。
6. 根据给出的过去 7 天事件判断“新增”或“延续”。延续事件填写 continuation_of；新增事件必须为 null。
7. 社媒没有可靠量化数据时，只写可验证的平台政策、公司公告或可靠媒体报道，并在 data_limitations 说明限制。不得编造小红书、抖音或任何平台热度。
8. published_at 使用来源显示的发布日期和时区；找不到明确日期的材料不要采用。
9. confidence 依据来源与交叉验证质量标记为高、中、低。单一二手来源通常不得标高。
10. Philo Insight 是结构化归纳，不得冒充外部事实；重点跟踪方向影响不得虚构用户持仓。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", dest="brief_date")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Local test fixture; bypasses the API and is never used by the workflow.",
    )
    return parser.parse_args()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())


def normalize_url(value: str) -> str:
    parts = urlsplit(value)
    path = re.sub(r"/+$", "", parts.path) or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def source_matches(candidate: str, sources: set[str]) -> bool:
    normalized = normalize_url(candidate)
    if normalized in sources:
        return True
    candidate_parts = urlsplit(normalized)
    return any(
        urlsplit(source).netloc == candidate_parts.netloc
        and urlsplit(source).path == candidate_parts.path
        for source in sources
    )


def collect_response_urls(payload: Any) -> set[str]:
    urls: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "url" and isinstance(child, str) and child.startswith(("http://", "https://")):
                    urls.add(normalize_url(child))
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return urls


def load_history(path: Path, brief_date: date) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at": None, "events": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    cutoff = brief_date - timedelta(days=6)
    kept = []
    for event in data.get("events", []):
        try:
            if date.fromisoformat(event["last_seen"]) >= cutoff:
                kept.append(event)
        except (KeyError, TypeError, ValueError):
            LOG.warning("Ignoring malformed historical event")
    return {"schema_version": 1, "updated_at": data.get("updated_at"), "events": kept}


def iter_news(brief: DailyBrief):
    for category, field_name in CATEGORY_FIELDS:
        for item in getattr(brief, field_name):
            yield category, item


def classify_and_deduplicate(brief: DailyBrief, history: dict[str, Any]) -> None:
    previous = history["events"]
    previous_by_id = {event["id"]: event for event in previous}
    seen_urls: set[str] = set()
    seen_titles: list[str] = []

    for _category, item in iter_news(brief):
        current_url = normalize_url(str(item.source_url))
        current_title = normalize_text(item.headline)
        if current_url in seen_urls:
            raise ValueError(f"Duplicate source URL in current brief: {current_url}")
        if any(SequenceMatcher(None, current_title, title).ratio() >= 0.82 for title in seen_titles):
            raise ValueError(f"Near-duplicate event in current brief: {item.headline}")
        seen_urls.add(current_url)
        seen_titles.append(current_title)

        matched_id: str | None = None
        if item.continuation_of in previous_by_id:
            matched_id = item.continuation_of
        else:
            for old in previous:
                same_url = normalize_url(old.get("source_url", "")) == current_url
                similarity = SequenceMatcher(
                    None, current_title, normalize_text(old.get("title", ""))
                ).ratio()
                if same_url or similarity >= 0.72:
                    matched_id = old["id"]
                    break

        if matched_id:
            item.status = "延续"
            item.continuation_of = matched_id
        else:
            item.status = "新增"
            item.continuation_of = None


def validate_sources(brief: DailyBrief, response_urls: set[str]) -> None:
    if len(response_urls) < 10:
        raise ValueError(
            f"Web search returned only {len(response_urls)} unique URLs; refusing to invent coverage"
        )
    missing = [
        str(item.source_url)
        for _category, item in iter_news(brief)
        if not source_matches(str(item.source_url), response_urls)
    ]
    if missing:
        raise ValueError(
            "Model returned source URLs absent from web_search results: " + ", ".join(missing[:3])
        )


def update_history(brief: DailyBrief, history: dict[str, Any]) -> dict[str, Any]:
    event_date = date.fromisoformat(brief.date)
    events_by_id = {event["id"]: dict(event) for event in history["events"]}

    for category, item in iter_news(brief):
        if item.continuation_of and item.continuation_of in events_by_id:
            event_id = item.continuation_of
            first_seen = events_by_id[event_id]["first_seen"]
        else:
            digest = hashlib.sha256(
                f"{category}|{normalize_text(item.headline)}".encode("utf-8")
            ).hexdigest()[:16]
            event_id = f"{brief.date}-{digest}"
            first_seen = brief.date
        events_by_id[event_id] = {
            "id": event_id,
            "first_seen": first_seen,
            "last_seen": brief.date,
            "category": category,
            "title": item.headline,
            "source_url": str(item.source_url),
        }

    cutoff = event_date - timedelta(days=6)
    events = [
        event
        for event in events_by_id.values()
        if date.fromisoformat(event["last_seen"]) >= cutoff
    ]
    events.sort(key=lambda event: (event["last_seen"], event["category"], event["title"]), reverse=True)
    return {
        "schema_version": 1,
        "updated_at": datetime.now(BEIJING).isoformat(timespec="seconds"),
        "events": events[:200],
    }


def request_brief(
    brief_date: str, history: dict[str, Any], model: str, timeout_seconds: float, max_tokens: int
) -> tuple[DailyBrief, set[str]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
    previous_json = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    user_prompt = f"""生成 {brief_date}（北京时间）的 Philo Daily Brief V3。
检索重点：全球金融、AI、半导体、社媒平台政策与可信行业趋势。
优先采用 {brief_date} 当天或此前 48 小时发布的信息；只有明确的新进展才可延续过去 7 天事件。

过去 7 天事件（仅用于去重与延续判断）：
{previous_json}

输出限制：每条资讯只保留发生了什么、为什么重要、来源与可信度；不要写成长篇背景。所有来源链接必须来自本次 web_search。
"""

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            LOG.info("Calling OpenAI Responses API (attempt %d/2, model=%s)", attempt + 1, model)
            response = client.responses.parse(
                model=model,
                reasoning={"effort": "medium"},
                tools=[{"type": "web_search", "search_context_size": "high"}],
                tool_choice="auto",
                include=["web_search_call.action.sources"],
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=DailyBrief,
                max_output_tokens=max_tokens,
                max_tool_calls=18,
                store=False,
            )
            if response.status != "completed" or response.output_parsed is None:
                raise RuntimeError(f"Incomplete OpenAI response: status={response.status}")
            urls = collect_response_urls(response.model_dump(mode="json"))
            return response.output_parsed, urls
        except Exception as exc:  # One controlled retry for API/network/service failures.
            last_error = exc
            status = getattr(exc, "status_code", None)
            request_id = getattr(exc, "request_id", None)
            LOG.error(
                "OpenAI request failed: type=%s status=%s request_id=%s",
                type(exc).__name__,
                status,
                request_id,
            )
            if attempt == 0:
                time.sleep(3)
    raise RuntimeError("OpenAI Responses API failed after one retry") from last_error


def render_html(root: Path, brief: DailyBrief) -> str:
    environment = Environment(
        loader=FileSystemLoader(root / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("daily_v3.html")
    return template.render(
        brief=brief.model_dump(mode="json"),
        generated_at=datetime.now(BEIJING).isoformat(timespec="minutes"),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    output_dir = (args.output_dir or root / ".build").resolve()
    brief_date = args.brief_date or datetime.now(BEIJING).date().isoformat()
    parsed_date = date.fromisoformat(brief_date)
    history = load_history(root / "data" / "previous_events.json", parsed_date)

    if args.input_json:
        LOG.info("Loading local test fixture; OpenAI API is bypassed")
        brief = DailyBrief.model_validate_json(args.input_json.read_text(encoding="utf-8"))
        response_urls = {normalize_url(str(item.source_url)) for _, item in iter_news(brief)}
    else:
        model = os.environ.get("OPENAI_MODEL", "gpt-5.6")
        timeout_seconds = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "120"))
        max_tokens = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "9000"))
        brief, response_urls = request_brief(
            brief_date, history, model, timeout_seconds, max_tokens
        )

    if brief.date != brief_date:
        raise ValueError(f"Brief date mismatch: requested {brief_date}, received {brief.date}")
    classify_and_deduplicate(brief, history)
    validate_sources(brief, response_urls)
    updated_history = update_history(brief, history)
    html = render_html(root, brief)

    atomic_write(output_dir / f"{brief_date}.html", html)
    atomic_write(
        output_dir / "brief.json",
        json.dumps(brief.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(
        output_dir / "previous_events.json",
        json.dumps(updated_history, ensure_ascii=False, indent=2) + "\n",
    )
    LOG.info("Candidate generated: %s", output_dir / f"{brief_date}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
