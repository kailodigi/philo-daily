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
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import dashscope
from dashscope import Generation
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    source_url: str = Field(min_length=12, max_length=2048)
    published_at: str = Field(min_length=10, max_length=48)
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value


class CandidateItem(StrictModel):
    category: Literal["全球金融", "AI行业", "半导体重点", "社媒趋势"]
    status_hint: Literal["新增", "延续"]
    headline: str = Field(min_length=6, max_length=42)
    what_happened: str = Field(min_length=20, max_length=110)
    why_important: str = Field(min_length=20, max_length=110)
    source_name: str = Field(min_length=2, max_length=60)
    source_index: int = Field(ge=1)
    source_url: str = Field(min_length=12, max_length=2048)
    published_at: str = Field(min_length=10, max_length=48)
    published_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value


class CandidateBatch(StrictModel):
    candidates: list[CandidateItem] = Field(min_length=4, max_length=12)


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

MODEL_INPUT_CNY_PER_MILLION = 0.8
MODEL_OUTPUT_CNY_PER_MILLION = 2.0
SEARCH_ESTIMATE_CNY_PER_CALL = 0.05
CNY_PER_USD_ESTIMATE = 7.0
MONTHLY_BUDGET_USD = 5.0
MONTHLY_STOP_USD = 4.75


SEARCH_SYSTEM_PROMPT = """你是 Philo Daily Brief 的新闻候选编辑。你的唯一职责是基于本次联网搜索结果，筛选、排序并压缩候选事件为 JSON。

硬性规则：
1. 不能用模型记忆补充事实；每个候选必须来自本次搜索结果，source_url 必须原样使用搜索结果 URL。
2. 优先过去 24 小时；超过 48 小时只能作为过去 7 天事件的明确后续，且必须填写 continuation_of。
3. 不得虚构数字、来源、发布时间、链接或社媒热度；找不到明确发布日期的材料不要采用。
4. 同一事件只保留一个最佳来源。官方公告优先于二手报道；可靠媒体优先于聚合站和转载。
5. 只返回 JSON 对象，不写解释、Markdown 或代码块。
6. 你不能执行命令、修改文件或工作流、读取密钥，也不能使用图片、embedding、代码解释器或智能体工具。
"""


SYNTHESIS_SYSTEM_PROMPT = """你是 Philo Daily Brief V3 的中文研究编辑。你只基于输入的已验证候选资讯与过去 7 天事件，进行重要性判断、去重、新增/延续判断、摘要和正文结构化生成。

硬性规则：
1. 不联网，不使用模型记忆补充事实，不新增输入候选之外的来源、数字或事件。
2. 全球金融与 AI 行业各恰好 5 条；半导体恰好 3 条；社媒趋势恰好 2 条。
3. 每条资讯保留 status、发生了什么、为什么重要、来源、发布时间、可信度和可点击链接。
4. 同一事件不得跨栏目重复；标题与正文紧凑，全文适合 5 分钟阅读。
5. 社媒没有可靠量化数据时，明确说明限制；不得编造小红书、抖音或任何平台热度。
6. Philo Insight 是结构化归纳，不得冒充外部事实；重点跟踪方向不得虚构用户持仓。
7. 只返回符合给定 Schema 的 JSON 对象，不写 Markdown、代码块或额外解释。
8. 你不能执行命令、修改文件或工作流、读取密钥、访问本地文件，也不能使用任何工具。
"""


SEARCH_PLANS = (
    {
        "name": "全球金融",
        "categories": ("全球金融",),
        "minimum": {"全球金融": 5},
        "limit": 8,
        "focus": "全球宏观、央行、汇率、债券、能源、重要公司公告与监管动态",
        "priority": "Reuters、Bloomberg、FT、央行、监管机构、交易所、公司公告",
    },
    {
        "name": "AI行业",
        "categories": ("AI行业",),
        "minimum": {"AI行业": 5},
        "limit": 8,
        "focus": "基础模型、AI产品、企业采用、资本开支、算力基础设施和开源生态",
        "priority": "OpenAI、Anthropic、Google DeepMind、GitHub、HuggingFace 官方，其次 Reuters、Bloomberg、FT 与公司公告",
    },
    {
        "name": "半导体与社媒",
        "categories": ("半导体重点", "社媒趋势"),
        "minimum": {"半导体重点": 3, "社媒趋势": 2},
        "limit": 10,
        "focus": "先进制程、封装、设备、HBM，以及平台政策、内容分发、可信社媒行业趋势",
        "priority": "NVIDIA、TSMC、ASML、SEMI 与公司公告；社媒采用平台官方公告或 Reuters 等可靠媒体",
    },
)


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


class UsageLedger:
    def __init__(self) -> None:
        self.api_attempts = 0
        self.successful_calls = 0
        self.search_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, response: Any, *, search_enabled: bool) -> None:
        usage = getattr(response, "usage", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        self.successful_calls += 1
        self.search_calls += int(search_enabled)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        LOG.info(
            "DashScope call usage: search=%s input_tokens=%d output_tokens=%d",
            search_enabled,
            input_tokens,
            output_tokens,
        )

    def as_record(self, brief_date: str, model: str) -> dict[str, Any]:
        model_cost_cny = (
            self.input_tokens * MODEL_INPUT_CNY_PER_MILLION
            + self.output_tokens * MODEL_OUTPUT_CNY_PER_MILLION
        ) / 1_000_000
        search_cost_cny = self.search_calls * SEARCH_ESTIMATE_CNY_PER_CALL
        total_cny = model_cost_cny + search_cost_cny
        return {
            "date": brief_date,
            "model": model,
            "generation_count": 1,
            "api_attempts": self.api_attempts,
            "successful_calls": self.successful_calls,
            "search_calls": self.search_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_model_cost_cny": round(model_cost_cny, 6),
            "estimated_search_cost_cny": round(search_cost_cny, 6),
            "estimated_total_cost_cny": round(total_cny, 6),
            "estimated_total_cost_usd": round(total_cny / CNY_PER_USD_ESTIMATE, 6),
        }


def empty_usage_history() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pricing_note": "Conservative estimate; actual Alibaba Cloud billing and FX may differ.",
        "pricing_assumptions": {
            "qwen_plus_input_cny_per_million_tokens": MODEL_INPUT_CNY_PER_MILLION,
            "qwen_plus_output_cny_per_million_tokens": MODEL_OUTPUT_CNY_PER_MILLION,
            "web_search_estimate_cny_per_call": SEARCH_ESTIMATE_CNY_PER_CALL,
            "cny_per_usd_estimate": CNY_PER_USD_ESTIMATE,
            "monthly_budget_usd": MONTHLY_BUDGET_USD,
        },
        "days": [],
    }


def load_usage_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_usage_history()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("days"), list):
        raise ValueError("data/usage.json has an invalid days field")
    return data


def enforce_monthly_budget(history: dict[str, Any], brief_date: str) -> None:
    month = brief_date[:7]
    spent = sum(
        float(item.get("estimated_total_cost_usd", 0))
        for item in history.get("days", [])
        if item.get("date", "")[:7] == month
    )
    if spent >= MONTHLY_STOP_USD:
        raise RuntimeError(
            f"Estimated monthly API cost is already ${spent:.2f}; stopping before the ${MONTHLY_BUDGET_USD:.2f} budget"
        )


def update_usage_history(
    history: dict[str, Any], brief_date: str, model: str, ledger: UsageLedger
) -> dict[str, Any]:
    record = ledger.as_record(brief_date, model)
    days = []
    merged = False
    additive_fields = (
        "generation_count",
        "api_attempts",
        "successful_calls",
        "search_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_model_cost_cny",
        "estimated_search_cost_cny",
        "estimated_total_cost_cny",
        "estimated_total_cost_usd",
    )
    for item in history.get("days", []):
        if item.get("date") != brief_date:
            days.append(item)
            continue
        combined = dict(item)
        for field in additive_fields:
            combined[field] = round(float(item.get(field, 0)) + float(record.get(field, 0)), 6)
        for field in (
            "generation_count",
            "api_attempts",
            "successful_calls",
            "search_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            combined[field] = int(combined[field])
        combined["model"] = model
        days.append(combined)
        merged = True
    if not merged:
        days.append(record)
    days.sort(key=lambda item: item["date"], reverse=True)
    updated = empty_usage_history()
    updated["updated_at"] = datetime.now(BEIJING).isoformat(timespec="seconds")
    updated["days"] = days[:400]
    return updated


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
            f"DashScope search returned only {len(response_urls)} unique URLs; refusing to invent coverage"
        )
    missing = [
        str(item.source_url)
        for _category, item in iter_news(brief)
        if not source_matches(str(item.source_url), response_urls)
    ]
    if missing:
        raise ValueError(
            "Qwen returned source URLs absent from verified candidates: " + ", ".join(missing[:3])
        )


def match_history(
    headline: str, source_url: str, continuation_of: str | None, history: dict[str, Any]
) -> str | None:
    previous_by_id = {event["id"]: event for event in history["events"]}
    if continuation_of in previous_by_id:
        return continuation_of
    normalized_url = normalize_url(source_url)
    normalized_title = normalize_text(headline)
    for old in history["events"]:
        same_url = normalize_url(old.get("source_url", "")) == normalized_url
        similarity = SequenceMatcher(
            None, normalized_title, normalize_text(old.get("title", ""))
        ).ratio()
        if same_url or similarity >= 0.72:
            return old["id"]
    return None


def filter_candidate_recency(
    candidates: list[CandidateItem], brief_date: date, history: dict[str, Any]
) -> list[CandidateItem]:
    kept: list[CandidateItem] = []
    for item in candidates:
        published = date.fromisoformat(item.published_date)
        age_days = (brief_date - published).days
        matched_id = match_history(
            item.headline, item.source_url, item.continuation_of, history
        )
        if age_days < 0 or age_days > 6:
            LOG.warning("Dropping candidate outside the 7-day window: %s", item.headline)
            continue
        if age_days >= 2 and not matched_id:
            LOG.warning("Dropping news older than 48h without continuation: %s", item.headline)
            continue
        if matched_id:
            item.status_hint = "延续"
            item.continuation_of = matched_id
        else:
            item.status_hint = "新增"
            item.continuation_of = None
        kept.append(item)
    return kept


def validate_final_recency(brief: DailyBrief, brief_date: date) -> None:
    for _category, item in iter_news(brief):
        match = re.search(r"\d{4}-\d{2}-\d{2}", item.published_at)
        if not match:
            raise ValueError(f"Published time lacks an ISO date: {item.headline}")
        age_days = (brief_date - date.fromisoformat(match.group(0))).days
        if age_days < 0 or age_days > 6:
            raise ValueError(f"Source date outside the 7-day window: {item.headline}")
        if age_days >= 2 and item.status != "延续":
            raise ValueError(f"News older than 48h is not a continuation: {item.headline}")


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


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Qwen output is not a JSON object")
    return payload


def call_dashscope_json(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: int,
    ledger: UsageLedger,
    search_enabled: bool,
) -> tuple[dict[str, Any], Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        ledger.api_attempts += 1
        try:
            LOG.info(
                "Calling DashScope Generation API (attempt %d/2, model=%s, search=%s)",
                attempt + 1,
                model,
                search_enabled,
            )
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "model": model,
                "messages": messages,
                "result_format": "message",
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "request_timeout": timeout_seconds,
            }
            if search_enabled:
                kwargs["enable_search"] = True
                kwargs["search_options"] = {
                    "forced_search": True,
                    "search_strategy": "max",
                    "enable_source": True,
                    "enable_citation": True,
                    "citation_format": "[<number>]",
                    "freshness": 7,
                }
            response = Generation.call(**kwargs)
            if response.status_code != HTTPStatus.OK:
                error = RuntimeError(
                    f"DashScope response status={response.status_code} code={response.code}"
                )
                setattr(error, "status_code", response.status_code)
                setattr(error, "request_id", response.request_id)
                setattr(error, "error_code", response.code)
                raise error
            content = response.output.choices[0].message.content
            payload = parse_json_object(content)
            ledger.record(response, search_enabled=search_enabled)
            return payload, response
        except Exception as exc:
            last_error = exc
            LOG.error(
                "DashScope request failed: type=%s status=%s code=%s request_id=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
                getattr(exc, "error_code", None),
                getattr(exc, "request_id", None),
            )
            if attempt == 0:
                time.sleep(3)
    raise RuntimeError("DashScope API failed after one retry") from last_error


def search_result_records(response: Any) -> list[dict[str, Any]]:
    search_info = getattr(response.output, "search_info", None) or {}
    results = search_info.get("search_results", []) or []
    return [
        item
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("url"), str)
        and item["url"].startswith(("https://", "http://"))
    ]


def resolve_candidate_source(
    item: CandidateItem, search_results: list[dict[str, Any]]
) -> bool:
    indexed_matches = [
        result
        for result in search_results
        if str(result.get("index", "")).strip() == str(item.source_index)
    ]
    if len(indexed_matches) == 1:
        matched = indexed_matches[0]
        item.source_url = matched["url"]
        site_name = str(matched.get("site_name", "")).strip()
        if site_name:
            item.source_name = site_name[:60]
        return True

    source_urls = {normalize_url(result["url"]) for result in search_results}
    if source_matches(item.source_url, source_urls):
        return True

    candidate_host = urlsplit(item.source_url).netloc.lower()
    same_host = [
        result
        for result in search_results
        if urlsplit(result["url"]).netloc.lower() == candidate_host
    ]
    matches = same_host
    if not matches:
        source_name = normalize_text(item.source_name)
        if len(source_name) >= 2:
            matches = []
            for result in search_results:
                site_name = normalize_text(str(result.get("site_name", "")))
                host = normalize_text(urlsplit(result["url"]).netloc)
                if source_name in f"{site_name} {host}" or (
                    site_name and site_name in source_name
                ):
                    matches.append(result)
    if not matches:
        return False

    candidate_text = normalize_text(f"{item.headline} {item.what_happened}")
    best = max(
        matches,
        key=lambda result: SequenceMatcher(
            None,
            candidate_text,
            normalize_text(str(result.get("title", ""))),
        ).ratio(),
    )
    item.source_url = best["url"]
    site_name = str(best.get("site_name", "")).strip()
    if site_name:
        item.source_name = site_name[:60]
    return True


def collect_candidates(
    *,
    brief_date: str,
    history: dict[str, Any],
    api_key: str,
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[list[CandidateItem], set[str]]:
    compact_history = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    all_candidates: list[CandidateItem] = []
    all_source_urls: set[str] = set()

    for plan in SEARCH_PLANS:
        categories = "、".join(plan["categories"])
        prompt = f"""今天是 {brief_date}（北京时间）。联网搜索并为“{plan['name']}”建立新闻候选池。

检索主题：{plan['focus']}
来源优先级：{plan['priority']}
时效要求：优先过去 24 小时；过去 24–48 小时只在确有重要性时采用；更早内容只能是下列过去 7 天事件的明确后续。
过去 7 天事件：{compact_history}

输出 JSON Schema：
{json.dumps(CandidateBatch.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

要求：
- 输出 6 到 {plan['limit']} 个按重要性排序的候选，category 只能是：{categories}。
- source_index 必须是本次搜索角标 [n] 中的整数 n；一个候选只能引用一个最直接的来源。
- source_url 必须逐字来自同一角标对应的 DashScope 搜索来源；published_at 必须以 YYYY-MM-DD 开头并保留来源显示的时间/时区，published_date 为同一日期。
- 找不到来源发布日期、无法验证链接或只是旧闻重复的内容不要输出。
- 社媒只采用平台公告或可靠媒体；没有量化证据时不得写热度数字。
- 只返回 JSON 对象。
"""
        payload, response = call_dashscope_json(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=search_max_tokens,
            timeout_seconds=timeout_seconds,
            ledger=ledger,
            search_enabled=True,
        )
        batch = CandidateBatch.model_validate(payload)
        search_results = search_result_records(response)
        source_urls = {normalize_url(result["url"]) for result in search_results}
        if len(source_urls) < sum(plan["minimum"].values()):
            raise ValueError(
                f"{plan['name']} search returned only {len(source_urls)} source URLs"
            )
        invalid_categories = [
            item.category for item in batch.candidates if item.category not in plan["categories"]
        ]
        if invalid_categories:
            raise ValueError(f"Unexpected candidate categories in {plan['name']}")
        verified = [
            item
            for item in batch.candidates
            if resolve_candidate_source(item, search_results)
        ]
        discarded = len(batch.candidates) - len(verified)
        if discarded:
            LOG.warning(
                "Discarded %d %s candidate(s) whose URLs were absent from DashScope search",
                discarded,
                plan["name"],
            )
        filtered = filter_candidate_recency(
            verified, date.fromisoformat(brief_date), history
        )
        for category, minimum in plan["minimum"].items():
            count = sum(item.category == category for item in filtered)
            if count < minimum:
                raise ValueError(
                    f"{plan['name']} has only {count} usable {category} candidates; need {minimum}"
                )
        all_candidates.extend(filtered)
        all_source_urls.update(source_urls)

    unique: list[CandidateItem] = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    for item in all_candidates:
        normalized_url = normalize_url(item.source_url)
        normalized_title = normalize_text(item.headline)
        if normalized_url in seen_urls:
            continue
        if any(SequenceMatcher(None, normalized_title, old).ratio() >= 0.82 for old in seen_titles):
            continue
        unique.append(item)
        seen_urls.add(normalized_url)
        seen_titles.append(normalized_title)

    required = {"全球金融": 5, "AI行业": 5, "半导体重点": 3, "社媒趋势": 2}
    for category, minimum in required.items():
        count = sum(item.category == category for item in unique)
        if count < minimum:
            raise ValueError(f"Only {count} unique {category} candidates remain; need {minimum}")
    return unique, all_source_urls


def request_brief(
    brief_date: str,
    history: dict[str, Any],
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    final_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[DailyBrief, set[str], list[CandidateItem]]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")

    candidates, search_urls = collect_candidates(
        brief_date=brief_date,
        history=history,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        search_max_tokens=search_max_tokens,
        ledger=ledger,
    )
    candidate_json = json.dumps(
        [item.model_dump(mode="json") for item in candidates],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    history_json = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    prompt = f"""为 {brief_date}（北京时间）生成 Philo Daily Brief V3 的 JSON 数据。

已验证候选（只能从中选择，禁止添加候选之外的事实和链接）：
{candidate_json}

过去 7 天事件（用于新增/延续判断）：
{history_json}

输出 JSON Schema：
{json.dumps(DailyBrief.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

再次强调：全球金融 5 条、AI 行业 5 条、半导体 3 条、社媒 2 条；source_url、source_name、published_at 必须原样继承候选。只返回 JSON 对象。
"""
    payload, _response = call_dashscope_json(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=final_max_tokens,
        timeout_seconds=timeout_seconds,
        ledger=ledger,
        search_enabled=False,
    )
    brief = DailyBrief.model_validate(payload)
    candidate_urls = {normalize_url(item.source_url) for item in candidates}
    if not candidate_urls.issubset(search_urls):
        raise ValueError("Candidate source set is not a subset of DashScope search results")
    return brief, candidate_urls, candidates


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
    usage_history = load_usage_history(root / "data" / "usage.json")
    enforce_monthly_budget(usage_history, brief_date)
    ledger = UsageLedger()
    model = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
    candidates: list[CandidateItem] = []

    if args.input_json:
        LOG.info("Loading local test fixture; DashScope API is bypassed")
        brief = DailyBrief.model_validate_json(args.input_json.read_text(encoding="utf-8"))
        response_urls = {normalize_url(str(item.source_url)) for _, item in iter_news(brief)}
    else:
        base_http_api_url = os.environ.get("DASHSCOPE_BASE_HTTP_API_URL", "").strip()
        if base_http_api_url:
            parsed_base_url = urlsplit(base_http_api_url)
            if (
                parsed_base_url.scheme != "https"
                or not parsed_base_url.hostname
                or not parsed_base_url.hostname.endswith(".aliyuncs.com")
                or parsed_base_url.path.rstrip("/") != "/api/v1"
            ):
                raise ValueError(
                    "DASHSCOPE_BASE_HTTP_API_URL must be an HTTPS Alibaba Cloud "
                    "Model Studio /api/v1 endpoint"
                )
            dashscope.base_http_api_url = base_http_api_url.rstrip("/")
        timeout_seconds = int(os.environ.get("DASHSCOPE_TIMEOUT_SECONDS", "120"))
        search_max_tokens = int(os.environ.get("DASHSCOPE_SEARCH_MAX_TOKENS", "2400"))
        final_max_tokens = int(os.environ.get("DASHSCOPE_FINAL_MAX_TOKENS", "6000"))
        brief, response_urls, candidates = request_brief(
            brief_date,
            history,
            model,
            timeout_seconds,
            search_max_tokens,
            final_max_tokens,
            ledger,
        )

    if brief.date != brief_date:
        raise ValueError(f"Brief date mismatch: requested {brief_date}, received {brief.date}")
    classify_and_deduplicate(brief, history)
    validate_sources(brief, response_urls)
    validate_final_recency(brief, parsed_date)
    updated_history = update_history(brief, history)
    updated_usage = update_usage_history(usage_history, brief_date, model, ledger)
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
    atomic_write(
        output_dir / "usage.json",
        json.dumps(updated_usage, ensure_ascii=False, indent=2) + "\n",
    )
    if candidates:
        atomic_write(
            output_dir / "candidates.json",
            json.dumps(
                [item.model_dump(mode="json") for item in candidates],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    usage = ledger.as_record(brief_date, model)
    LOG.info(
        "DashScope usage: successful_calls=%d search_calls=%d input_tokens=%d output_tokens=%d estimated_usd=%.4f",
        usage["successful_calls"],
        usage["search_calls"],
        usage["input_tokens"],
        usage["output_tokens"],
        usage["estimated_total_cost_usd"],
    )
    LOG.info("Candidate generated: %s", output_dir / f"{brief_date}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import dashscope
from dashscope import Generation
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    source_url: str = Field(min_length=12, max_length=2048)
    published_at: str = Field(min_length=10, max_length=48)
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value


class CandidateItem(StrictModel):
    category: Literal["全球金融", "AI行业", "半导体重点", "社媒趋势"]
    status_hint: Literal["新增", "延续"]
    headline: str = Field(min_length=6, max_length=42)
    what_happened: str = Field(min_length=20, max_length=110)
    why_important: str = Field(min_length=20, max_length=110)
    source_name: str = Field(min_length=2, max_length=60)
    source_index: int = Field(ge=1)
    source_url: str = Field(min_length=12, max_length=2048)
    published_at: str = Field(min_length=10, max_length=48)
    published_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value


class CandidateBatch(StrictModel):
    candidates: list[CandidateItem] = Field(min_length=4, max_length=12)


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

MODEL_INPUT_CNY_PER_MILLION = 0.8
MODEL_OUTPUT_CNY_PER_MILLION = 2.0
SEARCH_ESTIMATE_CNY_PER_CALL = 0.05
CNY_PER_USD_ESTIMATE = 7.0
MONTHLY_BUDGET_USD = 5.0
MONTHLY_STOP_USD = 4.75


SEARCH_SYSTEM_PROMPT = """你是 Philo Daily Brief 的新闻候选编辑。你的唯一职责是基于本次联网搜索结果，筛选、排序并压缩候选事件为 JSON。

硬性规则：
1. 不能用模型记忆补充事实；每个候选必须来自本次搜索结果，source_url 必须原样使用搜索结果 URL。
2. 优先过去 24 小时；超过 48 小时只能作为过去 7 天事件的明确后续，且必须填写 continuation_of。
3. 不得虚构数字、来源、发布时间、链接或社媒热度；找不到明确发布日期的材料不要采用。
4. 同一事件只保留一个最佳来源。官方公告优先于二手报道；可靠媒体优先于聚合站和转载。
5. 只返回 JSON 对象，不写解释、Markdown 或代码块。
6. 你不能执行命令、修改文件或工作流、读取密钥，也不能使用图片、embedding、代码解释器或智能体工具。
"""


SYNTHESIS_SYSTEM_PROMPT = """你是 Philo Daily Brief V3 的中文研究编辑。你只基于输入的已验证候选资讯与过去 7 天事件，进行重要性判断、去重、新增/延续判断、摘要和正文结构化生成。

硬性规则：
1. 不联网，不使用模型记忆补充事实，不新增输入候选之外的来源、数字或事件。
2. 全球金融与 AI 行业各恰好 5 条；半导体恰好 3 条；社媒趋势恰好 2 条。
3. 每条资讯保留 status、发生了什么、为什么重要、来源、发布时间、可信度和可点击链接。
4. 同一事件不得跨栏目重复；标题与正文紧凑，全文适合 5 分钟阅读。
5. 社媒没有可靠量化数据时，明确说明限制；不得编造小红书、抖音或任何平台热度。
6. Philo Insight 是结构化归纳，不得冒充外部事实；重点跟踪方向不得虚构用户持仓。
7. 只返回符合给定 Schema 的 JSON 对象，不写 Markdown、代码块或额外解释。
8. 你不能执行命令、修改文件或工作流、读取密钥、访问本地文件，也不能使用任何工具。
"""


SEARCH_PLANS = (
    {
        "name": "全球金融",
        "categories": ("全球金融",),
        "minimum": {"全球金融": 5},
        "limit": 8,
        "focus": "全球宏观、央行、汇率、债券、能源、重要公司公告与监管动态",
        "priority": "Reuters、Bloomberg、FT、央行、监管机构、交易所、公司公告",
    },
    {
        "name": "AI行业",
        "categories": ("AI行业",),
        "minimum": {"AI行业": 5},
        "limit": 8,
        "focus": "基础模型、AI产品、企业采用、资本开支、算力基础设施和开源生态",
        "priority": "OpenAI、Anthropic、Google DeepMind、GitHub、HuggingFace 官方，其次 Reuters、Bloomberg、FT 与公司公告",
    },
    {
        "name": "半导体与社媒",
        "categories": ("半导体重点", "社媒趋势"),
        "minimum": {"半导体重点": 3, "社媒趋势": 2},
        "limit": 10,
        "focus": "先进制程、封装、设备、HBM，以及平台政策、内容分发、可信社媒行业趋势",
        "priority": "NVIDIA、TSMC、ASML、SEMI 与公司公告；社媒采用平台官方公告或 Reuters 等可靠媒体",
    },
)


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


class UsageLedger:
    def __init__(self) -> None:
        self.api_attempts = 0
        self.successful_calls = 0
        self.search_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, response: Any, *, search_enabled: bool) -> None:
        usage = getattr(response, "usage", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        self.successful_calls += 1
        self.search_calls += int(search_enabled)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        LOG.info(
            "DashScope call usage: search=%s input_tokens=%d output_tokens=%d",
            search_enabled,
            input_tokens,
            output_tokens,
        )

    def as_record(self, brief_date: str, model: str) -> dict[str, Any]:
        model_cost_cny = (
            self.input_tokens * MODEL_INPUT_CNY_PER_MILLION
            + self.output_tokens * MODEL_OUTPUT_CNY_PER_MILLION
        ) / 1_000_000
        search_cost_cny = self.search_calls * SEARCH_ESTIMATE_CNY_PER_CALL
        total_cny = model_cost_cny + search_cost_cny
        return {
            "date": brief_date,
            "model": model,
            "generation_count": 1,
            "api_attempts": self.api_attempts,
            "successful_calls": self.successful_calls,
            "search_calls": self.search_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_model_cost_cny": round(model_cost_cny, 6),
            "estimated_search_cost_cny": round(search_cost_cny, 6),
            "estimated_total_cost_cny": round(total_cny, 6),
            "estimated_total_cost_usd": round(total_cny / CNY_PER_USD_ESTIMATE, 6),
        }


def empty_usage_history() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pricing_note": "Conservative estimate; actual Alibaba Cloud billing and FX may differ.",
        "pricing_assumptions": {
            "qwen_plus_input_cny_per_million_tokens": MODEL_INPUT_CNY_PER_MILLION,
            "qwen_plus_output_cny_per_million_tokens": MODEL_OUTPUT_CNY_PER_MILLION,
            "web_search_estimate_cny_per_call": SEARCH_ESTIMATE_CNY_PER_CALL,
            "cny_per_usd_estimate": CNY_PER_USD_ESTIMATE,
            "monthly_budget_usd": MONTHLY_BUDGET_USD,
        },
        "days": [],
    }


def load_usage_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_usage_history()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("days"), list):
        raise ValueError("data/usage.json has an invalid days field")
    return data


def enforce_monthly_budget(history: dict[str, Any], brief_date: str) -> None:
    month = brief_date[:7]
    spent = sum(
        float(item.get("estimated_total_cost_usd", 0))
        for item in history.get("days", [])
        if item.get("date", "")[:7] == month
    )
    if spent >= MONTHLY_STOP_USD:
        raise RuntimeError(
            f"Estimated monthly API cost is already ${spent:.2f}; stopping before the ${MONTHLY_BUDGET_USD:.2f} budget"
        )


def update_usage_history(
    history: dict[str, Any], brief_date: str, model: str, ledger: UsageLedger
) -> dict[str, Any]:
    record = ledger.as_record(brief_date, model)
    days = []
    merged = False
    additive_fields = (
        "generation_count",
        "api_attempts",
        "successful_calls",
        "search_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_model_cost_cny",
        "estimated_search_cost_cny",
        "estimated_total_cost_cny",
        "estimated_total_cost_usd",
    )
    for item in history.get("days", []):
        if item.get("date") != brief_date:
            days.append(item)
            continue
        combined = dict(item)
        for field in additive_fields:
            combined[field] = round(float(item.get(field, 0)) + float(record.get(field, 0)), 6)
        for field in (
            "generation_count",
            "api_attempts",
            "successful_calls",
            "search_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            combined[field] = int(combined[field])
        combined["model"] = model
        days.append(combined)
        merged = True
    if not merged:
        days.append(record)
    days.sort(key=lambda item: item["date"], reverse=True)
    updated = empty_usage_history()
    updated["updated_at"] = datetime.now(BEIJING).isoformat(timespec="seconds")
    updated["days"] = days[:400]
    return updated


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
            f"DashScope search returned only {len(response_urls)} unique URLs; refusing to invent coverage"
        )
    missing = [
        str(item.source_url)
        for _category, item in iter_news(brief)
        if not source_matches(str(item.source_url), response_urls)
    ]
    if missing:
        raise ValueError(
            "Qwen returned source URLs absent from verified candidates: " + ", ".join(missing[:3])
        )


def match_history(
    headline: str, source_url: str, continuation_of: str | None, history: dict[str, Any]
) -> str | None:
    previous_by_id = {event["id"]: event for event in history["events"]}
    if continuation_of in previous_by_id:
        return continuation_of
    normalized_url = normalize_url(source_url)
    normalized_title = normalize_text(headline)
    for old in history["events"]:
        same_url = normalize_url(old.get("source_url", "")) == normalized_url
        similarity = SequenceMatcher(
            None, normalized_title, normalize_text(old.get("title", ""))
        ).ratio()
        if same_url or similarity >= 0.72:
            return old["id"]
    return None


def filter_candidate_recency(
    candidates: list[CandidateItem], brief_date: date, history: dict[str, Any]
) -> list[CandidateItem]:
    kept: list[CandidateItem] = []
    for item in candidates:
        published = date.fromisoformat(item.published_date)
        age_days = (brief_date - published).days
        matched_id = match_history(
            item.headline, item.source_url, item.continuation_of, history
        )
        if age_days < 0 or age_days > 6:
            LOG.warning("Dropping candidate outside the 7-day window: %s", item.headline)
            continue
        if age_days >= 2 and not matched_id:
            LOG.warning("Dropping news older than 48h without continuation: %s", item.headline)
            continue
        if matched_id:
            item.status_hint = "延续"
            item.continuation_of = matched_id
        else:
            item.status_hint = "新增"
            item.continuation_of = None
        kept.append(item)
    return kept


def validate_final_recency(brief: DailyBrief, brief_date: date) -> None:
    for _category, item in iter_news(brief):
        match = re.search(r"\d{4}-\d{2}-\d{2}", item.published_at)
        if not match:
            raise ValueError(f"Published time lacks an ISO date: {item.headline}")
        age_days = (brief_date - date.fromisoformat(match.group(0))).days
        if age_days < 0 or age_days > 6:
            raise ValueError(f"Source date outside the 7-day window: {item.headline}")
        if age_days >= 2 and item.status != "延续":
            raise ValueError(f"News older than 48h is not a continuation: {item.headline}")


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


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Qwen output is not a JSON object")
    return payload


def call_dashscope_json(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: int,
    ledger: UsageLedger,
    search_enabled: bool,
) -> tuple[dict[str, Any], Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        ledger.api_attempts += 1
        try:
            LOG.info(
                "Calling DashScope Generation API (attempt %d/2, model=%s, search=%s)",
                attempt + 1,
                model,
                search_enabled,
            )
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "model": model,
                "messages": messages,
                "result_format": "message",
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "request_timeout": timeout_seconds,
            }
            if search_enabled:
                kwargs["enable_search"] = True
                kwargs["search_options"] = {
                    "forced_search": True,
                    "search_strategy": "max",
                    "enable_source": True,
                    "enable_citation": True,
                    "citation_format": "[<number>]",
                    "freshness": 7,
                }
            response = Generation.call(**kwargs)
            if response.status_code != HTTPStatus.OK:
                error = RuntimeError(
                    f"DashScope response status={response.status_code} code={response.code}"
                )
                setattr(error, "status_code", response.status_code)
                setattr(error, "request_id", response.request_id)
                setattr(error, "error_code", response.code)
                raise error
            content = response.output.choices[0].message.content
            payload = parse_json_object(content)
            ledger.record(response, search_enabled=search_enabled)
            return payload, response
        except Exception as exc:
            last_error = exc
            LOG.error(
                "DashScope request failed: type=%s status=%s code=%s request_id=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
                getattr(exc, "error_code", None),
                getattr(exc, "request_id", None),
            )
            if attempt == 0:
                time.sleep(3)
    raise RuntimeError("DashScope API failed after one retry") from last_error


def search_result_records(response: Any) -> list[dict[str, Any]]:
    search_info = getattr(response.output, "search_info", None) or {}
    results = search_info.get("search_results", []) or []
    return [
        item
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("url"), str)
        and item["url"].startswith(("https://", "http://"))
    ]


def resolve_candidate_source(
    item: CandidateItem, search_results: list[dict[str, Any]]
) -> bool:
    indexed_matches = [
        result
        for result in search_results
        if str(result.get("index", "")).strip() == str(item.source_index)
    ]
    if len(indexed_matches) == 1:
        matched = indexed_matches[0]
        item.source_url = matched["url"]
        site_name = str(matched.get("site_name", "")).strip()
        if site_name:
            item.source_name = site_name[:60]
        return True

    source_urls = {normalize_url(result["url"]) for result in search_results}
    if source_matches(item.source_url, source_urls):
        return True

    candidate_host = urlsplit(item.source_url).netloc.lower()
    same_host = [
        result
        for result in search_results
        if urlsplit(result["url"]).netloc.lower() == candidate_host
    ]
    matches = same_host
    if not matches:
        source_name = normalize_text(item.source_name)
        if len(source_name) >= 2:
            matches = []
            for result in search_results:
                site_name = normalize_text(str(result.get("site_name", "")))
                host = normalize_text(urlsplit(result["url"]).netloc)
                if source_name in f"{site_name} {host}" or (
                    site_name and site_name in source_name
                ):
                    matches.append(result)
    if not matches:
        return False

    candidate_text = normalize_text(f"{item.headline} {item.what_happened}")
    best = max(
        matches,
        key=lambda result: SequenceMatcher(
            None,
            candidate_text,
            normalize_text(str(result.get("title", ""))),
        ).ratio(),
    )
    item.source_url = best["url"]
    site_name = str(best.get("site_name", "")).strip()
    if site_name:
        item.source_name = site_name[:60]
    return True


def collect_candidates(
    *,
    brief_date: str,
    history: dict[str, Any],
    api_key: str,
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[list[CandidateItem], set[str]]:
    compact_history = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    all_candidates: list[CandidateItem] = []
    all_source_urls: set[str] = set()

    for plan in SEARCH_PLANS:
        categories = "、".join(plan["categories"])
        prompt = f"""今天是 {brief_date}（北京时间）。联网搜索并为“{plan['name']}”建立新闻候选池。

检索主题：{plan['focus']}
来源优先级：{plan['priority']}
时效要求：优先过去 24 小时；过去 24–48 小时只在确有重要性时采用；更早内容只能是下列过去 7 天事件的明确后续。
过去 7 天事件：{compact_history}

输出 JSON Schema：
{json.dumps(CandidateBatch.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

要求：
- 输出 6 到 {plan['limit']} 个按重要性排序的候选，category 只能是：{categories}。
- source_index 必须是本次搜索角标 [n] 中的整数 n；一个候选只能引用一个最直接的来源。
- source_url 必须逐字来自同一角标对应的 DashScope 搜索来源；published_at 必须以 YYYY-MM-DD 开头并保留来源显示的时间/时区，published_date 为同一日期。
- 找不到来源发布日期、无法验证链接或只是旧闻重复的内容不要输出。
- 社媒只采用平台公告或可靠媒体；没有量化证据时不得写热度数字。
- 只返回 JSON 对象。
"""
        payload, response = call_dashscope_json(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=search_max_tokens,
            timeout_seconds=timeout_seconds,
            ledger=ledger,
            search_enabled=True,
        )
        batch = CandidateBatch.model_validate(payload)
        search_results = search_result_records(response)
        source_urls = {normalize_url(result["url"]) for result in search_results}
        if len(source_urls) < sum(plan["minimum"].values()):
            raise ValueError(
                f"{plan['name']} search returned only {len(source_urls)} source URLs"
            )
        invalid_categories = [
            item.category for item in batch.candidates if item.category not in plan["categories"]
        ]
        if invalid_categories:
            raise ValueError(f"Unexpected candidate categories in {plan['name']}")
        verified = [
            item
            for item in batch.candidates
            if resolve_candidate_source(item, search_results)
        ]
        discarded = len(batch.candidates) - len(verified)
        if discarded:
            LOG.warning(
                "Discarded %d %s candidate(s) whose URLs were absent from DashScope search",
                discarded,
                plan["name"],
            )
        filtered = filter_candidate_recency(
            verified, date.fromisoformat(brief_date), history
        )
        for category, minimum in plan["minimum"].items():
            count = sum(item.category == category for item in filtered)
            if count < minimum:
                raise ValueError(
                    f"{plan['name']} has only {count} usable {category} candidates; need {minimum}"
                )
        all_candidates.extend(filtered)
        all_source_urls.update(source_urls)

    unique: list[CandidateItem] = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    for item in all_candidates:
        normalized_url = normalize_url(item.source_url)
        normalized_title = normalize_text(item.headline)
        if normalized_url in seen_urls:
            continue
        if any(SequenceMatcher(None, normalized_title, old).ratio() >= 0.82 for old in seen_titles):
            continue
        unique.append(item)
        seen_urls.add(normalized_url)
        seen_titles.append(normalized_title)

    required = {"全球金融": 5, "AI行业": 5, "半导体重点": 3, "社媒趋势": 2}
    for category, minimum in required.items():
        count = sum(item.category == category for item in unique)
        if count < minimum:
            raise ValueError(f"Only {count} unique {category} candidates remain; need {minimum}")
    return unique, all_source_urls


def request_brief(
    brief_date: str,
    history: dict[str, Any],
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    final_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[DailyBrief, set[str], list[CandidateItem]]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")

    candidates, search_urls = collect_candidates(
        brief_date=brief_date,
        history=history,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        search_max_tokens=search_max_tokens,
        ledger=ledger,
    )
    candidate_json = json.dumps(
        [item.model_dump(mode="json") for item in candidates],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    history_json = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    prompt = f"""为 {brief_date}（北京时间）生成 Philo Daily Brief V3 的 JSON 数据。

已验证候选（只能从中选择，禁止添加候选之外的事实和链接）：
{candidate_json}

过去 7 天事件（用于新增/延续判断）：
{history_json}

输出 JSON Schema：
{json.dumps(DailyBrief.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

再次强调：全球金融 5 条、AI 行业 5 条、半导体 3 条、社媒 2 条；source_url、source_name、published_at 必须原样继承候选。只返回 JSON 对象。
"""
    payload, _response = call_dashscope_json(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=final_max_tokens,
        timeout_seconds=timeout_seconds,
        ledger=ledger,
        search_enabled=False,
    )
    brief = DailyBrief.model_validate(payload)
    candidate_urls = {normalize_url(item.source_url) for item in candidates}
    if not candidate_urls.issubset(search_urls):
        raise ValueError("Candidate source set is not a subset of DashScope search results")
    return brief, candidate_urls, candidates


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
    usage_history = load_usage_history(root / "data" / "usage.json")
    enforce_monthly_budget(usage_history, brief_date)
    ledger = UsageLedger()
    model = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
    candidates: list[CandidateItem] = []

    if args.input_json:
        LOG.info("Loading local test fixture; DashScope API is bypassed")
        brief = DailyBrief.model_validate_json(args.input_json.read_text(encoding="utf-8"))
        response_urls = {normalize_url(str(item.source_url)) for _, item in iter_news(brief)}
    else:
        base_http_api_url = os.environ.get("DASHSCOPE_BASE_HTTP_API_URL", "").strip()
        if base_http_api_url:
            parsed_base_url = urlsplit(base_http_api_url)
            if (
                parsed_base_url.scheme != "https"
                or not parsed_base_url.hostname
                or not parsed_base_url.hostname.endswith(".aliyuncs.com")
                or parsed_base_url.path.rstrip("/") != "/api/v1"
            ):
                raise ValueError(
                    "DASHSCOPE_BASE_HTTP_API_URL must be an HTTPS Alibaba Cloud "
                    "Model Studio /api/v1 endpoint"
                )
            dashscope.base_http_api_url = base_http_api_url.rstrip("/")
        timeout_seconds = int(os.environ.get("DASHSCOPE_TIMEOUT_SECONDS", "120"))
        search_max_tokens = int(os.environ.get("DASHSCOPE_SEARCH_MAX_TOKENS", "2400"))
        final_max_tokens = int(os.environ.get("DASHSCOPE_FINAL_MAX_TOKENS", "6000"))
        brief, response_urls, candidates = request_brief(
            brief_date,
            history,
            model,
            timeout_seconds,
            search_max_tokens,
            final_max_tokens,
            ledger,
        )

    if brief.date != brief_date:
        raise ValueError(f"Brief date mismatch: requested {brief_date}, received {brief.date}")
    classify_and_deduplicate(brief, history)
    validate_sources(brief, response_urls)
    validate_final_recency(brief, parsed_date)
    updated_history = update_history(brief, history)
    updated_usage = update_usage_history(usage_history, brief_date, model, ledger)
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
    atomic_write(
        output_dir / "usage.json",
        json.dumps(updated_usage, ensure_ascii=False, indent=2) + "\n",
    )
    if candidates:
        atomic_write(
            output_dir / "candidates.json",
            json.dumps(
                [item.model_dump(mode="json") for item in candidates],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    usage = ledger.as_record(brief_date, model)
    LOG.info(
        "DashScope usage: successful_calls=%d search_calls=%d input_tokens=%d output_tokens=%d estimated_usd=%.4f",
        usage["successful_calls"],
        usage["search_calls"],
        usage["input_tokens"],
        usage["output_tokens"],
        usage["estimated_total_cost_usd"],
    )
    LOG.info("Candidate generated: %s", output_dir / f"{brief_date}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import dashscope
from dashscope import Generation
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    source_url: str = Field(min_length=12, max_length=2048)
    published_at: str = Field(min_length=10, max_length=48)
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value


class CandidateItem(StrictModel):
    category: Literal["全球金融", "AI行业", "半导体重点", "社媒趋势"]
    status_hint: Literal["新增", "延续"]
    headline: str = Field(min_length=6, max_length=42)
    what_happened: str = Field(min_length=20, max_length=110)
    why_important: str = Field(min_length=20, max_length=110)
    source_name: str = Field(min_length=2, max_length=60)
    source_index: int = Field(ge=1)
    source_url: str = Field(min_length=12, max_length=2048)
    published_at: str = Field(min_length=10, max_length=48)
    published_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value


class CandidateBatch(StrictModel):
    candidates: list[CandidateItem] = Field(min_length=4, max_length=12)


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

MODEL_INPUT_CNY_PER_MILLION = 0.8
MODEL_OUTPUT_CNY_PER_MILLION = 2.0
SEARCH_ESTIMATE_CNY_PER_CALL = 0.05
CNY_PER_USD_ESTIMATE = 7.0
MONTHLY_BUDGET_USD = 5.0
MONTHLY_STOP_USD = 4.75


SEARCH_SYSTEM_PROMPT = """你是 Philo Daily Brief 的新闻候选编辑。你的唯一职责是基于本次联网搜索结果，筛选、排序并压缩候选事件为 JSON。

硬性规则：
1. 不能用模型记忆补充事实；每个候选必须来自本次搜索结果，source_url 必须原样使用搜索结果 URL。
2. 优先过去 24 小时；超过 48 小时只能作为过去 7 天事件的明确后续，且必须填写 continuation_of。
3. 不得虚构数字、来源、发布时间、链接或社媒热度；找不到明确发布日期的材料不要采用。
4. 同一事件只保留一个最佳来源。官方公告优先于二手报道；可靠媒体优先于聚合站和转载。
5. 只返回 JSON 对象，不写解释、Markdown 或代码块。
6. 你不能执行命令、修改文件或工作流、读取密钥，也不能使用图片、embedding、代码解释器或智能体工具。
"""


SYNTHESIS_SYSTEM_PROMPT = """你是 Philo Daily Brief V3 的中文研究编辑。你只基于输入的已验证候选资讯与过去 7 天事件，进行重要性判断、去重、新增/延续判断、摘要和正文结构化生成。

硬性规则：
1. 不联网，不使用模型记忆补充事实，不新增输入候选之外的来源、数字或事件。
2. 全球金融与 AI 行业各恰好 5 条；半导体恰好 3 条；社媒趋势恰好 2 条。
3. 每条资讯保留 status、发生了什么、为什么重要、来源、发布时间、可信度和可点击链接。
4. 同一事件不得跨栏目重复；标题与正文紧凑，全文适合 5 分钟阅读。
5. 社媒没有可靠量化数据时，明确说明限制；不得编造小红书、抖音或任何平台热度。
6. Philo Insight 是结构化归纳，不得冒充外部事实；重点跟踪方向不得虚构用户持仓。
7. 只返回符合给定 Schema 的 JSON 对象，不写 Markdown、代码块或额外解释。
8. 你不能执行命令、修改文件或工作流、读取密钥、访问本地文件，也不能使用任何工具。
"""


SEARCH_PLANS = (
    {
        "name": "全球金融",
        "categories": ("全球金融",),
        "minimum": {"全球金融": 5},
        "limit": 8,
        "focus": "全球宏观、央行、汇率、债券、能源、重要公司公告与监管动态",
        "priority": "Reuters、Bloomberg、FT、央行、监管机构、交易所、公司公告",
    },
    {
        "name": "AI行业",
        "categories": ("AI行业",),
        "minimum": {"AI行业": 5},
        "limit": 8,
        "focus": "基础模型、AI产品、企业采用、资本开支、算力基础设施和开源生态",
        "priority": "OpenAI、Anthropic、Google DeepMind、GitHub、HuggingFace 官方，其次 Reuters、Bloomberg、FT 与公司公告",
    },
    {
        "name": "半导体与社媒",
        "categories": ("半导体重点", "社媒趋势"),
        "minimum": {"半导体重点": 3, "社媒趋势": 2},
        "limit": 10,
        "focus": "先进制程、封装、设备、HBM，以及平台政策、内容分发、可信社媒行业趋势",
        "priority": "NVIDIA、TSMC、ASML、SEMI 与公司公告；社媒采用平台官方公告或 Reuters 等可靠媒体",
    },
)


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


class UsageLedger:
    def __init__(self) -> None:
        self.api_attempts = 0
        self.successful_calls = 0
        self.search_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, response: Any, *, search_enabled: bool) -> None:
        usage = getattr(response, "usage", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        self.successful_calls += 1
        self.search_calls += int(search_enabled)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        LOG.info(
            "DashScope call usage: search=%s input_tokens=%d output_tokens=%d",
            search_enabled,
            input_tokens,
            output_tokens,
        )

    def as_record(self, brief_date: str, model: str) -> dict[str, Any]:
        model_cost_cny = (
            self.input_tokens * MODEL_INPUT_CNY_PER_MILLION
            + self.output_tokens * MODEL_OUTPUT_CNY_PER_MILLION
        ) / 1_000_000
        search_cost_cny = self.search_calls * SEARCH_ESTIMATE_CNY_PER_CALL
        total_cny = model_cost_cny + search_cost_cny
        return {
            "date": brief_date,
            "model": model,
            "generation_count": 1,
            "api_attempts": self.api_attempts,
            "successful_calls": self.successful_calls,
            "search_calls": self.search_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_model_cost_cny": round(model_cost_cny, 6),
            "estimated_search_cost_cny": round(search_cost_cny, 6),
            "estimated_total_cost_cny": round(total_cny, 6),
            "estimated_total_cost_usd": round(total_cny / CNY_PER_USD_ESTIMATE, 6),
        }


def empty_usage_history() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pricing_note": "Conservative estimate; actual Alibaba Cloud billing and FX may differ.",
        "pricing_assumptions": {
            "qwen_plus_input_cny_per_million_tokens": MODEL_INPUT_CNY_PER_MILLION,
            "qwen_plus_output_cny_per_million_tokens": MODEL_OUTPUT_CNY_PER_MILLION,
            "web_search_estimate_cny_per_call": SEARCH_ESTIMATE_CNY_PER_CALL,
            "cny_per_usd_estimate": CNY_PER_USD_ESTIMATE,
            "monthly_budget_usd": MONTHLY_BUDGET_USD,
        },
        "days": [],
    }


def load_usage_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_usage_history()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("days"), list):
        raise ValueError("data/usage.json has an invalid days field")
    return data


def enforce_monthly_budget(history: dict[str, Any], brief_date: str) -> None:
    month = brief_date[:7]
    spent = sum(
        float(item.get("estimated_total_cost_usd", 0))
        for item in history.get("days", [])
        if item.get("date", "")[:7] == month
    )
    if spent >= MONTHLY_STOP_USD:
        raise RuntimeError(
            f"Estimated monthly API cost is already ${spent:.2f}; stopping before the ${MONTHLY_BUDGET_USD:.2f} budget"
        )


def update_usage_history(
    history: dict[str, Any], brief_date: str, model: str, ledger: UsageLedger
) -> dict[str, Any]:
    record = ledger.as_record(brief_date, model)
    days = []
    merged = False
    additive_fields = (
        "generation_count",
        "api_attempts",
        "successful_calls",
        "search_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_model_cost_cny",
        "estimated_search_cost_cny",
        "estimated_total_cost_cny",
        "estimated_total_cost_usd",
    )
    for item in history.get("days", []):
        if item.get("date") != brief_date:
            days.append(item)
            continue
        combined = dict(item)
        for field in additive_fields:
            combined[field] = round(float(item.get(field, 0)) + float(record.get(field, 0)), 6)
        for field in (
            "generation_count",
            "api_attempts",
            "successful_calls",
            "search_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            combined[field] = int(combined[field])
        combined["model"] = model
        days.append(combined)
        merged = True
    if not merged:
        days.append(record)
    days.sort(key=lambda item: item["date"], reverse=True)
    updated = empty_usage_history()
    updated["updated_at"] = datetime.now(BEIJING).isoformat(timespec="seconds")
    updated["days"] = days[:400]
    return updated


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
            f"DashScope search returned only {len(response_urls)} unique URLs; refusing to invent coverage"
        )
    missing = [
        str(item.source_url)
        for _category, item in iter_news(brief)
        if not source_matches(str(item.source_url), response_urls)
    ]
    if missing:
        raise ValueError(
            "Qwen returned source URLs absent from verified candidates: " + ", ".join(missing[:3])
        )


def match_history(
    headline: str, source_url: str, continuation_of: str | None, history: dict[str, Any]
) -> str | None:
    previous_by_id = {event["id"]: event for event in history["events"]}
    if continuation_of in previous_by_id:
        return continuation_of
    normalized_url = normalize_url(source_url)
    normalized_title = normalize_text(headline)
    for old in history["events"]:
        same_url = normalize_url(old.get("source_url", "")) == normalized_url
        similarity = SequenceMatcher(
            None, normalized_title, normalize_text(old.get("title", ""))
        ).ratio()
        if same_url or similarity >= 0.72:
            return old["id"]
    return None


def filter_candidate_recency(
    candidates: list[CandidateItem], brief_date: date, history: dict[str, Any]
) -> list[CandidateItem]:
    kept: list[CandidateItem] = []
    for item in candidates:
        published = date.fromisoformat(item.published_date)
        age_days = (brief_date - published).days
        matched_id = match_history(
            item.headline, item.source_url, item.continuation_of, history
        )
        if age_days < 0 or age_days > 6:
            LOG.warning("Dropping candidate outside the 7-day window: %s", item.headline)
            continue
        if age_days >= 2 and not matched_id:
            LOG.warning("Dropping news older than 48h without continuation: %s", item.headline)
            continue
        if matched_id:
            item.status_hint = "延续"
            item.continuation_of = matched_id
        else:
            item.status_hint = "新增"
            item.continuation_of = None
        kept.append(item)
    return kept


def validate_final_recency(brief: DailyBrief, brief_date: date) -> None:
    for _category, item in iter_news(brief):
        match = re.search(r"\d{4}-\d{2}-\d{2}", item.published_at)
        if not match:
            raise ValueError(f"Published time lacks an ISO date: {item.headline}")
        age_days = (brief_date - date.fromisoformat(match.group(0))).days
        if age_days < 0 or age_days > 6:
            raise ValueError(f"Source date outside the 7-day window: {item.headline}")
        if age_days >= 2 and item.status != "延续":
            raise ValueError(f"News older than 48h is not a continuation: {item.headline}")


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


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Qwen output is not a JSON object")
    return payload


def call_dashscope_json(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: int,
    ledger: UsageLedger,
    search_enabled: bool,
) -> tuple[dict[str, Any], Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        ledger.api_attempts += 1
        try:
            LOG.info(
                "Calling DashScope Generation API (attempt %d/2, model=%s, search=%s)",
                attempt + 1,
                model,
                search_enabled,
            )
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "model": model,
                "messages": messages,
                "result_format": "message",
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "request_timeout": timeout_seconds,
            }
            if search_enabled:
                kwargs["enable_search"] = True
                kwargs["search_options"] = {
                    "forced_search": True,
                    "search_strategy": "max",
                    "enable_source": True,
                    "enable_citation": True,
                    "citation_format": "[<number>]",
                    "freshness": 7,
                }
            response = Generation.call(**kwargs)
            if response.status_code != HTTPStatus.OK:
                error = RuntimeError(
                    f"DashScope response status={response.status_code} code={response.code}"
                )
                setattr(error, "status_code", response.status_code)
                setattr(error, "request_id", response.request_id)
                setattr(error, "error_code", response.code)
                raise error
            content = response.output.choices[0].message.content
            payload = parse_json_object(content)
            ledger.record(response, search_enabled=search_enabled)
            return payload, response
        except Exception as exc:
            last_error = exc
            LOG.error(
                "DashScope request failed: type=%s status=%s code=%s request_id=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
                getattr(exc, "error_code", None),
                getattr(exc, "request_id", None),
            )
            if attempt == 0:
                time.sleep(3)
    raise RuntimeError("DashScope API failed after one retry") from last_error


def search_result_records(response: Any) -> list[dict[str, Any]]:
    search_info = getattr(response.output, "search_info", None) or {}
    results = search_info.get("search_results", []) or []
    return [
        item
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("url"), str)
        and item["url"].startswith(("https://", "http://"))
    ]


def resolve_candidate_source(
    item: CandidateItem, search_results: list[dict[str, Any]]
) -> bool:
    indexed_matches = [
        result
        for result in search_results
        if str(result.get("index", "")).strip() == str(item.source_index)
    ]
    if len(indexed_matches) == 1:
        matched = indexed_matches[0]
        item.source_url = matched["url"]
        site_name = str(matched.get("site_name", "")).strip()
        if site_name:
            item.source_name = site_name[:60]
        return True

    source_urls = {normalize_url(result["url"]) for result in search_results}
    if source_matches(item.source_url, source_urls):
        return True

    candidate_host = urlsplit(item.source_url).netloc.lower()
    same_host = [
        result
        for result in search_results
        if urlsplit(result["url"]).netloc.lower() == candidate_host
    ]
    matches = same_host
    if not matches:
        source_name = normalize_text(item.source_name)
        if len(source_name) >= 2:
            matches = []
            for result in search_results:
                site_name = normalize_text(str(result.get("site_name", "")))
                host = normalize_text(urlsplit(result["url"]).netloc)
                if source_name in f"{site_name} {host}" or (
                    site_name and site_name in source_name
                ):
                    matches.append(result)
    if not matches:
        return False

    candidate_text = normalize_text(f"{item.headline} {item.what_happened}")
    best = max(
        matches,
        key=lambda result: SequenceMatcher(
            None,
            candidate_text,
            normalize_text(str(result.get("title", ""))),
        ).ratio(),
    )
    item.source_url = best["url"]
    site_name = str(best.get("site_name", "")).strip()
    if site_name:
        item.source_name = site_name[:60]
    return True


def collect_candidates(
    *,
    brief_date: str,
    history: dict[str, Any],
    api_key: str,
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[list[CandidateItem], set[str]]:
    compact_history = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    all_candidates: list[CandidateItem] = []
    all_source_urls: set[str] = set()

    for plan in SEARCH_PLANS:
        categories = "、".join(plan["categories"])
        prompt = f"""今天是 {brief_date}（北京时间）。联网搜索并为“{plan['name']}”建立新闻候选池。

检索主题：{plan['focus']}
来源优先级：{plan['priority']}
时效要求：优先过去 24 小时；过去 24–48 小时只在确有重要性时采用；更早内容只能是下列过去 7 天事件的明确后续。
过去 7 天事件：{compact_history}

输出 JSON Schema：
{json.dumps(CandidateBatch.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

要求：
- 输出 6 到 {plan['limit']} 个按重要性排序的候选，category 只能是：{categories}。
- source_index 必须是本次搜索角标 [n] 中的整数 n；一个候选只能引用一个最直接的来源。
- source_url 必须逐字来自同一角标对应的 DashScope 搜索来源；published_at 必须以 YYYY-MM-DD 开头并保留来源显示的时间/时区，published_date 为同一日期。
- 找不到来源发布日期、无法验证链接或只是旧闻重复的内容不要输出。
- 社媒只采用平台公告或可靠媒体；没有量化证据时不得写热度数字。
- 只返回 JSON 对象。
"""
        payload, response = call_dashscope_json(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=search_max_tokens,
            timeout_seconds=timeout_seconds,
            ledger=ledger,
            search_enabled=True,
        )
        batch = CandidateBatch.model_validate(payload)
        search_results = search_result_records(response)
        source_urls = {normalize_url(result["url"]) for result in search_results}
        if len(source_urls) < sum(plan["minimum"].values()):
            raise ValueError(
                f"{plan['name']} search returned only {len(source_urls)} source URLs"
            )
        invalid_categories = [
            item.category for item in batch.candidates if item.category not in plan["categories"]
        ]
        if invalid_categories:
            raise ValueError(f"Unexpected candidate categories in {plan['name']}")
        verified = [
            item
            for item in batch.candidates
            if resolve_candidate_source(item, search_results)
        ]
        discarded = len(batch.candidates) - len(verified)
        if discarded:
            LOG.warning(
                "Discarded %d %s candidate(s) whose URLs were absent from DashScope search",
                discarded,
                plan["name"],
            )
        filtered = filter_candidate_recency(
            verified, date.fromisoformat(brief_date), history
        )
        for category, minimum in plan["minimum"].items():
            count = sum(item.category == category for item in filtered)
            if count < minimum:
                raise ValueError(
                    f"{plan['name']} has only {count} usable {category} candidates; need {minimum}"
                )
        all_candidates.extend(filtered)
        all_source_urls.update(source_urls)

    unique: list[CandidateItem] = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    for item in all_candidates:
        normalized_url = normalize_url(item.source_url)
        normalized_title = normalize_text(item.headline)
        if normalized_url in seen_urls:
            continue
        if any(SequenceMatcher(None, normalized_title, old).ratio() >= 0.82 for old in seen_titles):
            continue
        unique.append(item)
        seen_urls.add(normalized_url)
        seen_titles.append(normalized_title)

    required = {"全球金融": 5, "AI行业": 5, "半导体重点": 3, "社媒趋势": 2}
    for category, minimum in required.items():
        count = sum(item.category == category for item in unique)
        if count < minimum:
            raise ValueError(f"Only {count} unique {category} candidates remain; need {minimum}")
    return unique, all_source_urls


def request_brief(
    brief_date: str,
    history: dict[str, Any],
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    final_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[DailyBrief, set[str], list[CandidateItem]]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")

    candidates, search_urls = collect_candidates(
        brief_date=brief_date,
        history=history,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        search_max_tokens=search_max_tokens,
        ledger=ledger,
    )
    candidate_json = json.dumps(
        [item.model_dump(mode="json") for item in candidates],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    history_json = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    prompt = f"""为 {brief_date}（北京时间）生成 Philo Daily Brief V3 的 JSON 数据。

已验证候选（只能从中选择，禁止添加候选之外的事实和链接）：
{candidate_json}

过去 7 天事件（用于新增/延续判断）：
{history_json}

输出 JSON Schema：
{json.dumps(DailyBrief.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

再次强调：全球金融 5 条、AI 行业 5 条、半导体 3 条、社媒 2 条；source_url、source_name、published_at 必须原样继承候选。只返回 JSON 对象。
"""
    payload, _response = call_dashscope_json(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=final_max_tokens,
        timeout_seconds=timeout_seconds,
        ledger=ledger,
        search_enabled=False,
    )
    brief = DailyBrief.model_validate(payload)
    candidate_urls = {normalize_url(item.source_url) for item in candidates}
    if not candidate_urls.issubset(search_urls):
        raise ValueError("Candidate source set is not a subset of DashScope search results")
    return brief, candidate_urls, candidates


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
    usage_history = load_usage_history(root / "data" / "usage.json")
    enforce_monthly_budget(usage_history, brief_date)
    ledger = UsageLedger()
    model = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
    candidates: list[CandidateItem] = []

    if args.input_json:
        LOG.info("Loading local test fixture; DashScope API is bypassed")
        brief = DailyBrief.model_validate_json(args.input_json.read_text(encoding="utf-8"))
        response_urls = {normalize_url(str(item.source_url)) for _, item in iter_news(brief)}
    else:
        base_http_api_url = os.environ.get("DASHSCOPE_BASE_HTTP_API_URL", "").strip()
        if base_http_api_url:
            parsed_base_url = urlsplit(base_http_api_url)
            if (
                parsed_base_url.scheme != "https"
                or not parsed_base_url.hostname
                or not parsed_base_url.hostname.endswith(".aliyuncs.com")
                or parsed_base_url.path.rstrip("/") != "/api/v1"
            ):
                raise ValueError(
                    "DASHSCOPE_BASE_HTTP_API_URL must be an HTTPS Alibaba Cloud "
                    "Model Studio /api/v1 endpoint"
                )
            dashscope.base_http_api_url = base_http_api_url.rstrip("/")
        timeout_seconds = int(os.environ.get("DASHSCOPE_TIMEOUT_SECONDS", "120"))
        search_max_tokens = int(os.environ.get("DASHSCOPE_SEARCH_MAX_TOKENS", "2400"))
        final_max_tokens = int(os.environ.get("DASHSCOPE_FINAL_MAX_TOKENS", "6000"))
        brief, response_urls, candidates = request_brief(
            brief_date,
            history,
            model,
            timeout_seconds,
            search_max_tokens,
            final_max_tokens,
            ledger,
        )

    if brief.date != brief_date:
        raise ValueError(f"Brief date mismatch: requested {brief_date}, received {brief.date}")
    classify_and_deduplicate(brief, history)
    validate_sources(brief, response_urls)
    validate_final_recency(brief, parsed_date)
    updated_history = update_history(brief, history)
    updated_usage = update_usage_history(usage_history, brief_date, model, ledger)
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
    atomic_write(
        output_dir / "usage.json",
        json.dumps(updated_usage, ensure_ascii=False, indent=2) + "\n",
    )
    if candidates:
        atomic_write(
            output_dir / "candidates.json",
            json.dumps(
                [item.model_dump(mode="json") for item in candidates],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    usage = ledger.as_record(brief_date, model)
    LOG.info(
        "DashScope usage: successful_calls=%d search_calls=%d input_tokens=%d output_tokens=%d estimated_usd=%.4f",
        usage["successful_calls"],
        usage["search_calls"],
        usage["input_tokens"],
        usage["output_tokens"],
        usage["estimated_total_cost_usd"],
    )
    LOG.info("Candidate generated: %s", output_dir / f"{brief_date}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import dashscope
from dashscope import Generation
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    source_url: str = Field(min_length=12, max_length=2048)
    published_at: str = Field(min_length=10, max_length=48)
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value


class CandidateItem(StrictModel):
    category: Literal["全球金融", "AI行业", "半导体重点", "社媒趋势"]
    status_hint: Literal["新增", "延续"]
    headline: str = Field(min_length=6, max_length=42)
    what_happened: str = Field(min_length=20, max_length=110)
    why_important: str = Field(min_length=20, max_length=110)
    source_name: str = Field(min_length=2, max_length=60)
    source_url: str = Field(min_length=12, max_length=2048)
    published_at: str = Field(min_length=10, max_length=48)
    published_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    confidence: Literal["高", "中", "低"]
    continuation_of: str | None

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value


class CandidateBatch(StrictModel):
    candidates: list[CandidateItem] = Field(min_length=4, max_length=12)


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

MODEL_INPUT_CNY_PER_MILLION = 0.8
MODEL_OUTPUT_CNY_PER_MILLION = 2.0
SEARCH_ESTIMATE_CNY_PER_CALL = 0.05
CNY_PER_USD_ESTIMATE = 7.0
MONTHLY_BUDGET_USD = 5.0
MONTHLY_STOP_USD = 4.75


SEARCH_SYSTEM_PROMPT = """你是 Philo Daily Brief 的新闻候选编辑。你的唯一职责是基于本次联网搜索结果，筛选、排序并压缩候选事件为 JSON。

硬性规则：
1. 不能用模型记忆补充事实；每个候选必须来自本次搜索结果，source_url 必须原样使用搜索结果 URL。
2. 优先过去 24 小时；超过 48 小时只能作为过去 7 天事件的明确后续，且必须填写 continuation_of。
3. 不得虚构数字、来源、发布时间、链接或社媒热度；找不到明确发布日期的材料不要采用。
4. 同一事件只保留一个最佳来源。官方公告优先于二手报道；可靠媒体优先于聚合站和转载。
5. 只返回 JSON 对象，不写解释、Markdown 或代码块。
6. 你不能执行命令、修改文件或工作流、读取密钥，也不能使用图片、embedding、代码解释器或智能体工具。
"""


SYNTHESIS_SYSTEM_PROMPT = """你是 Philo Daily Brief V3 的中文研究编辑。你只基于输入的已验证候选资讯与过去 7 天事件，进行重要性判断、去重、新增/延续判断、摘要和正文结构化生成。

硬性规则：
1. 不联网，不使用模型记忆补充事实，不新增输入候选之外的来源、数字或事件。
2. 全球金融与 AI 行业各恰好 5 条；半导体恰好 3 条；社媒趋势恰好 2 条。
3. 每条资讯保留 status、发生了什么、为什么重要、来源、发布时间、可信度和可点击链接。
4. 同一事件不得跨栏目重复；标题与正文紧凑，全文适合 5 分钟阅读。
5. 社媒没有可靠量化数据时，明确说明限制；不得编造小红书、抖音或任何平台热度。
6. Philo Insight 是结构化归纳，不得冒充外部事实；重点跟踪方向不得虚构用户持仓。
7. 只返回符合给定 Schema 的 JSON 对象，不写 Markdown、代码块或额外解释。
8. 你不能执行命令、修改文件或工作流、读取密钥、访问本地文件，也不能使用任何工具。
"""


SEARCH_PLANS = (
    {
        "name": "全球金融",
        "categories": ("全球金融",),
        "minimum": {"全球金融": 5},
        "limit": 8,
        "focus": "全球宏观、央行、汇率、债券、能源、重要公司公告与监管动态",
        "priority": "Reuters、Bloomberg、FT、央行、监管机构、交易所、公司公告",
    },
    {
        "name": "AI行业",
        "categories": ("AI行业",),
        "minimum": {"AI行业": 5},
        "limit": 8,
        "focus": "基础模型、AI产品、企业采用、资本开支、算力基础设施和开源生态",
        "priority": "OpenAI、Anthropic、Google DeepMind、GitHub、HuggingFace 官方，其次 Reuters、Bloomberg、FT 与公司公告",
    },
    {
        "name": "半导体与社媒",
        "categories": ("半导体重点", "社媒趋势"),
        "minimum": {"半导体重点": 3, "社媒趋势": 2},
        "limit": 10,
        "focus": "先进制程、封装、设备、HBM，以及平台政策、内容分发、可信社媒行业趋势",
        "priority": "NVIDIA、TSMC、ASML、SEMI 与公司公告；社媒采用平台官方公告或 Reuters 等可靠媒体",
    },
)


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


class UsageLedger:
    def __init__(self) -> None:
        self.api_attempts = 0
        self.successful_calls = 0
        self.search_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, response: Any, *, search_enabled: bool) -> None:
        usage = getattr(response, "usage", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        self.successful_calls += 1
        self.search_calls += int(search_enabled)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        LOG.info(
            "DashScope call usage: search=%s input_tokens=%d output_tokens=%d",
            search_enabled,
            input_tokens,
            output_tokens,
        )

    def as_record(self, brief_date: str, model: str) -> dict[str, Any]:
        model_cost_cny = (
            self.input_tokens * MODEL_INPUT_CNY_PER_MILLION
            + self.output_tokens * MODEL_OUTPUT_CNY_PER_MILLION
        ) / 1_000_000
        search_cost_cny = self.search_calls * SEARCH_ESTIMATE_CNY_PER_CALL
        total_cny = model_cost_cny + search_cost_cny
        return {
            "date": brief_date,
            "model": model,
            "generation_count": 1,
            "api_attempts": self.api_attempts,
            "successful_calls": self.successful_calls,
            "search_calls": self.search_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_model_cost_cny": round(model_cost_cny, 6),
            "estimated_search_cost_cny": round(search_cost_cny, 6),
            "estimated_total_cost_cny": round(total_cny, 6),
            "estimated_total_cost_usd": round(total_cny / CNY_PER_USD_ESTIMATE, 6),
        }


def empty_usage_history() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pricing_note": "Conservative estimate; actual Alibaba Cloud billing and FX may differ.",
        "pricing_assumptions": {
            "qwen_plus_input_cny_per_million_tokens": MODEL_INPUT_CNY_PER_MILLION,
            "qwen_plus_output_cny_per_million_tokens": MODEL_OUTPUT_CNY_PER_MILLION,
            "web_search_estimate_cny_per_call": SEARCH_ESTIMATE_CNY_PER_CALL,
            "cny_per_usd_estimate": CNY_PER_USD_ESTIMATE,
            "monthly_budget_usd": MONTHLY_BUDGET_USD,
        },
        "days": [],
    }


def load_usage_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_usage_history()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("days"), list):
        raise ValueError("data/usage.json has an invalid days field")
    return data


def enforce_monthly_budget(history: dict[str, Any], brief_date: str) -> None:
    month = brief_date[:7]
    spent = sum(
        float(item.get("estimated_total_cost_usd", 0))
        for item in history.get("days", [])
        if item.get("date", "")[:7] == month
    )
    if spent >= MONTHLY_STOP_USD:
        raise RuntimeError(
            f"Estimated monthly API cost is already ${spent:.2f}; stopping before the ${MONTHLY_BUDGET_USD:.2f} budget"
        )


def update_usage_history(
    history: dict[str, Any], brief_date: str, model: str, ledger: UsageLedger
) -> dict[str, Any]:
    record = ledger.as_record(brief_date, model)
    days = []
    merged = False
    additive_fields = (
        "generation_count",
        "api_attempts",
        "successful_calls",
        "search_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_model_cost_cny",
        "estimated_search_cost_cny",
        "estimated_total_cost_cny",
        "estimated_total_cost_usd",
    )
    for item in history.get("days", []):
        if item.get("date") != brief_date:
            days.append(item)
            continue
        combined = dict(item)
        for field in additive_fields:
            combined[field] = round(float(item.get(field, 0)) + float(record.get(field, 0)), 6)
        for field in (
            "generation_count",
            "api_attempts",
            "successful_calls",
            "search_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            combined[field] = int(combined[field])
        combined["model"] = model
        days.append(combined)
        merged = True
    if not merged:
        days.append(record)
    days.sort(key=lambda item: item["date"], reverse=True)
    updated = empty_usage_history()
    updated["updated_at"] = datetime.now(BEIJING).isoformat(timespec="seconds")
    updated["days"] = days[:400]
    return updated


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
            f"DashScope search returned only {len(response_urls)} unique URLs; refusing to invent coverage"
        )
    missing = [
        str(item.source_url)
        for _category, item in iter_news(brief)
        if not source_matches(str(item.source_url), response_urls)
    ]
    if missing:
        raise ValueError(
            "Qwen returned source URLs absent from verified candidates: " + ", ".join(missing[:3])
        )


def match_history(
    headline: str, source_url: str, continuation_of: str | None, history: dict[str, Any]
) -> str | None:
    previous_by_id = {event["id"]: event for event in history["events"]}
    if continuation_of in previous_by_id:
        return continuation_of
    normalized_url = normalize_url(source_url)
    normalized_title = normalize_text(headline)
    for old in history["events"]:
        same_url = normalize_url(old.get("source_url", "")) == normalized_url
        similarity = SequenceMatcher(
            None, normalized_title, normalize_text(old.get("title", ""))
        ).ratio()
        if same_url or similarity >= 0.72:
            return old["id"]
    return None


def filter_candidate_recency(
    candidates: list[CandidateItem], brief_date: date, history: dict[str, Any]
) -> list[CandidateItem]:
    kept: list[CandidateItem] = []
    for item in candidates:
        published = date.fromisoformat(item.published_date)
        age_days = (brief_date - published).days
        matched_id = match_history(
            item.headline, item.source_url, item.continuation_of, history
        )
        if age_days < 0 or age_days > 6:
            LOG.warning("Dropping candidate outside the 7-day window: %s", item.headline)
            continue
        if age_days >= 2 and not matched_id:
            LOG.warning("Dropping news older than 48h without continuation: %s", item.headline)
            continue
        if matched_id:
            item.status_hint = "延续"
            item.continuation_of = matched_id
        else:
            item.status_hint = "新增"
            item.continuation_of = None
        kept.append(item)
    return kept


def validate_final_recency(brief: DailyBrief, brief_date: date) -> None:
    for _category, item in iter_news(brief):
        match = re.search(r"\d{4}-\d{2}-\d{2}", item.published_at)
        if not match:
            raise ValueError(f"Published time lacks an ISO date: {item.headline}")
        age_days = (brief_date - date.fromisoformat(match.group(0))).days
        if age_days < 0 or age_days > 6:
            raise ValueError(f"Source date outside the 7-day window: {item.headline}")
        if age_days >= 2 and item.status != "延续":
            raise ValueError(f"News older than 48h is not a continuation: {item.headline}")


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


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Qwen output is not a JSON object")
    return payload


def call_dashscope_json(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: int,
    ledger: UsageLedger,
    search_enabled: bool,
) -> tuple[dict[str, Any], Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        ledger.api_attempts += 1
        try:
            LOG.info(
                "Calling DashScope Generation API (attempt %d/2, model=%s, search=%s)",
                attempt + 1,
                model,
                search_enabled,
            )
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "model": model,
                "messages": messages,
                "result_format": "message",
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "request_timeout": timeout_seconds,
            }
            if search_enabled:
                kwargs["enable_search"] = True
                kwargs["search_options"] = {
                    "forced_search": True,
                    "search_strategy": "max",
                    "enable_source": True,
                }
            response = Generation.call(**kwargs)
            if response.status_code != HTTPStatus.OK:
                error = RuntimeError(
                    f"DashScope response status={response.status_code} code={response.code}"
                )
                setattr(error, "status_code", response.status_code)
                setattr(error, "request_id", response.request_id)
                setattr(error, "error_code", response.code)
                raise error
            content = response.output.choices[0].message.content
            payload = parse_json_object(content)
            ledger.record(response, search_enabled=search_enabled)
            return payload, response
        except Exception as exc:
            last_error = exc
            LOG.error(
                "DashScope request failed: type=%s status=%s code=%s request_id=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
                getattr(exc, "error_code", None),
                getattr(exc, "request_id", None),
            )
            if attempt == 0:
                time.sleep(3)
    raise RuntimeError("DashScope API failed after one retry") from last_error


def search_result_records(response: Any) -> list[dict[str, Any]]:
    search_info = getattr(response.output, "search_info", None) or {}
    results = search_info.get("search_results", []) or []
    return [
        item
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("url"), str)
        and item["url"].startswith(("https://", "http://"))
    ]


def resolve_candidate_source(
    item: CandidateItem, search_results: list[dict[str, Any]]
) -> bool:
    source_urls = {normalize_url(result["url"]) for result in search_results}
    if source_matches(item.source_url, source_urls):
        return True

    candidate_host = urlsplit(item.source_url).netloc.lower()
    same_host = [
        result
        for result in search_results
        if urlsplit(result["url"]).netloc.lower() == candidate_host
    ]
    matches = same_host
    if not matches:
        source_name = normalize_text(item.source_name)
        if len(source_name) >= 2:
            matches = [
                result
                for result in search_results
                if source_name
                in normalize_text(
                    f"{result.get('site_name', '')} {urlsplit(result['url']).netloc}"
                )
                or normalize_text(str(result.get("site_name", ""))) in source_name
            ]
    if not matches:
        return False

    candidate_text = normalize_text(f"{item.headline} {item.what_happened}")
    best = max(
        matches,
        key=lambda result: SequenceMatcher(
            None,
            candidate_text,
            normalize_text(str(result.get("title", ""))),
        ).ratio(),
    )
    item.source_url = best["url"]
    site_name = str(best.get("site_name", "")).strip()
    if site_name:
        item.source_name = site_name[:60]
    return True


def collect_candidates(
    *,
    brief_date: str,
    history: dict[str, Any],
    api_key: str,
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[list[CandidateItem], set[str]]:
    compact_history = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    all_candidates: list[CandidateItem] = []
    all_source_urls: set[str] = set()

    for plan in SEARCH_PLANS:
        categories = "、".join(plan["categories"])
        prompt = f"""今天是 {brief_date}（北京时间）。联网搜索并为“{plan['name']}”建立新闻候选池。

检索主题：{plan['focus']}
来源优先级：{plan['priority']}
时效要求：优先过去 24 小时；过去 24–48 小时只在确有重要性时采用；更早内容只能是下列过去 7 天事件的明确后续。
过去 7 天事件：{compact_history}

输出 JSON Schema：
{json.dumps(CandidateBatch.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

要求：
- 输出 6 到 {plan['limit']} 个按重要性排序的候选，category 只能是：{categories}。
- source_url 必须逐字来自本次 DashScope 搜索来源；published_at 必须以 YYYY-MM-DD 开头并保留来源显示的时间/时区，published_date 为同一日期。
- 找不到来源发布日期、无法验证链接或只是旧闻重复的内容不要输出。
- 社媒只采用平台公告或可靠媒体；没有量化证据时不得写热度数字。
- 只返回 JSON 对象。
"""
        payload, response = call_dashscope_json(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=search_max_tokens,
            timeout_seconds=timeout_seconds,
            ledger=ledger,
            search_enabled=True,
        )
        batch = CandidateBatch.model_validate(payload)
        search_results = search_result_records(response)
        source_urls = {normalize_url(result["url"]) for result in search_results}
        if len(source_urls) < sum(plan["minimum"].values()):
            raise ValueError(
                f"{plan['name']} search returned only {len(source_urls)} source URLs"
            )
        invalid_categories = [
            item.category for item in batch.candidates if item.category not in plan["categories"]
        ]
        if invalid_categories:
            raise ValueError(f"Unexpected candidate categories in {plan['name']}")
        verified = [
            item
            for item in batch.candidates
            if resolve_candidate_source(item, search_results)
        ]
        discarded = len(batch.candidates) - len(verified)
        if discarded:
            LOG.warning(
                "Discarded %d %s candidate(s) whose URLs were absent from DashScope search",
                discarded,
                plan["name"],
            )
        filtered = filter_candidate_recency(
            verified, date.fromisoformat(brief_date), history
        )
        for category, minimum in plan["minimum"].items():
            count = sum(item.category == category for item in filtered)
            if count < minimum:
                raise ValueError(
                    f"{plan['name']} has only {count} usable {category} candidates; need {minimum}"
                )
        all_candidates.extend(filtered)
        all_source_urls.update(source_urls)

    unique: list[CandidateItem] = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    for item in all_candidates:
        normalized_url = normalize_url(item.source_url)
        normalized_title = normalize_text(item.headline)
        if normalized_url in seen_urls:
            continue
        if any(SequenceMatcher(None, normalized_title, old).ratio() >= 0.82 for old in seen_titles):
            continue
        unique.append(item)
        seen_urls.add(normalized_url)
        seen_titles.append(normalized_title)

    required = {"全球金融": 5, "AI行业": 5, "半导体重点": 3, "社媒趋势": 2}
    for category, minimum in required.items():
        count = sum(item.category == category for item in unique)
        if count < minimum:
            raise ValueError(f"Only {count} unique {category} candidates remain; need {minimum}")
    return unique, all_source_urls


def request_brief(
    brief_date: str,
    history: dict[str, Any],
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    final_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[DailyBrief, set[str], list[CandidateItem]]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")

    candidates, search_urls = collect_candidates(
        brief_date=brief_date,
        history=history,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        search_max_tokens=search_max_tokens,
        ledger=ledger,
    )
    candidate_json = json.dumps(
        [item.model_dump(mode="json") for item in candidates],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    history_json = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    prompt = f"""为 {brief_date}（北京时间）生成 Philo Daily Brief V3 的 JSON 数据。

已验证候选（只能从中选择，禁止添加候选之外的事实和链接）：
{candidate_json}

过去 7 天事件（用于新增/延续判断）：
{history_json}

输出 JSON Schema：
{json.dumps(DailyBrief.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

再次强调：全球金融 5 条、AI 行业 5 条、半导体 3 条、社媒 2 条；source_url、source_name、published_at 必须原样继承候选。只返回 JSON 对象。
"""
    payload, _response = call_dashscope_json(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=final_max_tokens,
        timeout_seconds=timeout_seconds,
        ledger=ledger,
        search_enabled=False,
    )
    brief = DailyBrief.model_validate(payload)
    candidate_urls = {normalize_url(item.source_url) for item in candidates}
    if not candidate_urls.issubset(search_urls):
        raise ValueError("Candidate source set is not a subset of DashScope search results")
    return brief, candidate_urls, candidates


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
    usage_history = load_usage_history(root / "data" / "usage.json")
    enforce_monthly_budget(usage_history, brief_date)
    ledger = UsageLedger()
    model = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
    candidates: list[CandidateItem] = []

    if args.input_json:
        LOG.info("Loading local test fixture; DashScope API is bypassed")
        brief = DailyBrief.model_validate_json(args.input_json.read_text(encoding="utf-8"))
        response_urls = {normalize_url(str(item.source_url)) for _, item in iter_news(brief)}
    else:
        base_http_api_url = os.environ.get("DASHSCOPE_BASE_HTTP_API_URL", "").strip()
        if base_http_api_url:
            parsed_base_url = urlsplit(base_http_api_url)
            if (
                parsed_base_url.scheme != "https"
                or not parsed_base_url.hostname
                or not parsed_base_url.hostname.endswith(".aliyuncs.com")
                or parsed_base_url.path.rstrip("/") != "/api/v1"
            ):
                raise ValueError(
                    "DASHSCOPE_BASE_HTTP_API_URL must be an HTTPS Alibaba Cloud "
                    "Model Studio /api/v1 endpoint"
                )
            dashscope.base_http_api_url = base_http_api_url.rstrip("/")
        timeout_seconds = int(os.environ.get("DASHSCOPE_TIMEOUT_SECONDS", "120"))
        search_max_tokens = int(os.environ.get("DASHSCOPE_SEARCH_MAX_TOKENS", "2400"))
        final_max_tokens = int(os.environ.get("DASHSCOPE_FINAL_MAX_TOKENS", "6000"))
        brief, response_urls, candidates = request_brief(
            brief_date,
            history,
            model,
            timeout_seconds,
            search_max_tokens,
            final_max_tokens,
            ledger,
        )

    if brief.date != brief_date:
        raise ValueError(f"Brief date mismatch: requested {brief_date}, received {brief.date}")
    classify_and_deduplicate(brief, history)
    validate_sources(brief, response_urls)
    validate_final_recency(brief, parsed_date)
    updated_history = update_history(brief, history)
    updated_usage = update_usage_history(usage_history, brief_date, model, ledger)
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
    atomic_write(
        output_dir / "usage.json",
        json.dumps(updated_usage, ensure_ascii=False, indent=2) + "\n",
    )
    if candidates:
        atomic_write(
            output_dir / "candidates.json",
            json.dumps(
                [item.model_dump(mode="json") for item in candidates],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    usage = ledger.as_record(brief_date, model)
    LOG.info(
        "DashScope usage: successful_calls=%d search_calls=%d input_tokens=%d output_tokens=%d estimated_usd=%.4f",
        usage["successful_calls"],
        usage["search_calls"],
        usage["input_tokens"],
        usage["output_tokens"],
        usage["estimated_total_cost_usd"],
    )
    LOG.info("Candidate generated: %s", output_dir / f"{brief_date}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
