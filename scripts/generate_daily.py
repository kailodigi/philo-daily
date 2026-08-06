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
from collections import Counter
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from quality_gate import SOCIAL_SHORTFALL_NOTICE, assert_brief_quality
except ModuleNotFoundError:  # Imported as scripts.generate_daily by offline tests.
    from scripts.quality_gate import SOCIAL_SHORTFALL_NOTICE, assert_brief_quality


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
    finance_lens: Literal["市场", "宏观", "政策", "公司", "行业", "资本事件"] | None = None
    importance_score: int | None = Field(default=None, ge=1, le=100)

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value

    @model_validator(mode="after")
    def require_finance_ranking_fields(self) -> "CandidateItem":
        if self.category == "全球金融" and (
            self.finance_lens is None or self.importance_score is None
        ):
            raise ValueError(
                "global finance candidates require finance_lens and importance_score"
            )
        return self


class CandidateBatch(StrictModel):
    candidates: list[CandidateItem] = Field(default_factory=list, max_length=12)


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

    @field_validator("impact", "watch")
    @classmethod
    def reject_unsourced_numbers(cls, value: str) -> str:
        if re.search(r"\d", value):
            raise ValueError("tracking impacts must be qualitative and contain no digits")
        return value


class DailyBrief(StrictModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    title: str = Field(min_length=4, max_length=40)
    summary: str = Field(min_length=30, max_length=150)
    core_conclusions: list[Annotated[str, Field(min_length=20, max_length=100)]] = Field(
        min_length=3, max_length=3
    )
    signal_board: list[Signal] = Field(min_length=4, max_length=6)
    global_finance: list[NewsItem] = Field(min_length=3, max_length=5)
    ai_industry: list[NewsItem] = Field(min_length=3, max_length=5)
    semiconductors: list[NewsItem] = Field(min_length=2, max_length=3)
    social_trends: list[NewsItem] = Field(default_factory=list, max_length=5)
    social_limit_notice: str | None = Field(default=None, max_length=40)
    philo_insight: str = Field(min_length=80, max_length=420)
    tomorrow_watch: list[WatchItem] = Field(min_length=4, max_length=5)
    tracking_impacts: list[ImpactItem] = Field(min_length=3, max_length=5)
    data_limitations: list[Annotated[str, Field(min_length=12, max_length=120)]] = Field(
        min_length=1, max_length=4
    )

    @model_validator(mode="after")
    def require_social_shortfall_notice(self) -> "DailyBrief":
        count = len(self.social_trends)
        if count < 3 and self.social_limit_notice != SOCIAL_SHORTFALL_NOTICE:
            raise ValueError(
                f"fewer than 3 social items requires: {SOCIAL_SHORTFALL_NOTICE}"
            )
        if 3 <= count <= 5 and self.social_limit_notice not in (None, ""):
            raise ValueError("social_limit_notice must be empty when 3–5 items exist")
        return self


CATEGORY_FIELDS = (
    ("全球金融", "global_finance"),
    ("AI行业", "ai_industry"),
    ("半导体重点", "semiconductors"),
    ("社媒趋势", "social_trends"),
)

FINANCE_MARKET_LENSES = frozenset({"市场", "宏观", "政策"})
FINANCE_COMPANY_LENSES = frozenset({"公司", "行业"})
FINANCE_SEARCH_MIN_VALIDATED = 3
FINANCE_SEARCH_TARGET_VALIDATED = 5
UNKNOWN_SOURCE_NAMES = frozenset(
    {"unknown", "unknownsource", "na", "none", "未知", "未知来源", "不详"}
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
2. 全球金融与 AI 行业目标各 5 条、最低各 3 条，只能使用已验证候选，候选不足时不得用旧闻或低可信来源补足；半导体按已验证候选供给输出 2–3 条；社媒趋势只使用可靠候选，最多 5 条。社媒少于 3 条时不得凑数，social_limit_notice 必须逐字填写“今日可靠社媒趋势不足”。
3. 每条资讯保留 status、发生了什么、为什么重要、来源、发布时间、可信度和可点击链接。
4. 同一事件不得跨栏目重复；标题与正文紧凑，全文适合 5 分钟阅读。
5. 社媒没有可靠量化数据时，明确说明限制；不得编造小红书、抖音或任何平台热度。
6. Philo Insight 是结构化归纳，不得冒充外部事实；重点跟踪方向不得虚构用户持仓，impact 与 watch 必须是纯定性表述且不得包含数字。
7. 只返回符合给定 Schema 的 JSON 对象，不写 Markdown、代码块或额外解释。
8. 你不能执行命令、修改文件或工作流、读取密钥、访问本地文件，也不能使用任何工具。
"""


SEARCH_PLANS = (
    {
        "name": "统一金融候选池",
        "prefix": "G",
        "categories": ("全球金融",),
        "minimum": {"全球金融": 3},
        "limit": 8,
        "output_min": 3,
        "finance_structure": True,
        "search_rounds": (
            {
                "query_group": "全球市场/宏观/中国政策",
                "queries": (
                    "global markets stocks bonds currencies today Reuters",
                    "US stocks close today Reuters",
                    "European stocks today Reuters",
                    "Asia markets today Reuters",
                    "gold oil dollar today Reuters",
                    "oil gold dollar bond yields today Reuters",
                    "PBOC policy today Reuters",
                    "China stocks today Reuters",
                ),
            },
            {
                "query_group": "公司/行业/资本事件",
                "queries": (
                    "company earnings today Reuters",
                    "company guidance today Reuters",
                    "company investment today Reuters",
                    "company acquisition today Reuters",
                    "company partnership today Reuters",
                ),
            },
            {
                "query_group": "AI/半导体/中国科技/港股",
                "queries": (
                    "semiconductor company news today Reuters",
                    "AI chip company today Reuters",
                    "AI chip company news today Reuters",
                    "NVIDIA AMD TSMC news today",
                    "memory chip market today",
                    "advanced packaging semiconductor today",
                    "Hong Kong stocks today Reuters",
                    "China technology stocks today Reuters",
                    "China semiconductor today Reuters",
                ),
            },
        ),
        "preflight_min_validated": FINANCE_SEARCH_MIN_VALIDATED,
        "target_validated": FINANCE_SEARCH_TARGET_VALIDATED,
        "focus": "统一覆盖全球金融的宏观、市场、公司、行业、政策与资本事件，不为任何固定子类预留数量",
        "priority": "Reuters、Bloomberg、FT、央行、监管机构、交易所与公司投资者关系公告",
        "sites": (
            "reuters.com", "bloomberg.com", "ft.com", "federalreserve.gov",
            "ecb.europa.eu", "bankofengland.co.uk", "imf.org", "bis.org",
            "pbc.gov.cn", "gov.cn", "csrc.gov.cn", "sse.com.cn", "szse.cn",
            "hkex.com.hk", "news.cn", "wsj.com", "cnbc.com", "apnews.com",
            "spglobal.com", "nasdaq.com", "caixin.com", "yicai.com",
            "nbd.com.cn", "sina.com.cn", "eastmoney.com", "sec.gov",
            "amazon.com", "microsoft.com",
            "apple.com", "nvidia.com", "investor.fb.com", "abc.xyz",
            "tsmc.com", "asml.com", "amd.com", "intel.com", "micron.com",
            "samsung.com", "skhynix.com", "semi.org",
            "bankofchina.com",
        ),
    },
    {
        "name": "AI行业",
        "prefix": "A",
        "categories": ("AI行业",),
        "minimum": {"AI行业": 3},
        "limit": 8,
        "output_min": 3,
        "target_total": 5,
        "focus": "基础模型、AI产品、企业采用、资本开支、算力基础设施和开源生态",
        "priority": "OpenAI、Anthropic、Google DeepMind、GitHub、HuggingFace 官方，其次 Reuters、Bloomberg、FT 与公司公告",
        "sites": (
            "openai.com", "anthropic.com", "deepmind.google", "blog.google",
            "github.blog", "huggingface.co", "microsoft.com", "nvidia.com",
            "reuters.com", "bloomberg.com", "ft.com", "news.cn",
            "techcrunch.com", "wired.com", "arstechnica.com", "venturebeat.com",
            "theverge.com", "nature.com", "ai.google", "meta.com", "about.fb.com",
            "apple.com", "amazon.science", "aws.amazon.com",
            "sina.com.cn",
        ),
    },
    {
        "name": "半导体重点",
        "prefix": "S",
        "categories": ("半导体重点",),
        "minimum": {"半导体重点": 2},
        "limit": 6,
        "target_total": 3,
        "focus": "先进制程、先进封装、设备、HBM、AI芯片与关键公司公告",
        "priority": "NVIDIA、TSMC、ASML、SEMI 与公司公告，其次 Reuters、Bloomberg、FT 与专业产业媒体",
        "sites": (
            "nvidia.com", "tsmc.com", "asml.com", "semi.org", "onsemi.com",
            "intel.com", "amd.com", "micron.com", "samsung.com", "skhynix.com",
            "reuters.com", "bloomberg.com", "ft.com", "news.cn", "gov.cn",
            "semiwiki.com", "digitimes.com", "tomshardware.com", "bjnews.com.cn",
            "hubeidaily.net", "jfdaily.com", "10jqka.com.cn", "dfcfw.com",
            "tmtpost.com", "stcn.com",
        ),
    },
    {
        "name": "社媒趋势",
        "prefix": "T",
        "categories": ("社媒趋势",),
        "minimum": {"社媒趋势": 0},
        "limit": 6,
        "output_min": 0,
        "target_total": 5,
        "focus": "社交平台政策、内容分发、创作者生态、AI内容治理与可信社媒行业趋势",
        "priority": "平台官方公告优先，其次 Reuters、Bloomberg、FT 与可靠社媒行业媒体",
        "sites": (
            "linkedin.com", "about.linkedin.com", "newsroom.tiktok.com",
            "about.fb.com", "blog.youtube", "x.com", "redditinc.com",
            "reuters.com", "bloomberg.com", "ft.com", "apnews.com", "news.cn",
            "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
            "socialmediatoday.com", "searchengineland.com", "socialsamosa.com",
            "ppc.land", "adweek.com", "digiday.com", "platformer.news",
            "36kr.com", "sina.com.cn",
        ),
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

    def as_record(
        self,
        brief_date: str,
        model: str,
        *,
        succeeded: bool,
        workflow_run_id: str,
    ) -> dict[str, Any]:
        model_cost_cny = (
            self.input_tokens * MODEL_INPUT_CNY_PER_MILLION
            + self.output_tokens * MODEL_OUTPUT_CNY_PER_MILLION
        ) / 1_000_000
        search_cost_cny = self.search_calls * SEARCH_ESTIMATE_CNY_PER_CALL
        total_cny = model_cost_cny + search_cost_cny
        return {
            "date": brief_date,
            "workflow_run_id": workflow_run_id,
            "model": model,
            "status": "success" if succeeded else "failure",
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


def empty_usage_history(month: str = "") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "month": month,
        "pricing_note": "Conservative estimate; actual Alibaba Cloud billing and FX may differ.",
        "pricing_assumptions": {
            "qwen_plus_input_cny_per_million_tokens": MODEL_INPUT_CNY_PER_MILLION,
            "qwen_plus_output_cny_per_million_tokens": MODEL_OUTPUT_CNY_PER_MILLION,
            "web_search_estimate_cny_per_call": SEARCH_ESTIMATE_CNY_PER_CALL,
            "cny_per_usd_estimate": CNY_PER_USD_ESTIMATE,
            "monthly_budget_usd": MONTHLY_BUDGET_USD,
        },
        "runs": [],
    }


def load_usage_history(path: Path, month: str) -> dict[str, Any]:
    if not path.exists():
        return empty_usage_history(month)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("month") != month or not isinstance(data.get("runs"), list):
        raise ValueError(f"{path} is not a valid usage file for {month}")
    return data


def enforce_monthly_budget(history: dict[str, Any], brief_date: str) -> None:
    month = brief_date[:7]
    spent = sum(
        float(item.get("estimated_total_cost_usd", 0))
        for item in history.get("runs", [])
        if item.get("date", "")[:7] == month
    )
    if spent >= MONTHLY_STOP_USD:
        raise RuntimeError(
            f"Estimated monthly API cost is already ${spent:.2f}; stopping before the ${MONTHLY_BUDGET_USD:.2f} budget"
        )


def update_usage_history(
    history: dict[str, Any],
    brief_date: str,
    model: str,
    ledger: UsageLedger,
    *,
    succeeded: bool,
    workflow_run_id: str,
) -> dict[str, Any]:
    record = ledger.as_record(
        brief_date,
        model,
        succeeded=succeeded,
        workflow_run_id=workflow_run_id,
    )
    runs = [
        item
        for item in history.get("runs", [])
        if str(item.get("workflow_run_id")) != workflow_run_id
    ]
    runs.append(record)
    runs.sort(
        key=lambda item: (str(item.get("date", "")), str(item.get("workflow_run_id", ""))),
        reverse=True,
    )
    updated = empty_usage_history(brief_date[:7])
    updated["updated_at"] = datetime.now(BEIJING).isoformat(timespec="seconds")
    updated["runs"] = runs[:400]
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
    host = parts.netloc.lower()
    for prefix in ("www.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    return urlunsplit(("https", host, path, "", ""))


def source_host_allowed(url: str, allowed_sites: tuple[str, ...]) -> bool:
    host = urlsplit(url).netloc.lower().split(":", 1)[0]
    return any(host == site or host.endswith("." + site) for site in allowed_sites)


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
    required_urls = sum(1 for _category, _item in iter_news(brief))
    if len(response_urls) < required_urls:
        raise ValueError(
            "Verified search source coverage is insufficient: "
            f"unique_urls={len(response_urls)}; news_items={required_urls}"
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


def lock_verified_source_fields(
    brief: DailyBrief, candidates: list[CandidateItem]
) -> None:
    if not candidates:
        return
    verified = {
        (item.category, normalize_url(item.source_url)): item for item in candidates
    }
    for category, item in iter_news(brief):
        candidate = verified.get((category, normalize_url(item.source_url)))
        if candidate is None:
            raise ValueError(
                f"Final item is not linked to a verified {category} candidate"
            )
        item.source_name = candidate.source_name
        item.source_url = candidate.source_url
        item.published_at = candidate.published_at


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
        and urlsplit(item["url"]).path.rstrip("/")
    ]


def source_date_from_result(result: dict[str, Any]) -> str | None:
    for key in ("published_at", "publish_time", "published_time", "pub_date", "date"):
        value = str(result.get(key, ""))
        match = re.search(r"20\d{2}-\d{2}-\d{2}", value)
        if match:
            return match.group(0)
    url = result.get("url", "")
    for pattern in (
        r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)",
        r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)",
    ):
        match = re.search(pattern, url)
        if match:
            try:
                return date(
                    int(match.group(1)), int(match.group(2)), int(match.group(3))
                ).isoformat()
            except ValueError:
                continue
    return None


def finance_source_result_allowed(
    result: dict[str, Any],
    allowed_sites: tuple[str, ...],
    brief_day: date,
) -> tuple[bool, str]:
    """Apply the finance source allowlist plus page-level Bank of China checks."""
    url = str(result.get("url", ""))
    if not source_host_allowed(url, allowed_sites):
        return False, "untrusted_domain"
    host = urlsplit(url).netloc.lower().split(":", 1)[0]
    if not (host == "bankofchina.com" or host.endswith(".bankofchina.com")):
        return True, ""

    page_text = " ".join(
        str(result.get(key, ""))
        for key in ("title", "snippet", "description", "site_name")
    ).casefold()
    marketing_markers = (
        "营销", "优惠", "促销", "活动", "信用卡", "产品推荐", "理财产品",
        "promotion", "campaign", "special offer", "wealth product",
    )
    official_markers = (
        "公告", "声明", "报告", "市场数据", "外汇牌价", "经济金融",
        "official", "announcement", "report", "market data", "exchange rate",
    )
    if any(marker in page_text for marker in marketing_markers):
        return False, "bankofchina_marketing_or_product"
    if not any(marker in page_text for marker in official_markers):
        return False, "bankofchina_non_official_page"
    published = source_date_from_result(result)
    if not published:
        return False, "bankofchina_missing_published_date"
    age_days = (brief_day - date.fromisoformat(published)).days
    if age_days < 0 or age_days > 6:
        return False, "bankofchina_stale_or_future"
    return True, ""


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
    candidate_text = normalize_text(f"{item.headline} {item.what_happened}")
    if matches:
        best = max(
            matches,
            key=lambda result: SequenceMatcher(
                None,
                candidate_text,
                normalize_text(str(result.get("title", ""))),
            ).ratio(),
        )
    else:
        ranked = sorted(
            (
                (
                    SequenceMatcher(
                        None,
                        normalize_text(item.headline),
                        normalize_text(str(result.get("title", ""))),
                    ).ratio(),
                    result,
                )
                for result in search_results
                if normalize_text(str(result.get("title", "")))
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.42:
            return False
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.05:
            return False
        best = ranked[0][1]
    item.source_url = best["url"]
    site_name = str(best.get("site_name", "")).strip()
    if site_name:
        item.source_name = site_name[:60]
    return True


def finance_search_preflight_errors(
    *,
    candidate_count: int,
    validated_candidates: list[CandidateItem],
    minimum_validated: int = FINANCE_SEARCH_MIN_VALIDATED,
) -> list[str]:
    """Validate the aggregated finance search pool before final Qwen synthesis."""
    validated_count = len(validated_candidates)
    errors: list[str] = []
    if validated_count < minimum_validated:
        errors.append(
            "total candidates insufficient: "
            f"validated_count={validated_count}; need {minimum_validated}"
        )
    return errors


def filter_finance_recency_with_reasons(
    candidates: list[CandidateItem],
    brief_day: date,
    history: dict[str, Any],
) -> tuple[list[CandidateItem], Counter[str]]:
    kept: list[CandidateItem] = []
    reasons: Counter[str] = Counter()
    for item in candidates:
        published = date.fromisoformat(item.published_date)
        age_days = (brief_day - published).days
        matched_id = match_history(
            item.headline, item.source_url, item.continuation_of, history
        )
        if age_days < 0:
            reasons["future_published_date"] += 1
            continue
        if age_days > 6:
            reasons["outside_7_day_window"] += 1
            continue
        if age_days >= 2 and not matched_id:
            reasons["older_than_48h_without_continuation"] += 1
            continue
        if matched_id:
            item.status_hint = "延续"
            item.continuation_of = matched_id
        else:
            item.status_hint = "新增"
            item.continuation_of = None
        kept.append(item)
    return kept, reasons


def format_filtered_reason_summary(reasons: Counter[str]) -> str:
    if not reasons:
        return "none"
    return ",".join(f"{key}={reasons[key]}" for key in sorted(reasons))


def collect_finance_candidates(
    *,
    plan: dict[str, Any],
    brief_date: str,
    compact_history: str,
    history: dict[str, Any],
    api_key: str,
    model: str,
    timeout_seconds: int,
    search_max_tokens: int,
    ledger: UsageLedger,
) -> tuple[list[CandidateItem], set[str]]:
    """Run up to three finance rounds; publish 3–5 verified candidates."""
    brief_day = date.fromisoformat(brief_date)
    eligible_drafts: list[CandidateItem] = []
    trusted_results_by_url: dict[str, dict[str, Any]] = {}
    cumulative_raw_count = 0
    cumulative_source_bound_count = 0
    cumulative_timely_count = 0
    final_errors: list[str] = []

    for round_number, search_round in enumerate(plan["search_rounds"], start=1):
        query_group = search_round["query_group"]
        query_lines = "\n".join(f"- {query}" for query in search_round["queries"])
        prompt = f"""今天是 {brief_date}（北京时间）。联网搜索并为“{plan['name']}”执行第 {round_number} 轮“{query_group}”检索。

检索主题：{plan['focus']}
本轮 query 组合（逐项覆盖）：
{query_lines}
来源优先级：{plan['priority']}
时效要求：优先过去 24 小时；过去 24–48 小时只在确有重要性时采用；更早内容只能是下列过去 7 天事件的明确后续。
过去 7 天事件：{compact_history}

输出 JSON Schema：
{json.dumps(CandidateBatch.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

要求：
- 目标输出 5 个、最多 {plan['limit']} 个按重要性排序的全球金融候选；若真实合格来源不足，可输出 {plan['output_min']}–4 个，不得用旧闻、低可信来源或未绑定链接补足。不同轮次只负责扩大检索覆盖，不设置固定子类别配额。
- 每条必须填写 finance_lens（市场/宏观/政策/公司/行业/资本事件）与 importance_score（1–100）。
- source_index 必须是本次搜索角标 [n] 中的整数 n；一个候选只能引用一个最直接的来源。
- source_url 必须逐字来自同一角标对应的 DashScope 搜索来源；published_at 必须以 YYYY-MM-DD 开头并保留来源显示的时间/时区，published_date 为同一日期。
- 找不到来源发布日期、无法验证链接或只是旧闻重复的内容不要输出。
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
        try:
            batch = CandidateBatch.model_validate(payload)
        except Exception as exc:
            raise ValueError(
                f"{plan['name']} round={round_number} candidate JSON failed schema validation"
            ) from exc

        raw_search_results = search_result_records(response)
        cumulative_raw_count += len(raw_search_results)
        round_reasons: Counter[str] = Counter()
        trusted_results: list[dict[str, Any]] = []
        for result in raw_search_results:
            allowed, reason = finance_source_result_allowed(
                result, plan["sites"], brief_day
            )
            if allowed:
                trusted_results.append(result)
            else:
                round_reasons[reason] += 1
        trusted_results = trusted_results[: plan["limit"] + 4]

        for result in trusted_results:
            trusted_results_by_url.setdefault(normalize_url(result["url"]), result)
        bound_candidates: list[CandidateItem] = []
        for item in batch.candidates:
            if item.category not in plan["categories"]:
                round_reasons["unexpected_category"] += 1
                continue
            if not resolve_candidate_source(item, trusted_results):
                round_reasons["unbound_source_url"] += 1
                continue
            matched_result = next(
                (
                    result
                    for result in trusted_results
                    if normalize_url(result["url"])
                    == normalize_url(item.source_url)
                ),
                None,
            )
            if matched_result is None:
                round_reasons["unbound_source_url"] += 1
                continue
            if normalize_text(item.source_name) in UNKNOWN_SOURCE_NAMES:
                round_reasons["unknown_source"] += 1
                continue
            source_date = source_date_from_result(matched_result)
            if source_date:
                item.published_date = source_date
                item.published_at = source_date
            bound_candidates.append(item)

        cumulative_source_bound_count += len(bound_candidates)
        timely_candidates, recency_reasons = filter_finance_recency_with_reasons(
            bound_candidates, brief_day, history
        )
        round_reasons.update(recency_reasons)
        cumulative_timely_count += len(timely_candidates)
        previous_validated_count = len(eligible_drafts)
        deduplicated = deduplicate_ranked_candidates(
            eligible_drafts + timely_candidates
        )
        duplicate_count = (
            previous_validated_count + len(timely_candidates) - len(deduplicated)
        )
        if duplicate_count:
            round_reasons["duplicate"] += duplicate_count
        deduplicated_count = len(deduplicated) - previous_validated_count
        eligible_drafts = deduplicated

        market_count = sum(
            item.finance_lens in FINANCE_MARKET_LENSES
            for item in eligible_drafts
        )
        company_count = sum(
            item.finance_lens in FINANCE_COMPANY_LENSES
            for item in eligible_drafts
        )
        final_errors = finance_search_preflight_errors(
            candidate_count=len(trusted_results_by_url),
            validated_candidates=eligible_drafts,
            minimum_validated=int(plan["preflight_min_validated"]),
        )
        LOG.info(
            "finance_search_round round_number=%d query_group=%s "
            "raw_candidate_count=%d source_bound_count=%d timely_count=%d "
            "deduplicated_count=%d market_macro_policy_count=%d "
            "company_industry_count=%d cumulative_validated_count=%d "
            "filtered_reason_summary=%s",
            round_number,
            query_group,
            len(raw_search_results),
            len(bound_candidates),
            len(timely_candidates),
            deduplicated_count,
            market_count,
            company_count,
            len(eligible_drafts),
            format_filtered_reason_summary(round_reasons),
        )
        if len(eligible_drafts) >= int(plan["target_validated"]):
            LOG.info(
                "finance_search_stop round_number=%d validated_count=%d "
                "target_validated=%d reason=target_reached",
                round_number,
                len(eligible_drafts),
                int(plan["target_validated"]),
            )
            break

    if final_errors:
        failure_reasons = list(final_errors)
        minimum_validated = int(plan["preflight_min_validated"])
        if cumulative_source_bound_count < minimum_validated:
            failure_reasons.append(
                "source binding insufficient: "
                f"source_bound_count={cumulative_source_bound_count}"
            )
        elif cumulative_timely_count < minimum_validated:
            failure_reasons.append(
                "time filtering removed too many candidates: "
                f"timely_count={cumulative_timely_count}"
            )
        elif len(eligible_drafts) < minimum_validated:
            failure_reasons.append(
                "deduplication left too few candidates: "
                f"deduplicated_count={len(eligible_drafts)}"
            )
        failure_reason = "; ".join(failure_reasons)
        LOG.error(
            "finance_search_failed candidate_count=%d validated_count=%d "
            "raw_candidate_count=%d source_bound_count=%d timely_count=%d "
            "failure_reason=%s",
            len(trusted_results_by_url),
            len(eligible_drafts),
            cumulative_raw_count,
            cumulative_source_bound_count,
            cumulative_timely_count,
            failure_reason,
        )
        raise ValueError(
            f"{plan['name']} search preflight failed: "
            f"candidate_count={len(trusted_results_by_url)}; "
            f"validated_count={len(eligible_drafts)}; "
            f"failure_reason={failure_reason}"
        )
    if len(eligible_drafts) < int(plan["target_validated"]):
        LOG.info(
            "finance_search_partial validated_count=%d target_validated=%d "
            "reason=minimum_publishable_satisfied_after_final_round",
            len(eligible_drafts),
            int(plan["target_validated"]),
        )
    return eligible_drafts, set(trusted_results_by_url)


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
    plan_pools: dict[str, list[CandidateItem]] = {}
    all_source_urls: set[str] = set()
    brief_day = date.fromisoformat(brief_date)

    for plan in SEARCH_PLANS:
        if plan.get("finance_structure"):
            finance_candidates, finance_urls = collect_finance_candidates(
                plan=plan,
                brief_date=brief_date,
                compact_history=compact_history,
                history=history,
                api_key=api_key,
                model=model,
                timeout_seconds=timeout_seconds,
                search_max_tokens=search_max_tokens,
                ledger=ledger,
            )
            plan_pools.setdefault(plan["prefix"], []).extend(finance_candidates)
            all_source_urls.update(finance_urls)
            continue
        categories = "、".join(plan["categories"])
        minimums = "、".join(
            f"{category}至少{minimum}条"
            for category, minimum in plan["minimum"].items()
        )
        query_lines = "\n".join(
            f"- {query}" for query in plan.get("queries", ())
        )
        query_block = f"检索 query 组合（全部覆盖）：\n{query_lines}\n" if query_lines else ""
        finance_rules = (
            "- 全球金融候选必须填写 finance_lens（市场/宏观/政策/公司/行业/资本事件）与 importance_score（1–100）。\n"
            "- importance_score 只表示事件重要性；按分数降序输出。候选池至少包含 2 条市场/宏观/政策、2 条公司/行业，其余不限。\n"
            if plan.get("finance_structure")
            else ""
        )
        output_min = int(plan.get("output_min", 6))
        target_total = int(plan.get("target_total", plan["limit"]))
        prompt = f"""今天是 {brief_date}（北京时间）。联网搜索并为“{plan['name']}”建立新闻候选池。

检索主题：{plan['focus']}
{query_block}来源优先级：{plan['priority']}
时效要求：优先过去 24 小时；过去 24–48 小时只在确有重要性时采用；更早内容只能是下列过去 7 天事件的明确后续。
过去 7 天事件：{compact_history}

输出 JSON Schema：
{json.dumps(CandidateBatch.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

要求：
- 目标输出 {target_total} 个、最多 {plan['limit']} 个按重要性排序的候选；若真实合格来源不足，可少至 {output_min} 个，不得用旧闻、低可信来源或未绑定链接补足。category 只能是：{categories}。
- 候选数量必须满足：{minimums}。
{finance_rules}- source_index 必须是本次搜索角标 [n] 中的整数 n；一个候选只能引用一个最直接的来源。
- source_url 必须逐字来自同一角标对应的 DashScope 搜索来源；published_at 必须以 YYYY-MM-DD 开头并保留来源显示的时间/时区，published_date 为同一日期。
- 找不到来源发布日期、无法验证链接或只是旧闻重复的内容不要输出。
- 社媒只采用平台公告或可靠媒体；没有量化证据时不得写热度数字。
- 只返回 JSON 对象。
"""
        search_results: list[dict[str, Any]] = []
        eligible_drafts: list[CandidateItem] = []
        source_urls: set[str] = set()
        quality_error = ""
        for quality_attempt in range(2):
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
            try:
                batch = CandidateBatch.model_validate(payload)
            except Exception as exc:
                quality_error = "candidate JSON failed schema validation"
                if quality_attempt == 0:
                    LOG.warning(
                        "%s %s; retrying once",
                        plan["name"],
                        quality_error,
                    )
                    continue
                raise ValueError(
                    f"{plan['name']} search quality failed: {quality_error}"
                ) from exc
            raw_search_results = search_result_records(response)
            search_results = [
                result
                for result in raw_search_results
                if source_host_allowed(result["url"], plan["sites"])
            ][: plan["limit"] + 4]
            LOG.info(
                "%s source filter: returned=%d trusted=%d quality_attempt=%d/2",
                plan["name"],
                len(raw_search_results),
                len(search_results),
                quality_attempt + 1,
            )
            untrusted_domains = sorted(
                {
                    urlsplit(result["url"]).netloc.lower()
                    for result in raw_search_results
                    if not source_host_allowed(result["url"], plan["sites"])
                }
            )
            if untrusted_domains:
                LOG.info(
                    "%s filtered source domains: %s",
                    plan["name"],
                    ", ".join(untrusted_domains),
                )
            invalid_categories = [
                item.category
                for item in batch.candidates
                if item.category not in plan["categories"]
            ]
            bound_candidates: list[CandidateItem] = []
            for item in batch.candidates:
                if item.category not in plan["categories"]:
                    continue
                if not resolve_candidate_source(item, search_results):
                    LOG.warning(
                        "Dropping candidate without a verified search source: %s",
                        item.headline,
                    )
                    continue
                matched_result = next(
                    (
                        result
                        for result in search_results
                        if normalize_url(result["url"])
                        == normalize_url(item.source_url)
                    ),
                    None,
                )
                if matched_result is None:
                    continue
                source_date = source_date_from_result(matched_result)
                if source_date:
                    item.published_date = source_date
                    item.published_at = source_date
                bound_candidates.append(item)
            eligible_drafts = filter_candidate_recency(
                bound_candidates, brief_day, history
            )
            eligible_drafts = deduplicate_ranked_candidates(eligible_drafts)
            source_urls = {
                normalize_url(result["url"]) for result in search_results
            }
            quality_errors: list[str] = []
            required_sources = sum(plan["minimum"].values())
            if len(source_urls) < required_sources:
                quality_errors.append(
                    f"only {len(source_urls)} trusted source URLs; need {required_sources}"
                )
            if invalid_categories:
                quality_errors.append("unexpected candidate categories")
            for category, minimum in plan["minimum"].items():
                eligible_count = sum(
                    item.category == category for item in eligible_drafts
                )
                if eligible_count < minimum:
                    quality_errors.append(
                        f"only {eligible_count} timely {category} candidates; need {minimum}"
                    )
            if not quality_errors:
                break
            quality_error = "; ".join(quality_errors)
            if quality_attempt == 0:
                LOG.warning(
                    "%s search quality insufficient (%s); retrying once",
                    plan["name"],
                    quality_error,
                )
        else:
            raise ValueError(f"{plan['name']} search quality failed: {quality_error}")
        plan_pools.setdefault(plan["prefix"], []).extend(eligible_drafts)
        all_source_urls.update(source_urls)

    return select_ranked_candidates(plan_pools), all_source_urls


def deduplicate_ranked_candidates(
    candidates: list[CandidateItem],
) -> list[CandidateItem]:
    unique: list[CandidateItem] = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    for item in candidates:
        normalized_url = normalize_url(item.source_url)
        normalized_title = normalize_text(item.headline)
        if normalized_url in seen_urls:
            continue
        if any(SequenceMatcher(None, normalized_title, old).ratio() >= 0.82 for old in seen_titles):
            continue
        unique.append(item)
        seen_urls.add(normalized_url)
        seen_titles.append(normalized_title)
    return unique


def select_finance_candidates(candidates: list[CandidateItem]) -> list[CandidateItem]:
    """Select up to five verified items; diversity only breaks quality ties."""
    unique = deduplicate_ranked_candidates(
        [item for item in candidates if item.category == "全球金融"]
    )
    if len(unique) < FINANCE_SEARCH_MIN_VALIDATED:
        raise ValueError(
            "Fewer than 3 unique global finance candidates remain"
        )

    confidence_rank = {"高": 2, "中": 1, "低": 0}

    def quality_key(item: CandidateItem) -> tuple[int, str, int]:
        return (
            confidence_rank[item.confidence],
            item.published_date,
            item.importance_score or 0,
        )

    remaining = list(enumerate(unique))
    selected: list[CandidateItem] = []
    seen_lenses: set[str] = set()
    while remaining and len(selected) < 5:
        best_quality = max(quality_key(item) for _index, item in remaining)
        tied = [pair for pair in remaining if quality_key(pair[1]) == best_quality]
        chosen = next(
            (pair for pair in tied if pair[1].finance_lens not in seen_lenses),
            tied[0],
        )
        remaining.remove(chosen)
        selected.append(chosen[1])
        if chosen[1].finance_lens:
            seen_lenses.add(chosen[1].finance_lens)
    return selected


def select_ranked_candidates(
    plan_pools: dict[str, list[CandidateItem]],
) -> list[CandidateItem]:
    """Apply deterministic structure to Qwen-ranked, source-verified pools."""
    selected = select_finance_candidates(plan_pools.get("G", []))
    seen_urls = {normalize_url(item.source_url) for item in selected}
    seen_titles = [normalize_text(item.headline) for item in selected]

    def add_ranked(pool: list[CandidateItem], target_total: int) -> None:
        for item in pool:
            if len(selected) >= target_total:
                return
            normalized_url = normalize_url(item.source_url)
            normalized_title = normalize_text(item.headline)
            if normalized_url in seen_urls:
                continue
            if any(
                SequenceMatcher(None, normalized_title, old).ratio() >= 0.82
                for old in seen_titles
            ):
                continue
            selected.append(item)
            seen_urls.add(normalized_url)
            seen_titles.append(normalized_title)

    before_ai = len(selected)
    add_ranked(plan_pools.get("A", []), before_ai + 5)
    if len(selected) - before_ai < 3:
        raise ValueError("Fewer than 3 unique AI candidates remain")

    before_semiconductors = len(selected)
    add_ranked(plan_pools.get("S", []), before_semiconductors + 3)
    if len(selected) - before_semiconductors < 2:
        raise ValueError("Fewer than 2 unique semiconductor candidates remain")

    before_social = len(selected)
    add_ranked(plan_pools.get("T", []), before_social + 5)
    return selected


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
    semiconductor_count = sum(
        item.category == "半导体重点" for item in candidates
    )
    finance_count = sum(item.category == "全球金融" for item in candidates)
    ai_count = sum(item.category == "AI行业" for item in candidates)
    social_count = sum(item.category == "社媒趋势" for item in candidates)
    semiconductor_limit_note = (
        "当天仅有 2 条半导体候选满足时效与来源校验，data_limitations 必须明确说明此限制。"
        if semiconductor_count == 2
        else ""
    )
    history_json = json.dumps(history["events"], ensure_ascii=False, separators=(",", ":"))
    prompt = f"""为 {brief_date}（北京时间）生成 Philo Daily Brief V3 的 JSON 数据。

已验证候选（只能从中选择，禁止添加候选之外的事实和链接）：
{candidate_json}

过去 7 天事件（用于新增/延续判断）：
{history_json}

输出 JSON Schema：
{json.dumps(DailyBrief.model_json_schema(), ensure_ascii=False, separators=(',', ':'))}

再次强调：全球金融 {finance_count} 条、AI 行业 {ai_count} 条、半导体 {semiconductor_count} 条、社媒 {social_count} 条；金融与 AI 的目标是各 5 条、最低 3 条，候选不足时使用实际条数，不得补写；社媒候选少于 3 条时不得补写，social_limit_notice 必须逐字填写“{SOCIAL_SHORTFALL_NOTICE}”。source_url、source_name、published_at 必须原样继承候选。{semiconductor_limit_note}只返回 JSON 对象。
"""
    candidate_urls = {normalize_url(item.source_url) for item in candidates}
    if not candidate_urls.issubset(search_urls):
        raise ValueError("Candidate source set is not a subset of DashScope search results")
    candidate_by_url = {
        normalize_url(item.source_url): item for item in candidates
    }
    synthesis_error = ""
    brief: DailyBrief | None = None
    for synthesis_attempt in range(2):
        correction = (
            "\n上一输出未通过代码验收："
            f"{synthesis_error}。仅修正结构与候选字段继承，不得添加新事实。"
            if synthesis_error
            else ""
        )
        payload, _response = call_dashscope_json(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt + correction},
            ],
            max_tokens=final_max_tokens,
            timeout_seconds=timeout_seconds,
            ledger=ledger,
            search_enabled=False,
        )
        try:
            candidate_brief = DailyBrief.model_validate(payload)
            expected_counts = {
                "global_finance": finance_count,
                "ai_industry": ai_count,
                "semiconductors": semiconductor_count,
                "social_trends": social_count,
            }
            for field_name, expected in expected_counts.items():
                actual = len(getattr(candidate_brief, field_name))
                if actual != expected:
                    raise ValueError(
                        f"正文 {field_name} 条目数为 {actual}，应为 {expected}"
                    )
            for _category, field_name in CATEGORY_FIELDS:
                for news in getattr(candidate_brief, field_name):
                    source = candidate_by_url.get(normalize_url(news.source_url))
                    if source is None:
                        raise ValueError("正文包含候选池之外的来源链接")
                    news.source_name = source.source_name
                    news.source_url = source.source_url
                    news.published_at = source.published_at
            classify_and_deduplicate(candidate_brief, history)
            validate_final_recency(candidate_brief, date.fromisoformat(brief_date))
            brief = candidate_brief
            break
        except Exception as exc:
            synthesis_error = f"{type(exc).__name__}: {str(exc)[:180]}"
            if synthesis_attempt == 0:
                LOG.warning(
                    "Final synthesis quality failed (%s); retrying once",
                    synthesis_error,
                )
    if brief is None:
        raise ValueError(
            f"Final synthesis failed after one quality retry: {synthesis_error}"
        )
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
    usage_month = brief_date[:7]
    ledger = UsageLedger()
    model = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
    workflow_run_id = os.environ.get("GITHUB_RUN_ID") or (
        "local-" + datetime.now(BEIJING).strftime("%Y%m%dT%H%M%S%z")
    )
    candidates: list[CandidateItem] = []
    live_run = args.input_json is None
    usage_history = empty_usage_history(usage_month)

    try:
        parsed_date = date.fromisoformat(brief_date)
        history = load_history(root / "data" / "previous_events.json", parsed_date)
        usage_history = load_usage_history(
            root / "data" / "usage" / f"{usage_month}.json",
            usage_month,
        )
        enforce_monthly_budget(usage_history, brief_date)
        if args.input_json:
            LOG.info("Loading local test fixture; DashScope API is bypassed")
            brief = DailyBrief.model_validate_json(
                args.input_json.read_text(encoding="utf-8")
            )
            response_urls = {
                normalize_url(str(item.source_url)) for _, item in iter_news(brief)
            }
        else:
            base_http_api_url = os.environ.get(
                "DASHSCOPE_BASE_HTTP_API_URL", ""
            ).strip()
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
            timeout_seconds = int(
                os.environ.get("DASHSCOPE_TIMEOUT_SECONDS", "120")
            )
            search_max_tokens = int(
                os.environ.get("DASHSCOPE_SEARCH_MAX_TOKENS", "2400")
            )
            final_max_tokens = int(
                os.environ.get("DASHSCOPE_FINAL_MAX_TOKENS", "6000")
            )
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
            raise ValueError(
                f"Brief date mismatch: requested {brief_date}, received {brief.date}"
            )
        lock_verified_source_fields(brief, candidates)
        classify_and_deduplicate(brief, history)
        validate_sources(brief, response_urls)
        validate_final_recency(brief, parsed_date)
        assert_brief_quality(
            brief.model_dump(mode="json"),
            brief_date=brief_date,
            verified_source_urls=response_urls,
        )
        updated_history = update_history(brief, history)
        updated_usage = update_usage_history(
            usage_history,
            brief_date,
            model,
            ledger,
            succeeded=True,
            workflow_run_id=workflow_run_id,
        )
        html = render_html(root, brief)

        atomic_write(output_dir / f"{brief_date}.html", html)
        atomic_write(
            output_dir / "brief.json",
            json.dumps(brief.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
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
    except Exception:
        if live_run:
            failed_usage = update_usage_history(
                usage_history,
                brief_date,
                model,
                ledger,
                succeeded=False,
                workflow_run_id=workflow_run_id,
            )
            atomic_write(
                output_dir / "usage.json",
                json.dumps(failed_usage, ensure_ascii=False, indent=2) + "\n",
            )
            failed_record = ledger.as_record(
                brief_date,
                model,
                succeeded=False,
                workflow_run_id=workflow_run_id,
            )
            LOG.error(
                "Failed generation usage saved: calls=%d search_calls=%d total_tokens=%d estimated_usd=%.4f",
                failed_record["successful_calls"],
                failed_record["search_calls"],
                failed_record["total_tokens"],
                failed_record["estimated_total_cost_usd"],
            )
        raise

    usage = ledger.as_record(
        brief_date,
        model,
        succeeded=True,
        workflow_run_id=workflow_run_id,
    )
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
