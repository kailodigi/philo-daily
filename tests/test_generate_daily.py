import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from scripts.generate_daily import (
    CandidateItem,
    DailyBrief,
    SEARCH_PLANS,
    UsageLedger,
    collect_finance_candidates,
    empty_usage_history,
    filter_finance_recency_with_reasons,
    finance_search_preflight_errors,
    finance_source_result_allowed,
    render_html,
    resolve_candidate_source,
    select_finance_candidates,
    select_ranked_candidates,
    update_usage_history,
)
from scripts.quality_gate import SOCIAL_SHORTFALL_NOTICE, validate_brief_payload
from scripts.validate_html import validate as validate_html


def candidate(
    category: str,
    prefix: str,
    number: int,
    *,
    finance_lens: str | None = None,
    score: int | None = None,
) -> CandidateItem:
    topics = {
        "G": ("全球股票市场重新定价", "央行更新利率政策路径", "大型企业发布盈利指引", "关键行业披露投资计划", "跨境资本事件出现进展", "监管机构更新市场政策"),
        "A": ("基础模型增强企业服务", "开源社区发布模型工具", "云厂商扩展算力平台", "开发平台升级智能功能", "研究机构公布安全方法"),
        "S": ("先进封装产线公布扩产", "设备公司披露订单变化", "存储厂商更新产品路线"),
        "T": ("平台更新内容治理规则", "创作者工具增加透明说明", "平台调整推荐透明度", "社媒发布安全治理报告", "创作者后台增加来源标签"),
    }
    topic_list = topics[prefix]
    topic = topic_list[(number - 1) % len(topic_list)]
    suffix = f"第{number}项" if number > len(topic_list) else ""
    return CandidateItem(
        category=category,
        status_hint="新增",
        headline=f"{prefix}{topic}{suffix}",
        what_happened="这是经过真实搜索来源绑定的候选事件事实摘要内容。",
        why_important="这项变化会影响行业判断并需要继续观察后续进展。",
        source_name=f"{prefix} Source",
        source_index=number,
        source_url=f"https://example.com/{prefix.lower()}/{number}",
        published_at="2026-08-05",
        published_date="2026-08-05",
        confidence="高",
        continuation_of=None,
        finance_lens=finance_lens,
        importance_score=score,
    )


def finance_pool() -> list[CandidateItem]:
    return [
        candidate("全球金融", "G", 1, finance_lens="市场", score=90),
        candidate("全球金融", "G", 2, finance_lens="宏观", score=80),
        candidate("全球金融", "G", 3, finance_lens="公司", score=95),
        candidate("全球金融", "G", 4, finance_lens="行业", score=85),
        candidate("全球金融", "G", 5, finance_lens="资本事件", score=99),
        candidate("全球金融", "G", 6, finance_lens="政策", score=70),
    ]


FINANCE_MOCK_HEADLINES = (
    "美股收盘呈现板块分化", "欧洲股市受利率预期影响", "亚洲市场跟随汇率波动",
    "全球债券收益率出现调整", "黄金与美元走势重新定价", "原油市场评估供应变化",
    "大型银行公布季度盈利", "消费企业更新全年指引", "云计算公司宣布资本投资",
    "工业集团推进跨境收购", "科技企业签署战略合作", "制药公司调整研发预算",
    "芯片设计企业发布新品", "人工智能芯片需求变化", "晶圆代工厂更新扩产计划",
    "存储芯片市场价格调整", "先进封装产业增加投资", "半导体设备公司披露订单",
    "中国股票市场成交改善", "香港股市科技板块回升", "央行公布最新政策操作",
    "中国科技公司发布业绩", "本土半导体行业推进融资", "监管机构更新资本市场规则",
    "跨境支付企业调整战略", "保险行业公布经营数据", "航运企业上调运力计划",
    "能源公司推进资产出售", "汽车行业更新交付预期", "交易所公布制度调整",
)


def finance_round_response(
    round_number: int,
    lenses: list[str],
) -> tuple[dict, list[dict]]:
    payload_candidates = []
    search_results = []
    offset = (round_number - 1) * 10
    for local_number, lens in enumerate(lenses, start=1):
        headline = FINANCE_MOCK_HEADLINES[offset + local_number - 1]
        url = (
            "https://www.reuters.com/markets/"
            f"round-{round_number}-item-{local_number}"
        )
        payload_candidates.append(
            {
                "category": "全球金融",
                "status_hint": "新增",
                "headline": headline,
                "what_happened": "这是由本轮真实搜索结果绑定的候选事件事实摘要内容。",
                "why_important": "这项变化会影响市场或行业判断并需要跟踪后续发展。",
                "source_name": "Reuters",
                "source_index": local_number,
                "source_url": url,
                "published_at": "2026-08-06",
                "published_date": "2026-08-06",
                "confidence": "高",
                "continuation_of": None,
                "finance_lens": lens,
                "importance_score": 100 - offset - local_number,
            }
        )
        search_results.append(
            {
                "index": local_number,
                "title": headline,
                "site_name": "Reuters",
                "url": url,
                "published_at": "2026-08-06",
            }
        )
    return {"candidates": payload_candidates}, search_results


class CandidateSelectionTests(unittest.TestCase):
    def test_deterministic_quotas(self) -> None:
        pools = {
            "G": finance_pool(),
            "A": [candidate("AI行业", "A", i) for i in range(1, 6)],
            "S": [candidate("半导体重点", "S", i) for i in range(1, 4)],
            "T": [candidate("社媒趋势", "T", i) for i in range(1, 6)],
        }

        selected = select_ranked_candidates(pools)
        counts = {
            category: sum(item.category == category for item in selected)
            for category in ("全球金融", "AI行业", "半导体重点", "社媒趋势")
        }

        self.assertEqual(
            counts,
            {"全球金融": 5, "AI行业": 5, "半导体重点": 3, "社媒趋势": 5},
        )
        self.assertEqual(len({item.source_url for item in selected}), len(selected))

    def test_social_shortfall_is_not_padded(self) -> None:
        pools = {
            "G": finance_pool(),
            "A": [candidate("AI行业", "A", i) for i in range(1, 6)],
            "S": [candidate("半导体重点", "S", i) for i in range(1, 3)],
            "T": [candidate("社媒趋势", "T", 1)],
        }
        selected = select_ranked_candidates(pools)
        social = [item for item in selected if item.category == "社媒趋势"]
        self.assertEqual(len(social), 1)

    def test_finance_top_five_respects_two_plus_two_plus_one(self) -> None:
        selected = select_finance_candidates(finance_pool())
        self.assertEqual([item.importance_score for item in selected], [99, 95, 90, 85, 80])
        self.assertGreaterEqual(
            sum(item.finance_lens in {"市场", "宏观", "政策"} for item in selected),
            2,
        )
        self.assertGreaterEqual(
            sum(item.finance_lens in {"公司", "行业"} for item in selected),
            2,
        )

    def test_finance_structure_fails_closed_without_two_company_items(self) -> None:
        pool = finance_pool()
        pool[3].finance_lens = "资本事件"
        with self.assertRaisesRegex(ValueError, "company/industry"):
            select_finance_candidates(pool)

    def test_source_index_mismatch_uses_unique_trusted_title(self) -> None:
        item = candidate(
            "全球金融", "G", 2, finance_lens="宏观", score=90
        )
        item.source_index = 99
        item.source_name = "Unknown source"
        item.source_url = "https://untrusted.example/story"
        results = [
            {
                "index": 1,
                "title": "央行更新利率政策路径",
                "site_name": "Reuters",
                "url": "https://www.reuters.com/markets/rates/story-1",
            },
            {
                "index": 2,
                "title": "国际原油供应出现变化",
                "site_name": "Financial Times",
                "url": "https://www.ft.com/content/story-2",
            },
        ]

        self.assertTrue(resolve_candidate_source(item, results))
        self.assertEqual(
            item.source_url,
            "https://www.reuters.com/markets/rates/story-1",
        )
        self.assertEqual(item.source_name, "Reuters")


class FailedRunRegressionTests(unittest.TestCase):
    def test_run_31002863347_old_quota_fails_but_unified_pool_passes(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "run_31002863347.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        observed = fixture["observed_failure"]
        self.assertEqual(fixture["run_id"], 31002863347)
        self.assertLess(
            observed["macro_attempt_1_timely"], observed["legacy_macro_minimum"]
        )
        self.assertLess(
            observed["company_attempt_1_timely"], observed["legacy_company_minimum"]
        )

        pool = [
            candidate(
                "全球金融",
                "G",
                item["number"],
                finance_lens=item["finance_lens"],
                score=item["importance_score"],
            )
            for item in fixture["unified_source_bound_mock"]
        ]
        selected = select_finance_candidates(pool)
        verified_urls = {item.source_url for item in pool}
        self.assertEqual(len(selected), 5)
        self.assertTrue({item.source_url for item in selected}.issubset(verified_urls))

    def test_unified_finance_plan_contains_required_queries(self) -> None:
        finance_plans = [
            plan for plan in SEARCH_PLANS if "全球金融" in plan["categories"]
        ]
        self.assertEqual(len(finance_plans), 1)
        plan = finance_plans[0]
        self.assertEqual(plan["prefix"], "G")
        self.assertEqual(plan["minimum"], {"全球金融": 5})
        self.assertTrue(plan["finance_structure"])
        self.assertEqual(plan["preflight_min_validated"], 7)
        self.assertEqual(len(plan["search_rounds"]), 3)
        self.assertEqual(
            [item["query_group"] for item in plan["search_rounds"]],
            [
                "全球市场/宏观/中国政策",
                "公司/行业/资本事件",
                "AI/半导体/中国科技/港股",
            ],
        )
        queries = [
            query
            for search_round in plan["search_rounds"]
            for query in search_round["queries"]
        ]
        for required in (
                "global markets stocks bonds currencies today Reuters",
                "US stocks close today Reuters",
                "European stocks today Reuters",
                "Asia markets today Reuters",
                "oil gold dollar bond yields today Reuters",
                "company earnings today Reuters",
                "company guidance today Reuters",
                "company investment today Reuters",
                "company acquisition today Reuters",
                "company partnership today Reuters",
                "semiconductor company news today Reuters",
                "AI chip company news today Reuters",
                "NVIDIA AMD TSMC news today",
                "memory chip market today",
                "advanced packaging semiconductor today",
                "China stocks today Reuters",
                "Hong Kong stocks today Reuters",
                "PBOC policy today Reuters",
                "China technology stocks today Reuters",
                "China semiconductor today Reuters",
        ):
            self.assertIn(required, queries)
        self.assertIn("gold oil dollar today Reuters", queries)
        self.assertIn("AI chip company today Reuters", queries)

    def test_run_31075828732_preflight_fails_old_pool_and_passes_expanded_mock(
        self,
    ) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "run_31075828732.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["run_id"], 31075828732)

        old = fixture["observed_failure"]
        old_pool = [
            candidate(
                "全球金融",
                "G",
                item["number"],
                finance_lens=item["finance_lens"],
                score=item["importance_score"],
            )
            for item in old["validated_mock"]
        ]
        old_errors = finance_search_preflight_errors(
            candidate_count=old["candidate_count"],
            validated_candidates=old_pool,
        )
        self.assertTrue(any("validated_count=3" in error for error in old_errors))
        self.assertTrue(any("company/industry" in error for error in old_errors))

        expanded = fixture["expanded_search_mock"]
        expanded_pool = [
            candidate(
                "全球金融",
                "G",
                item["number"],
                finance_lens=item["finance_lens"],
                score=item["importance_score"],
            )
            for item in expanded["validated_mock"]
        ]
        self.assertEqual(
            finance_search_preflight_errors(
                candidate_count=expanded["candidate_count"],
                validated_candidates=expanded_pool,
            ),
            [],
        )
        self.assertEqual(len(select_finance_candidates(expanded_pool)), 5)

    def collect_rounds(self, responses):
        finance_plan = next(
            plan for plan in SEARCH_PLANS if plan.get("finance_structure")
        )
        with patch(
            "scripts.generate_daily.search_result_records",
            side_effect=lambda value: value,
        ), patch("scripts.generate_daily.call_dashscope_json") as mock_call:
            mock_call.side_effect = responses
            with self.assertLogs("philo-daily", level="INFO") as captured:
                candidates, source_urls = collect_finance_candidates(
                    plan=finance_plan,
                    brief_date="2026-08-06",
                    compact_history="[]",
                    history={"events": []},
                    api_key="offline-placeholder",
                    model="qwen-plus",
                    timeout_seconds=1,
                    search_max_tokens=2400,
                    ledger=UsageLedger(),
                )
            return candidates, source_urls, mock_call.call_count, "\n".join(captured.output)

    def test_first_round_reaches_seven_and_stops(self) -> None:
        first = finance_round_response(
            1, ["市场", "宏观", "政策", "公司", "行业", "资本事件", "市场"]
        )
        candidates, _urls, calls, logs = self.collect_rounds([first])
        self.assertEqual(len(candidates), 7)
        self.assertEqual(calls, 1)
        for field in (
            "round_number=1",
            "query_group=全球市场/宏观/中国政策",
            "raw_candidate_count=",
            "source_bound_count=",
            "timely_count=",
            "deduplicated_count=",
            "market_macro_policy_count=",
            "company_industry_count=",
            "filtered_reason_summary=",
        ):
            self.assertIn(field, logs)
        self.assertIn("finance_search_stop", logs)

    def test_second_round_fills_shortfall_and_stops(self) -> None:
        responses = [
            finance_round_response(1, ["市场", "宏观", "公司", "资本事件"]),
            finance_round_response(2, ["行业", "公司", "政策", "市场"]),
        ]
        candidates, _urls, calls, _logs = self.collect_rounds(responses)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(calls, 2)

    def test_company_shortfall_after_round_two_enters_round_three(self) -> None:
        responses = [
            finance_round_response(1, ["市场", "宏观", "政策", "资本事件"]),
            finance_round_response(2, ["市场", "政策", "资本事件", "公司"]),
            finance_round_response(3, ["行业", "公司", "资本事件", "市场"]),
        ]
        candidates, _urls, calls, logs = self.collect_rounds(responses)
        self.assertGreaterEqual(len(select_finance_candidates(candidates)), 5)
        self.assertEqual(calls, 3)
        self.assertIn("round_number=3", logs)

    def test_three_rounds_can_complete_finance_structure(self) -> None:
        responses = [
            finance_round_response(1, ["市场", "宏观", "政策", "资本事件"]),
            finance_round_response(2, ["市场", "政策", "资本事件", "公司"]),
            finance_round_response(3, ["公司", "行业", "资本事件", "市场"]),
        ]
        candidates, _urls, _calls, _logs = self.collect_rounds(responses)
        selected = select_finance_candidates(candidates)
        self.assertEqual(len(selected), 5)
        self.assertGreaterEqual(
            sum(item.finance_lens in {"市场", "宏观", "政策"} for item in selected), 2
        )
        self.assertGreaterEqual(
            sum(item.finance_lens in {"公司", "行业"} for item in selected), 2
        )

    def test_three_rounds_still_structurally_short_fails(self) -> None:
        responses = [
            finance_round_response(1, ["市场", "宏观", "政策", "资本事件"]),
            finance_round_response(2, ["市场", "宏观", "政策", "资本事件"]),
            finance_round_response(3, ["市场", "宏观", "政策", "资本事件"]),
        ]
        finance_plan = next(
            plan for plan in SEARCH_PLANS if plan.get("finance_structure")
        )
        with patch(
            "scripts.generate_daily.search_result_records",
            side_effect=lambda value: value,
        ), patch("scripts.generate_daily.call_dashscope_json") as mock_call:
            mock_call.side_effect = responses
            with self.assertRaisesRegex(ValueError, "company/industry"):
                collect_finance_candidates(
                    plan=finance_plan,
                    brief_date="2026-08-06",
                    compact_history="[]",
                    history={"events": []},
                    api_key="offline-placeholder",
                    model="qwen-plus",
                    timeout_seconds=1,
                    search_max_tokens=2400,
                    ledger=UsageLedger(),
                )
        self.assertEqual(mock_call.call_count, 3)

    def test_seven_validated_candidates_pass_preflight(self) -> None:
        pool = finance_pool() + [
            candidate("全球金融", "G", 7, finance_lens="市场", score=60)
        ]
        self.assertEqual(
            finance_search_preflight_errors(
                candidate_count=7, validated_candidates=pool
            ),
            [],
        )

    def test_six_validated_candidates_fail_preflight(self) -> None:
        errors = finance_search_preflight_errors(
            candidate_count=6, validated_candidates=finance_pool()
        )
        self.assertTrue(any("validated_count=6" in error for error in errors))

    def test_old_unknown_and_unbound_candidates_are_filtered(self) -> None:
        response = finance_round_response(
            1,
            [
                "市场", "宏观", "政策", "公司", "行业", "资本事件", "市场",
                "公司", "行业", "资本事件",
            ],
        )
        payload, results = response
        results[7]["published_at"] = "2026-07-20"
        results[8]["site_name"] = "Unknown Source"
        results.pop(9)
        results[0]["title"] = "无法匹配的同名搜索结果"
        results[1]["title"] = "无法匹配的同名搜索结果"
        payload["candidates"][9]["headline"] = "无法匹配的同名搜索结果"
        payload["candidates"][9]["source_name"] = "No Matching Publisher"
        payload["candidates"][9]["source_index"] = 99
        payload["candidates"][9]["source_url"] = "https://untrusted.example/absent"

        candidates, _urls, calls, logs = self.collect_rounds([(payload, results)])
        self.assertEqual(len(candidates), 7)
        self.assertEqual(calls, 1)
        self.assertIn("outside_7_day_window=1", logs)
        self.assertIn("unknown_source=1", logs)
        self.assertIn("unbound_source_url=1", logs)

    def test_bankofchina_official_announcement_is_allowed(self) -> None:
        finance_plan = next(
            plan for plan in SEARCH_PLANS if plan.get("finance_structure")
        )
        allowed, reason = finance_source_result_allowed(
            {
                "url": "https://www.bankofchina.com/aboutboc/bi1/202608/t20260806.html",
                "title": "中国银行关于发布官方市场数据的公告",
                "site_name": "中国银行",
                "published_at": "2026-08-06",
            },
            finance_plan["sites"],
            date(2026, 8, 6),
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_bankofchina_old_or_irrelevant_pages_are_filtered(self) -> None:
        finance_plan = next(
            plan for plan in SEARCH_PLANS if plan.get("finance_structure")
        )
        old_allowed, old_reason = finance_source_result_allowed(
            {
                "url": "https://www.bankofchina.com/research/202501/report.html",
                "title": "中国银行官方宏观报告",
                "published_at": "2025-01-10",
            },
            finance_plan["sites"],
            date(2026, 8, 6),
        )
        marketing_allowed, marketing_reason = finance_source_result_allowed(
            {
                "url": "https://www.bankofchina.com/card/promotion.html",
                "title": "信用卡优惠促销活动",
                "published_at": "2026-08-06",
            },
            finance_plan["sites"],
            date(2026, 8, 6),
        )
        self.assertFalse(old_allowed)
        self.assertEqual(old_reason, "bankofchina_stale_or_future")
        self.assertFalse(marketing_allowed)
        self.assertEqual(marketing_reason, "bankofchina_marketing_or_product")

    def test_stop_condition_prevents_extra_query_rounds(self) -> None:
        responses = [
            finance_round_response(
                1, ["市场", "宏观", "政策", "公司", "行业", "资本事件", "市场"]
            ),
            AssertionError("round two must not execute"),
        ]
        _candidates, _urls, calls, _logs = self.collect_rounds(responses)
        self.assertEqual(calls, 1)


class UsageLedgerTests(unittest.TestCase):
    def test_failed_generation_is_counted(self) -> None:
        updated = update_usage_history(
            empty_usage_history("2026-08"),
            "2026-08-05",
            "qwen-plus",
            UsageLedger(),
            succeeded=False,
            workflow_run_id="30995037103",
        )
        record = updated["runs"][0]
        self.assertEqual(record["workflow_run_id"], "30995037103")
        self.assertEqual(record["model"], "qwen-plus")
        self.assertEqual(record["status"], "failure")
        self.assertEqual(record["input_tokens"], 0)
        self.assertEqual(record["output_tokens"], 0)
        self.assertIn("estimated_total_cost_usd", record)


def quality_item(number: int, *, status: str = "新增", published: str = "2026-08-05") -> dict:
    return {
        "status": status,
        "headline": f"真实来源行业事件标题{number}",
        "what_happened": "该事件已由公开搜索候选绑定并形成可核验的事实摘要。",
        "why_important": "该事件可能改变行业预期，因此需要持续跟踪后续进展。",
        "source_name": "Reuters",
        "source_url": f"https://www.reuters.com/world/story-{number}",
        "published_at": published,
        "confidence": "高",
        "continuation_of": None,
    }


def quality_brief() -> dict:
    return {
        "date": "2026-08-05",
        "global_finance": [quality_item(i) for i in range(1, 6)],
        "ai_industry": [quality_item(i) for i in range(6, 11)],
        "semiconductors": [quality_item(i) for i in range(11, 13)],
        "social_trends": [quality_item(i) for i in range(13, 15)],
        "social_limit_notice": SOCIAL_SHORTFALL_NOTICE,
    }


def complete_daily_brief() -> dict:
    brief = quality_brief()
    brief.update(
        {
            "title": "Philo Daily Brief",
            "summary": "本期聚焦全球市场、人工智能与半导体产业的可靠公开信号，并区分事实、影响和后续观察方向。",
            "core_conclusions": [
                "全球市场正在重新评估利率路径与企业盈利之间的平衡关系。",
                "人工智能产业继续围绕模型能力、基础设施投入和企业采用展开。",
                "半导体供需与先进封装投资仍是未来几个季度的重要观察线索。",
            ],
            "signal_board": [
                {"label": "市场", "value": "风险偏好分化", "tone": "中性"},
                {"label": "政策", "value": "等待新增信号", "tone": "中性"},
                {"label": "AI", "value": "产业投入延续", "tone": "积极"},
                {"label": "芯片", "value": "供需仍需验证", "tone": "警惕"},
            ],
            "philo_insight": "当天可靠信息显示，市场定价、人工智能资本投入与半导体供给变化正在相互影响。短期判断应优先区分正式公告和媒体报道，避免把单一事件外推为长期趋势，并持续核对企业指引、监管文件和后续经营数据。",
            "tomorrow_watch": [
                {"timeframe": "未来一天", "item": "关注主要市场收盘后的风险偏好变化"},
                {"timeframe": "未来一天", "item": "关注央行和监管机构是否发布正式文件"},
                {"timeframe": "未来一天", "item": "关注人工智能企业的产品与投资公告"},
                {"timeframe": "未来一天", "item": "关注半导体企业对需求和供给的最新表述"},
            ],
            "tracking_impacts": [
                {"direction": "全球市场", "impact": "风险偏好变化可能影响成长资产估值", "watch": "观察利率与盈利预期是否同步变化"},
                {"direction": "人工智能", "impact": "基础设施投入延续有助于产业需求", "watch": "观察企业采用是否转化为持续收入"},
                {"direction": "半导体", "impact": "供需变化可能影响产业链议价能力", "watch": "观察公司指引与库存变化是否一致"},
            ],
            "data_limitations": [
                "本页只使用离线 mock 数据验证结构、来源绑定和页面完整性，不代表真实新闻。"
            ],
        }
    )
    return brief


class QualityGateTests(unittest.TestCase):
    def test_valid_mock_passes_with_social_shortfall_notice(self) -> None:
        brief = quality_brief()
        verified = {
            item["source_url"]
            for field in ("global_finance", "ai_industry", "semiconductors", "social_trends")
            for item in brief[field]
        }
        self.assertEqual(
            validate_brief_payload(
                brief,
                brief_date="2026-08-05",
                verified_source_urls=verified,
            ),
            [],
        )

    def test_old_new_item_unknown_source_and_unverified_url_fail(self) -> None:
        brief = quality_brief()
        brief["global_finance"][0]["published_at"] = "2026-08-03"
        brief["global_finance"][1]["source_name"] = "未知来源"
        verified = {
            item["source_url"]
            for field in ("global_finance", "ai_industry", "semiconductors", "social_trends")
            for item in brief[field]
        }
        verified.remove(brief["ai_industry"][0]["source_url"])
        errors = validate_brief_payload(
            brief,
            brief_date="2026-08-05",
            verified_source_urls=verified,
        )
        joined = "\n".join(errors)
        self.assertIn("48h", joined)
        self.assertIn("unknown source", joined)
        self.assertIn("absent from verified search candidates", joined)

    def test_social_below_three_without_notice_fails(self) -> None:
        brief = quality_brief()
        brief["social_limit_notice"] = None
        errors = validate_brief_payload(brief, brief_date="2026-08-05")
        self.assertTrue(any(SOCIAL_SHORTFALL_NOTICE in error for error in errors))

    def test_offline_html_generation_and_quality_gate_pass(self) -> None:
        payload = complete_daily_brief()
        verified = {
            item["source_url"]
            for field in ("global_finance", "ai_industry", "semiconductors", "social_trends")
            for item in payload[field]
        }
        self.assertEqual(
            validate_brief_payload(
                payload,
                brief_date="2026-08-05",
                verified_source_urls=verified,
            ),
            [],
        )
        root = Path(__file__).resolve().parents[1]
        html = render_html(root, DailyBrief.model_validate(payload))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "2026-08-05.html"
            output.write_text(html, encoding="utf-8")
            self.assertEqual(
                validate_html(output, root, "2026-08-05.html"),
                [],
            )


if __name__ == "__main__":
    unittest.main()
