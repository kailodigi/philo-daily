import json
import unittest
from pathlib import Path

from scripts.generate_daily import (
    CandidateItem,
    SEARCH_PLANS,
    UsageLedger,
    empty_usage_history,
    resolve_candidate_source,
    select_finance_candidates,
    select_ranked_candidates,
    update_usage_history,
)
from scripts.quality_gate import SOCIAL_SHORTFALL_NOTICE, validate_brief_payload


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
    return CandidateItem(
        category=category,
        status_hint="新增",
        headline=f"{prefix}{topics[prefix][number - 1]}",
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
        queries = "\n".join(plan["queries"])
        for required in (
            "markets stocks bonds oil gold dollar",
            "earnings guidance investment acquisition partnership",
            "AI semiconductor chips memory packaging",
            "PBOC China stocks Hong Kong stocks policy",
        ):
            self.assertIn(required, queries)


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


if __name__ == "__main__":
    unittest.main()
