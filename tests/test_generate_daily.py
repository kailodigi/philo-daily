import unittest

from scripts.generate_daily import (
    CandidateItem,
    UsageLedger,
    empty_usage_history,
    resolve_candidate_source,
    select_ranked_candidates,
    update_usage_history,
)


def candidate(category: str, prefix: str, number: int) -> CandidateItem:
    topics = {
        "M": ("央行调整利率路径", "原油供应预期变化", "主要汇率波动扩大", "国债收益率重新定价"),
        "F": ("大型企业发布资本开支", "监管机构更新交易规则", "跨境并购出现新进展"),
        "A": ("基础模型增强企业服务", "开源社区发布模型工具", "云厂商扩展算力平台", "开发平台升级智能功能", "研究机构公布安全方法"),
        "S": ("先进封装产线公布扩产", "设备公司披露订单变化", "存储厂商更新产品路线"),
        "T": ("平台更新内容治理规则", "创作者工具增加透明说明"),
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
    )


class CandidateSelectionTests(unittest.TestCase):
    def test_deterministic_quotas(self) -> None:
        pools = {
            "M": [candidate("全球金融", "M", i) for i in range(1, 5)],
            "F": [candidate("全球金融", "F", i) for i in range(1, 4)],
            "A": [candidate("AI行业", "A", i) for i in range(1, 6)],
            "S": [candidate("半导体重点", "S", i) for i in range(1, 4)],
            "T": [candidate("社媒趋势", "T", i) for i in range(1, 3)],
        }

        selected = select_ranked_candidates(pools)
        counts = {
            category: sum(item.category == category for item in selected)
            for category in ("全球金融", "AI行业", "半导体重点", "社媒趋势")
        }

        self.assertEqual(
            counts,
            {"全球金融": 5, "AI行业": 5, "半导体重点": 3, "社媒趋势": 2},
        )
        self.assertEqual(len({item.source_url for item in selected}), len(selected))

    def test_missing_required_pool_fails_closed(self) -> None:
        pools = {
            "M": [candidate("全球金融", "M", i) for i in range(1, 4)],
            "F": [candidate("全球金融", "F", i) for i in range(1, 3)],
            "A": [candidate("AI行业", "A", i) for i in range(1, 6)],
            "S": [candidate("半导体重点", "S", i) for i in range(1, 3)],
            "T": [candidate("社媒趋势", "T", 1)],
        }
        with self.assertRaisesRegex(ValueError, "social"):
            select_ranked_candidates(pools)

    def test_source_index_mismatch_uses_unique_trusted_title(self) -> None:
        item = candidate("全球金融", "M", 1)
        item.source_index = 99
        item.source_name = "Unknown source"
        item.source_url = "https://untrusted.example/story"
        results = [
            {
                "index": 1,
                "title": "央行调整利率路径",
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


class UsageLedgerTests(unittest.TestCase):
    def test_failed_generation_is_counted(self) -> None:
        updated = update_usage_history(
            empty_usage_history(),
            "2026-08-05",
            "qwen-plus",
            UsageLedger(),
            succeeded=False,
        )
        record = updated["days"][0]
        self.assertEqual(record["generation_count"], 1)
        self.assertEqual(record["successful_generations"], 0)
        self.assertEqual(record["failed_generations"], 1)


if __name__ == "__main__":
    unittest.main()
