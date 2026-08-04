#!/usr/bin/env python3
"""Promote a validated candidate and atomically refresh homepage and archive."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


DATE_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.html$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, dest="brief_date")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--build-dir", type=Path)
    return parser.parse_args()


def atomic_write(path: Path, content: str) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    return temporary


def edition_dates(root: Path, candidate_date: str) -> list[str]:
    dates = {candidate_date}
    for path in root.glob("*.html"):
        match = DATE_FILE.match(path.name)
        if match:
            dates.add(match.group(1))
    return sorted(dates, reverse=True)


def recent_rows(dates: list[str], limit: int | None = None) -> str:
    selected = dates[:limit] if limit else dates
    return "\n".join(
        f'''        <li>
          <a href="{value}.html">
            <time datetime="{value}">{value[5:7]} / {value[8:10]}</time>
            <strong>{value} 行业热点日报</strong>
            <span class="arrow" aria-hidden="true">→</span>
          </a>
        </li>'''
        for value in selected
    )


def render_index(brief: dict, dates: list[str]) -> str:
    conclusions = "\n".join(
        f"        <li><strong>{html.escape(text)}</strong></li>"
        for text in brief["core_conclusions"]
    )
    brief_date = brief["date"]
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Philo Daily Brief：中文行业热点日报与投资决策线索。">
  <title>Philo Daily Brief</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="site-header shell">
    <nav class="nav" aria-label="主导航">
      <a class="brand" href="index.html">Philo Daily</a>
      <div class="nav-links"><a href="{brief_date}.html">今日日报</a><a href="archive.html">归档</a></div>
    </nav>
  </header>
  <main class="shell">
    <section aria-labelledby="page-title">
      <p class="eyebrow">V3 · Industry intelligence</p>
      <h1 id="page-title">Philo Daily Brief</h1>
      <p class="deck">{html.escape(brief["summary"])}</p>
      <div class="meta-row"><span>{brief_date} · 北京时间 07:15</span><span>每日自动更新</span></div>
      <div class="hero-rule" aria-hidden="true"></div>
    </section>
    <section class="section" aria-labelledby="core-title">
      <span class="section-label">Today in three</span>
      <h2 id="core-title">今日核心结论</h2>
      <ol class="conclusion-list">
{conclusions}
      </ol>
      <div class="button-row">
        <a class="button" href="{brief_date}.html">阅读今日完整日报 <span aria-hidden="true">→</span></a>
        <a class="button button--ghost" href="archive.html">历史归档</a>
      </div>
    </section>
    <section class="section" aria-labelledby="recent-title">
      <span class="section-label">Recent editions</span>
      <h2 id="recent-title">最近 7 期</h2>
      <ul class="recent-list">
{recent_rows(dates, 7)}
      </ul>
    </section>
  </main>
  <footer class="site-footer shell"><p>Philo Daily Brief · 事实、外部观点与结构化判断分层呈现。</p></footer>
</body>
</html>
'''


def render_archive(dates: list[str]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for value in dates:
        grouped[value[:7]].append(value)
    month_blocks = []
    for month in sorted(grouped, reverse=True):
        year, month_number = month.split("-")
        month_blocks.append(
            f'''    <section aria-labelledby="month-{month}">
      <h2 class="archive-year" id="month-{month}">{year} 年 {int(month_number)} 月</h2>
      <ul class="recent-list">
{recent_rows(grouped[month])}
      </ul>
    </section>'''
        )
    latest = dates[0]
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Philo Daily Brief 历史日报归档。">
  <title>历史归档 · Philo Daily Brief</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="site-header shell">
    <nav class="nav" aria-label="主导航">
      <a class="brand" href="index.html">Philo Daily</a>
      <div class="nav-links"><a href="{latest}.html">今日日报</a><a href="archive.html" aria-current="page">归档</a></div>
    </nav>
  </header>
  <main class="shell">
    <section aria-labelledby="archive-title">
      <p class="eyebrow">Archive</p>
      <h1 id="archive-title">历史归档</h1>
      <p class="deck">按月份浏览全部中文行业热点日报。</p>
      <div class="hero-rule" aria-hidden="true"></div>
    </section>
{chr(10).join(month_blocks)}
    <div class="button-row"><a class="button button--ghost" href="index.html">← 返回首页</a></div>
  </main>
  <footer class="site-footer shell"><p>Philo Daily Brief · Archive</p></footer>
</body>
</html>
'''


def main() -> int:
    args = parse_args()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    build_dir = (args.build_dir or root / ".build").resolve()
    date.fromisoformat(args.brief_date)

    candidate = build_dir / f"{args.brief_date}.html"
    brief_path = build_dir / "brief.json"
    events_path = build_dir / "previous_events.json"
    for required in (candidate, brief_path, events_path):
        if not required.exists() or required.stat().st_size == 0:
            raise FileNotFoundError(f"Missing generated artifact: {required}")

    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    if brief.get("date") != args.brief_date:
        raise ValueError("Generated brief date does not match requested date")
    dates = edition_dates(root, args.brief_date)

    staged_daily = root / f"{args.brief_date}.html.tmp"
    shutil.copyfile(candidate, staged_daily)
    staged_index = atomic_write(root / "index.html", render_index(brief, dates))
    staged_archive = atomic_write(root / "archive.html", render_archive(dates))
    staged_events = root / "data" / "previous_events.json.tmp"
    shutil.copyfile(events_path, staged_events)

    os.replace(staged_daily, root / f"{args.brief_date}.html")
    os.replace(staged_index, root / "index.html")
    os.replace(staged_archive, root / "archive.html")
    os.replace(staged_events, root / "data" / "previous_events.json")
    print(f"Promoted {args.brief_date} and refreshed index/archive at {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

