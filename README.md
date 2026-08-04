# Philo Daily Brief

Philo Daily Brief 是一个由 GitHub Actions 定时生成和发布的中文行业热点日报。固定首页：

<https://kailodigi.github.io/philo-daily/>

## 自动化流程

- 每天北京时间 07:15（UTC 23:15）运行，也支持在 Actions 页面手动触发。
- Python 3.12 调用 OpenAI Responses API，并使用内置 web search 获取当天资讯。
- 生成文件先保存在 `.build/` 候选区；只有 HTML 验收通过，才会更新当天日报、首页、归档和过去 7 天事件记录。
- 同一次工作流直接构建并部署 GitHub Pages，不依赖自动提交再次触发工作流。
- API 请求失败只重试一次；失败时不覆盖上一期正常首页，也不会部署失败页面。

## 仓库结构

```text
.
├── .github/workflows/daily-brief.yml
├── scripts/
│   ├── generate_daily.py
│   ├── update_archive.py
│   └── validate_html.py
├── templates/daily_v3.html
├── data/previous_events.json
├── requirements.txt
├── index.html
├── archive.html
├── YYYY-MM-DD.html
└── .nojekyll
```

## 必需设置

仓库 `Settings → Secrets and variables → Actions` 中需要存在名为 `OPENAI_API_KEY` 的 Repository secret。密钥只由工作流注入进程环境，不得写入代码、日志或提交记录。

GitHub Pages 的 Build and deployment Source 需要设置为 **GitHub Actions**。

## 本地验收

安装依赖后，可以对现有站点执行：

```bash
python scripts/validate_html.py index.html archive.html --site-root .
```

生成过程还会检查：当天 HTML 非空且可按 UTF-8 读取、HTML 结构完整、金融和 AI 各有 5 条、每条资讯包含状态/事实/重要性/来源/发布时间/可信度、无明显占位符，以及全部内部链接有效。

## 内容边界

日报优先使用官方来源、公司官网、监管机构和 Reuters 等可靠媒体。系统不会虚构数字或社媒热度；没有可验证的小红书、抖音等平台量化数据时，会在页面中明确披露限制。重点跟踪方向是研究框架，不会冒充用户实际持仓。
