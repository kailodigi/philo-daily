# Philo Daily Brief

Philo Daily Brief 是一个由 GitHub Actions 定时生成和发布的中文行业热点日报。固定首页：

<https://kailodigi.github.io/philo-daily/>

## 自动化流程

- 每天北京时间 07:15（UTC 23:15）运行，也支持在 Actions 页面手动触发。
- Python 3.12 调用阿里百炼 DashScope 原生 SDK，默认使用 `qwen-plus`。
- 每日按金融宏观/市场、金融公司/监管、AI、半导体、社媒执行 5 次互补的强制联网候选采集。代码按搜索角标确定性绑定真实来源、按模型已给出的重要性顺序去重和配额选择，最后只用 1 次不联网调用生成 V3 正文。
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
├── tests/test_generate_daily.py
├── templates/daily_v3.html
├── data/
│   ├── previous_events.json
│   └── usage.json
├── requirements.txt
├── index.html
├── archive.html
├── YYYY-MM-DD.html
└── .nojekyll
```

## 必需设置

仓库 `Settings → Secrets and variables → Actions` 中需要存在名为 `DASHSCOPE_API_KEY` 的 Repository secret。密钥只由工作流注入进程环境，不得写入代码、日志、HTML 或提交记录。

生产工作流使用 DashScope SDK 的默认北京公共端点，适配同地域工作空间 Key。生成器也支持通过非敏感环境变量 `DASHSCOPE_BASE_HTTP_API_URL` 配置业务空间专属端点，并且只接受 HTTPS、域名以 `.aliyuncs.com` 结尾的 `/api/v1` 地址。

GitHub Pages 的 Build and deployment Source 需要设置为 **GitHub Actions**。

## 本地验收

安装依赖后，可以对现有站点执行：

```bash
python scripts/validate_html.py index.html archive.html --site-root .
```

生成过程还会检查：当天 HTML 非空且可按 UTF-8 读取、HTML 结构完整、金融和 AI 各有 5 条、每条资讯包含状态/事实/重要性/来源/发布时间/可信度、无明显占位符，以及全部内部链接有效。

## 内容边界

日报的五组搜索先使用多源检索获取候选，再由代码按官方来源、公司官网、监管机构及 Reuters、Bloomberg、FT 等可靠媒体域名白名单过滤；首页型链接会被拒绝，搜索元数据或 URL 中可验证的日期会锁定发布时间。系统不会虚构数字或社媒热度；没有可验证的小红书、抖音等平台量化数据时，会在页面中明确披露限制。重点跟踪方向是研究框架，只允许定性表述，不会冒充用户实际持仓或生成未经来源支持的数字。

Qwen 只处理候选排序、重要性判断、新增/延续判断、摘要与正文 JSON；它不执行命令、不修改工作流或权限、不读取 Secret，也不使用图片、embedding、代码解释器或智能体能力。联网阶段启用 DashScope 的强制搜索并核对返回来源，正文阶段关闭搜索，禁止模型记忆替代来源。

## 成本控制

工作流先运行不联网单元测试，通过后才允许调用 API。系统默认每天 6 次成功模型调用（5 次搜索、1 次正文生成），单组搜索若返回的可信来源不足只允许再尝试一次，正文结构不合格也只允许重试一次，并限制候选与正文输出长度。`data/usage.json` 记录每日成功和失败生成次数、API 尝试、成功调用、搜索调用、输入/输出 Token 及人民币和美元费用估算；失败运行只提交用量快照，工作流随后以非零状态退出，不会部署或覆盖正常首页。月度估算达到 4.75 美元时会在调用前停止，以保留 5 美元预算余量。估算采用仓库记录的保守单价与固定汇率，阿里云实际账单和汇率可能不同，应以控制台为准。

## 安全边界

工作流只使用 `ubuntu-latest` 和 GitHub 官方 Action，不使用 self-hosted runner 或 `pull_request_target`。生成器只读取仓库内的事件与用量 JSON，不访问用户电脑、本地知识库、EagleLite 或 Obsidian。
