# Agent 消息时间线

Codex、Antigravity 和普通 Gemini 通过 `agent_timeline.py` 写入同一种有序内容块。新增提供商只需将其输出转成这些内容块，并提供对应的用量、正文和结束原因。PocketBase 根据来源协议记录 `protocol`，RikkaHub 按内容块类型渲染。

## 存储与接口

| 位置 | 内容 |
| --- | --- |
| `aw_gemini_jobs` | 提交参数、所有者、执行身份、调度状态、会话关联 |
| `aw_gemini_messages` | 每个分段的正文 `reply`、状态、时间、用量、`structuredOutput` 和终态 `timeline` |
| `aw_agent_parts` | 流式内容块。按 `message + key` 更新，按 `sequence` 排序，按 `revision` 增量读取 |

执行器通过 `POST …/progress` 发送带 `executionId` 和递增批次号的 `{key, sequence, part}`。客户端用 `GET …/parts?after=N` 读取更新。消息完成时，`POST …/complete` 在一个事务中保存终态时间线和正文，并清理流式记录。取消或无终态上报的中断保留已收到的流式分片。

终态 `timeline` 由 `version: 1`、`parts`、`annotations`、`omittedParts` 组成。终态内容块额外带有 `sequence`，用量、执行身份与状态仍由消息和任务分别管理。

| 内容块 | 字段 |
| --- | --- |
| `text` | `text`，可带正文局部引用 `annotations` |
| `reasoning` | 提供商公开的 `reasoning` 和 `reasoningType: "summary_text"` |
| `server_tool` / `tool` | `toolCallId`、`toolName`、`input`、`output`，后两者是 JSON |
| `image` / `document` | `url`、`mimeType`，可带 `fileName` |
| `annotations` / `status` | 流式引用和执行状态 |

内容块带 `status`，可选 `providerType` 和 `truncated`。工具结果更新原调用卡片的位置。引用使用 `title` 和 `url`，上游给出区间时保留块内 `startIndex` 和 `endIndex`。RikkaHub 复用原有的来源列表。时间线最大 5 MiB，流式内容块约 60 KiB。超限信息通过 `truncated` 和 `omittedParts` 传递。

Codex 的终端、文件修改、搜索、MCP 和计划在适配层采集。终端日志支持流式更新，文件 diff 读取 App Server 的文件修改项。Gemini 和 Antigravity 的工具、公开摘要和正文按上游顺序转换。签名、私有推理和凭据字段不进入展示数据。格式参考 [Codex App Server](https://learn.chatgpt.com/docs/app-server)、[Gemini generateContent](https://ai.google.dev/api/generate-content) 和 [Interactions steps](https://aistudio.google.com/docs/interactions-breaking-changes-may-2026)。

## 部署

先等待已有任务结束，再暂停 Codex 和 Gemini 执行器，应用 AnyWorkflow-workers 的 `1795350000_agent_timeline.js` 和对应 hooks，再启动新版执行器并发布 RikkaHub。迁移会删掉旧结构化输出和流式记录，保留任务、会话正文、用量与各自的所有者。部署检查包括两个队列的健康状态、同一工具块的实时更新、完成后重新读取时间线和 Android 构建。
