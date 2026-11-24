# Claude Code 执行器

先发布同一功能分支的 AnyWorkflow-workers PocketBase hooks。Android 的 Claude Code 和 Claude Code GPT 分组提交到同一个 `claude-code` 队列，服务端根据所选模型使用对应的 ModelFlare 凭据。图片先经资料库上传，再以内联图片传给 Claude Code CLI。续聊使用服务端持久化的 CLI 会话。

将本分支安装至 `/opt/anyworkflow-agent-claude-code`，在该路径的 `.venv` 中安装 `requirements-server.txt`。将支持 `--restricted` 和 `--effort` 的 Claude Code CLI 安装到 `/usr/local/bin/claude`，创建 `aw-claude-code-agent` 系统账户，并安装相邻的 systemd unit。服务账号的持久目录由 systemd 创建。

将 `GEMINI_PB_URL`、PocketBase 执行器账号以及两组 `MODELFLARE_CLAUDE_KEY`、`MODELFLARE_GPT_KEY` 配置到 `/etc/anyworkflow/agent-claude-code.env`，文件权限为 `0600`。安装 unit 后执行 `systemctl daemon-reload` 和 `systemctl enable --now anyworkflow-agent-claude-code`。检查 `http://127.0.0.1:18096/health` 的 `pb_executor_queue` 和 `pb_polling_ready`，再用两个模型分组分别测试文本、图片和续聊，然后发布 Android 应用。

执行器使用 Claude Code 的受限模式，文件工具作用于会话工作区。CLI 的内置系统提示保留，助手配置里的系统提示作为追加指令。
