# Command Code 执行器

Android 的 Agent 模型选择器中选 **Command Code**。客户端使用已连接的 PocketBase 账号提交文本或 Markdown 到 `command-code` 队列。执行器用原生 `cmd acp` 处理请求，流式保存消息和工具步骤。普通续聊，以及已建立会话后的失败或取消续接，使用同一 ACP 会话和工作目录。模型由服务账号的 Command Code 配置决定。

部署时先更新包含该队列路由的 AnyWorkflow-workers PocketBase hooks。在执行主机安装 Node.js 22+、`command-code` 和 `requirements-server.txt`。参考 `anyworkflow-agent-command-code.service`，创建 `aw-command-code-agent` 系统账号，再将执行器 checkout、Node.js 安装目录和 `command-code` npm 包只读绑定到 unit 中的 `/run/anyworkflow-command-code-*` 路径。使用 `/etc/anyworkflow/agent-command-code.env` 设置 PocketBase 执行器账号、`GEMINI_PB_URL` 和 `SERVER_AUTH_TOKEN`，文件权限设为 `0600`。

服务可通过环境文件中的 `COMMAND_CODE_API_KEY` 登录。也可先在主机用 `cmd login` 完成浏览器授权，再把登录文件装到服务账号的私有目录：

```sh
sudo install -d -m 0700 -o aw-command-code-agent -g aw-command-code-agent \
  /var/lib/anyworkflow-agent-command-code/.commandcode
sudo install -m 0600 -o aw-command-code-agent -g aw-command-code-agent \
  ~/.commandcode/auth.json /var/lib/anyworkflow-agent-command-code/.commandcode/auth.json
sudo systemctl restart anyworkflow-agent-command-code.service
curl -fsS http://127.0.0.1:18095/health
```

正常启动后，健康检查中的 `pb_executor_queue` 为 `command-code`，`pb_polling_ready` 为 `true`。工作区内的文件编辑和终端命令可自动审批。CLI 的子进程在 Linux Landlock 沙箱内运行，能读写此会话的工作目录，并读取编译和运行所需的系统文件。部署终端功能需要 Landlock ABI 6 或更高。输入 `/model` 可以直接查询当前 ACP 会话的实际模型。

自建部署可以在 `/etc/anyworkflow/agent-command-code.env` 设置 `COMMAND_CODE_SKIP_PERMISSIONS=true`。执行器会为新建和续接的 ACP 会话启用 `bypass` 模式，终端子进程也会停用 Landlock 工作区沙箱。实际能访问的主机文件仍由 systemd 和执行器 Linux 账号的权限决定。
