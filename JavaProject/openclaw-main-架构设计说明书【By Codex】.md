# OpenClaw 项目架构设计说明书

- 文档版本：v1.0
- 生成日期：2026-02-07
- 项目路径：`E:\agi_vault\JavaProject\openclaw-main`

## 1. 项目概述

OpenClaw 是一个本地优先（Local-first）的个人 AI 助手平台。系统以 Gateway 作为统一控制平面，连接多消息渠道、多客户端与多设备节点，并通过统一协议提供会话、工具、任务与事件能力。

### 1.1 核心目标

1. 统一接入：通过单 Gateway 统一接入 WhatsApp/Telegram/Slack/Discord 等渠道与 Web/CLI/移动端客户端。
2. 高扩展性：通过插件机制扩展通道、网关方法、工具、HTTP 路由、服务。
3. 可运维性：支持后台服务化运行、状态探测、热重载、日志与诊断。
4. 安全可控：具备连接鉴权、设备身份签名、配对审批、角色与作用域授权。

## 2. 技术栈与工程形态

### 2.1 技术栈

- 后端运行时：Node.js（要求 >= 22.12.0）
- 语言：TypeScript（主）、JavaScript（脚本）、Swift/Kotlin（客户端与共享库）
- 协议传输：WebSocket + HTTP（同端口复用）
- 前端：Lit + Vite（Control UI）
- 包管理：pnpm workspace

### 2.2 工程结构

仓库为多包单仓（Monorepo）形态，核心目录包括：

- `src/`：网关、CLI、代理运行时、路由、会话、协议、插件框架
- `ui/`：Web 控制台
- `extensions/`：扩展插件
- `apps/`：macOS/iOS/Android 与共享组件
- `docs/`：架构与运维文档

## 3. 总体架构设计

OpenClaw 采用“控制平面集中 + 能力插件化 + 客户端多形态接入”的设计。

### 3.1 逻辑分层

1. 接入层
- CLI 入口：`openclaw` 命令体系
- Web 控制台：浏览器端 UI
- 设备节点：macOS/iOS/Android/headless node

2. 控制平面层（Gateway）
- 统一 WS 协议入口
- 统一 HTTP API/控制台/Canvas/Hook 入口
- 请求分发、事件广播、连接管理

3. 能力层
- Agent 执行（本地嵌入 + CLI backend）
- Channel 管理（按渠道/账号运行时）
- Node 管理（设备能力声明与调用）
- Session 管理（会话状态、策略、快照）

4. 扩展层
- 插件发现、加载、注册与能力注入
- 可扩展 channel/tool/method/http/service/command

### 3.2 核心运行拓扑

```text
Clients (CLI / Web UI / Apps / Nodes)
            |
            v
     Gateway (WS + HTTP)
        |       |      \
        |       |       \__ Plugins (methods/routes/channels/tools)
        |       |
        |       +__ Control UI / OpenAI-Compatible API / OpenResponses / Hooks / Canvas
        |
        +__ Agent Runtime (embedded or CLI backend)
        +__ Channel Runtime Manager
        +__ Node Registry + Pairing + Invoke
        +__ Session Store + Routing
```

## 4. 启动与命令架构

### 4.1 启动链路

- `openclaw.mjs` 作为可执行入口，加载 `dist/entry.js`
- `entry.ts` 负责环境归一化、参数预处理、再进入 `cli/run-main`
- `run-main.ts` 构建程序并按主命令动态加载子命令

### 4.2 命令组织

命令注册采用“注册中心 + 懒加载”模式：

- 核心命令注册集中在 `command-registry`
- 子命令由 `register.subclis` 按 primary command 按需注册
- 减少启动时不必要模块加载成本

## 5. Gateway 架构设计

### 5.1 网关职责

1. 统一接入 WS 客户端与 Node 设备
2. 统一承载 HTTP 端点（控制台、API、hook、canvas）
3. 方法分发与事件广播
4. 渠道生命周期管理
5. Agent 执行调度与结果流转

### 5.2 运行期组装流程

`startGatewayServer` 主要阶段：

1. 配置快照读取、旧配置迁移、schema 校验
2. 插件加载并合并网关方法
3. 创建运行时状态（HTTP Server、WSS、广播器、聊天运行态、去重缓存）
4. 挂载 WS 处理器与 HTTP 处理器
5. 启动 sidecar（browser control、channels、hooks/plugin services）
6. 启动配置热重载与维护定时任务

### 5.3 传输层设计（WS + HTTP）

- HTTP 与 WS 共享监听端口，通过 Upgrade 升级到 WS
- WS 首帧必须是 `connect` 请求
- HTTP 处理链支持：hooks、tools invoke、openai 兼容、openresponses、control UI、canvas

### 5.4 协议与握手

连接握手包含：

1. `connect.challenge`（服务端 nonce）
2. 客户端 `connect` 帧（含协议版本、client 信息、auth、device 信息）
3. 服务端校验后返回 `hello-ok`（方法列表、事件列表、快照、策略、可选设备 token）

校验内容包括：

- 协议版本范围
- 请求帧结构合法性（AJV + schema）
- Origin 校验（Control UI/WebChat）
- Auth 校验（token/password/tailscale）
- 设备身份签名校验
- 配对状态与 role/scope 升级校验

### 5.5 授权模型

角色：

- `operator`
- `node`

作用域：

- `operator.admin`
- `operator.read`
- `operator.write`
- `operator.approvals`
- `operator.pairing`

请求按 method 进行 role + scope 校验，拒绝越权访问。

### 5.6 事件模型

典型事件：

- `agent`
- `chat`
- `presence`
- `tick`
- `shutdown`
- `node.pair.*`
- `device.pair.*`
- `voicewake.changed`
- `exec.approval.*`

## 6. Agent 执行架构

### 6.1 入口流程

`agentCommand` 负责：

1. 解析目标会话（`to` / `sessionId` / `sessionKey` / `agentId`）
2. 解析模型与思考等级
3. 准备技能快照与会话上下文
4. 执行模型回退策略
5. 调用具体执行器（embedded 或 CLI backend）
6. 回写 session 状态并投递结果

### 6.2 Embedded 运行器

特性：

- session lane + global lane 双层排队
- 活跃运行追踪（可排队消息、可中断）
- 失败分类与降级（thinking fallback）
- 认证 profile 轮换
- context overflow 自动 compact（有限重试）

### 6.3 CLI Backend 运行器

适配外部 CLI 模型后端，保持与 embedded 模式一致的会话语义与失败处理策略。

## 7. 通道与路由架构

### 7.1 通道插件化

通道并非硬编码在网关内，而是通过插件注册到运行时 registry。网关按插件能力启动/停止对应 account runtime。

### 7.2 路由决策

路由输入维度：

- `channel`
- `accountId`
- `peer` / `parentPeer`
- `guildId` / `teamId`

输出：

- `agentId`
- `sessionKey`
- `mainSessionKey`
- `matchedBy`（匹配来源）

用于实现“不同渠道/群组/账号指向不同 agent 和会话隔离域”。

## 8. 插件系统架构

### 8.1 发现顺序

插件发现来源（按设计优先级叠加）：

1. 配置指定路径
2. workspace 扩展目录
3. 全局扩展目录
4. bundled 扩展目录

### 8.2 注册能力

插件可注册：

- tool
- hook / typed hook
- channel
- provider
- gateway method
- http handler / route
- cli registrar
- service
- command

并对冲突（如重复 gateway method）进行诊断与阻断。

### 8.3 运行时注入

Gateway 启动阶段加载插件，合并插件方法集到可见网关方法列表，并将插件 handler 注入请求分发链。

## 9. 会话与持久化设计

### 9.1 Session 数据模型

`SessionEntry` 覆盖以下关键状态：

- 会话标识与更新时间
- 模型/provider override
- thinking/verbose/reasoning 等策略
- token 统计
- delivery context
- skills snapshot
- system prompt report

### 9.2 存储与一致性

- 存储介质：JSON session store
- 读缓存：进程内 TTL cache
- 并发控制：文件锁 + 串行写
- 写入策略：原子性保护（平台差异化处理）

### 9.3 路径隔离

会话按 agent 维度隔离存储：

- `state/agents/<agentId>/sessions/sessions.json`
- transcript 也按 agent 目录隔离

## 10. Web Control UI 架构

### 10.1 前端框架

- Lit 自定义元素（`openclaw-app`）
- Vite 构建
- 通过 `GatewayBrowserClient` 与网关 WS 协议直接通信

### 10.2 前端连接能力

`GatewayBrowserClient` 支持：

- 自动重连与退避
- connect challenge 处理
- 设备身份签名（安全上下文）
- 设备 token 本地持久化与回退

## 11. 多端与共享组件

`apps/shared/OpenClawKit` 提供三类 Swift Package 产品：

1. `OpenClawProtocol`：协议模型层
2. `OpenClawKit`：业务能力层
3. `OpenClawChatUI`：聊天 UI 组件层

用于 macOS/iOS 客户端共享协议与核心能力。

## 12. 运维与服务化设计

### 12.1 服务化

按平台抽象统一服务接口：

- macOS：LaunchAgent
- Linux：systemd
- Windows：Scheduled Task

### 12.2 网关运维能力

CLI 支持：

- install/uninstall
- stop/restart
- status/health
- 端口冲突强制释放（`--force`）

并支持多 profile 场景下的实例隔离运行。

## 13. 架构优势与风险

### 13.1 优势

1. 控制面统一，跨客户端一致性高。
2. 插件扩展边界清晰，具备生态演进能力。
3. 会话/路由/授权模型较完整，适配多渠道复杂场景。
4. 运维面成熟，支持常见桌面与服务器环境。

### 13.2 风险与复杂点

1. `server.impl` 组装职责较重，演进时需控制耦合扩散。
2. Control UI 单组件状态较大，需持续模块化治理。
3. 鉴权与配对链路较长，回归测试必须覆盖关键分支。

## 14. 后续优化建议

1. 将 Gateway 启动编排继续拆分为可观测的阶段化 pipeline（init/load/serve/sidecar/reload）。
2. 为 WS 握手与授权链路补充更细粒度的 contract tests（role/scope/pairing matrix）。
3. 对 UI 状态层引入更显式的状态机边界（连接态/会话态/审批态）。
4. 为插件接口增加版本协商与兼容层，降低插件升级断裂风险。

## 15. 关键文件索引

- 入口与命令
  - `openclaw.mjs`
  - `src/entry.ts`
  - `src/cli/run-main.ts`
  - `src/cli/program/command-registry.ts`
  - `src/cli/program/register.subclis.ts`
- 网关核心
  - `src/gateway/server.impl.ts`
  - `src/gateway/server-runtime-state.ts`
  - `src/gateway/server-http.ts`
  - `src/gateway/server-methods.ts`
  - `src/gateway/server-methods-list.ts`
  - `src/gateway/server/ws-connection.ts`
  - `src/gateway/server/ws-connection/message-handler.ts`
- Agent 与会话
  - `src/commands/agent.ts`
  - `src/agents/pi-embedded-runner/run.ts`
  - `src/agents/pi-embedded-runner/runs.ts`
  - `src/agents/cli-runner.ts`
  - `src/config/sessions/types.ts`
  - `src/config/sessions/store.ts`
  - `src/config/sessions/paths.ts`
- 插件与路由
  - `src/plugins/discovery.ts`
  - `src/plugins/loader.ts`
  - `src/plugins/registry.ts`
  - `src/gateway/server-plugins.ts`
  - `src/routing/resolve-route.ts`
- 控制台与多端
  - `ui/src/ui/gateway.ts`
  - `ui/src/ui/app.ts`
  - `apps/shared/OpenClawKit/Package.swift`

---

本说明书基于当前仓库源码与文档快照整理，适合作为后续架构评审、模块拆分和交付设计基线。
