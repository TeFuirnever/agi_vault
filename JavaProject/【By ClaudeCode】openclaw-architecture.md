# OpenClaw 项目架构设计说明书

## 1. 项目概述

| 属性 | 详情 |
|------|------|
| **项目名称** | openclaw |
| **版本** | 2026.2.6 |
| **类型** | WhatsApp Gateway CLI with Pi RPC Agent |
| **技术栈** | Node.js 22.12+, TypeScript, pnpm |
| **许可证** | MIT |
| **核心依赖** | @whiskeysockets/baileys (WhatsApp), @mariozechner/pi-* (AI Agent) |

**核心功能**：
- WhatsApp 消息网关（基于 Baileys Web）
- 多平台消息集成（Telegram, Discord, Slack, Line, iMessage, Signal 等）
- AI Agent 运行时（Pi Protocol）
- 插件扩展系统
- 跨平台应用支持（macOS, iOS, Android）

---

## 2. 技术架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          应用层 (Applications)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   macOS     │  │    iOS      │  │   Android   │  │  Terminal   │       │
│  │   App       │  │    App      │  │    App      │  │    (TUI)    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
├─────────────────────────────────────────────────────────────────────────┤
│                          包层 (Packages)                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐                                       │
│  │  clawdbot   │  │  moltbot     │  ← 独立 bot 子系统                    │
│  └─────────────┘  └─────────────┘                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                         核心源码层 (Src)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     CLI / 入口层                                  │    │
│  │  entry.ts → cli/run-main.ts → cli/program.ts                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │    Gateway        │  │    Agents         │  │    Channels       │     │
│  │  (WhatsApp核心)   │  │   (AI Agent)      │  │   (多平台集成)    │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │    Plugins        │  │    Providers     │  │    Config         │     │
│  │   (插件系统)      │  │   (模型提供者)    │  │   (配置管理)      │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
├─────────────────────────────────────────────────────────────────────────┤
│                        扩展层 (Extensions)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  whatsapp | telegram | discord | slack | line | imessage | signal | ... │
│  memory-core | memory-lancedb | voice-call | googlechat | msteams | ...  │
├─────────────────────────────────────────────────────────────────────────┤
│                        基础设施层                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Docker | SQLite-Vec | Playwright | Baileys | Express | Hono | ...      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 项目目录结构

```
openclaw-main/
├── apps/                    # 平台特定应用
│   ├── macos/              # macOS 原生应用 (Swift)
│   ├── ios/                # iOS 应用 (Swift)
│   ├── android/             # Android 应用 (Kotlin/Java)
│   └── shared/             # 跨平台共享代码 (OpenClawKit)
│
├── packages/               # 子包 (独立发布单元)
│   ├── clawdbot/           # ClawDBot 子系统
│   └── moltbot/            # MoltBot 子系统
│
├── src/                    # 核心源码
│   ├── agents/             # AI Agent 模块
│   │   ├── pi-embedded-*   # Pi Protocol 嵌入式运行
│   │   ├── auth-profiles/  # 认证配置
│   │   ├── model-*/        # 模型配置
│   │   ├── tools/          # 工具集
│   │   ├── sandbox/        # 沙箱管理
│   │   └── skills/         # 技能系统
│   │
│   ├── gateway/            # WhatsApp 网关
│   │   ├── server/         # 网关服务器
│   │   ├── hooks/          # 钩子系统
│   │   └── protocol/        # 协议定义
│   │
│   ├── channels/           # 多平台消息集成
│   │   ├── web/            # Web Channel
│   │   ├── plugins/         # Channel 插件
│   │   └── (各平台)         # telegram, discord, slack, etc.
│   │
│   ├── plugins/            # 插件系统
│   │   ├── runtime/        # 运行时
│   │   ├── loader.ts       # 加载器
│   │   └── registry.ts     # 注册表
│   │
│   ├── cli/                # CLI 命令行
│   │   ├── program/        # 命令程序
│   │   ├── gateway-cli/    # 网关命令
│   │   ├── plugins-cli/    # 插件命令
│   │   └── nodes-cli/      # 节点命令
│   │
│   ├── config/             # 配置系统
│   │   ├── schema.ts       # 配置Schema
│   │   ├── zod-schema/     # Zod验证
│   │   ├── legacy/         # 旧配置迁移
│   │   └── sessions/       # 会话配置
│   │
│   ├── providers/          # AI 模型提供者
│   │   └── github-copilot-* # GitHub Copilot
│   │
│   ├── tui/                # 终端UI
│   ├── memory/              # 记忆系统
│   ├── infra/               # 基础设施
│   ├── process/             # 进程管理
│   ├── media/               # 媒体处理
│   ├── markdown/            # Markdown处理
│   ├── utils/               # 工具函数
│   └── ...
│
├── extensions/              # 平台扩展 (27+)
│   ├── whatsapp/
│   ├── telegram/
│   ├── discord/
│   ├── slack/
│   ├── line/
│   ├── imessage/
│   ├── signal/
│   ├── memory-core/
│   ├── memory-lancedb/
│   └── ...
│
├── skills/                  # 内置技能
├── assets/                  # 静态资源
├── docs/                    # 文档
├── test/                    # 测试
├── scripts/                 # 构建脚本
├── ui/                      # Web UI
└── openclaw.mjs            # CLI 入口
```

---

## 4. 核心技术栈

### 4.1 核心运行时

```
{
  "node": ">=22.12.0",
  "typescript": "^5.9.3",
  "pnpm": "^10.23.0"
}
```

### 4.2 关键依赖

| 类别 | 依赖 | 版本 | 用途 |
|------|------|------|------|
| **消息网关** | @whiskeysockets/baileys | 7.0.0-rc.9 | WhatsApp Web 协议实现 |
| **AI Agent** | @mariozechner/pi-agent-core | 0.52.7 | Pi Agent 核心 |
| | @mariozechner/pi-ai | 0.52.7 | AI 集成 |
| | @mariozechner/pi-coding-agent | 0.52.7 | 编码 Agent |
| | @mariozechner/pi-tui | 0.52.7 | TUI 界面 |
| **消息平台** | grammy / @grammyjs/runner | 1.39.3 | Telegram Bot SDK |
| | @slack/bolt | 4.6.0 | Slack Bot 框架 |
| | @line/bot-sdk | 10.6.0 | Line Bot SDK |
| | @homebridge/ciao | 1.3.4 | mDNS 发现 |
| **Web 框架** | express | 5.2.1 | HTTP 服务器 |
| | hono | 4.11.8 | 轻量 Web 框架 |
| **AI 提供商** | @aws-sdk/client-bedrock | ^3.985.0 | AWS Bedrock |
| | openai (via pi-ai) | - | OpenAI API |
| | google-generativeai | - | Google Gemini |
| **数据库** | sqlite-vec | 0.1.7-alpha.2 | 向量搜索 |
| **测试** | vitest | ^4.0.18 | 测试框架 |
| | playwright-core | 1.58.2 | 浏览器自动化 |

---

## 5. 模块详解

### 5.1 Gateway 模块 (WhatsApp 网关)

**路径**: `src/gateway/`

**核心组件**:

| 文件 | 功能 |
|------|------|
| `server/impl.ts` | Gateway 服务器实现 |
| `server-http.ts` | HTTP 接口 |
| `server-chat.ts` | 聊天处理 |
| `session-utils.ts` | 会话工具 |
| `hooks.ts` | 钩子系统 |
| `client.ts` | Baileys 客户端 |
| `auth.ts` | 认证管理 |

**数据流**:
```
WhatsApp User → Baileys WebSocket → Gateway Server → Agent → Response → User
```

### 5.2 Agents 模块 (AI Agent 系统)

**路径**: `src/agents/`

**子模块**:

| 子模块 | 功能 |
|--------|------|
| `pi-embedded-runner/` | 嵌入式 Pi Agent 运行器 |
| `pi-embedded-subscribe/` | 订阅式消息处理 |
| `pi-tools/` | 工具集管理 |
| `auth-profiles/` | 认证配置管理 |
| `model-*/` | 模型配置和发现 |
| `bash-tools.*` | Bash 命令工具 |
| `sandbox/` | 沙箱管理 |
| `skills/` | 技能系统 |

**核心接口**:
```
Agent Context → Tools → LLM → Response → Tools → Output
```

### 5.3 Channels 模块 (多平台集成)

**路径**: `src/channels/`

**支持的平台**:
- `telegram/` - Telegram
- `discord/` - Discord
- `slack/` - Slack
- `line/` - Line
- `imessage/` - iMessage (via BlueBubbles/MacOS)
- `signal/` - Signal
- `web/` - Web Channel

**Channel 通用接口**:
```
Platform → Channel Adapter → Gateway → Agent → Response → Gateway → Platform
```

### 5.4 Plugins 模块 (插件系统)

**路径**: `src/plugins/`

**核心组件**:

| 文件 | 功能 |
|------|------|
| `loader.ts` | 插件加载器 |
| `registry.ts` | 插件注册表 |
| `install.ts` | 安装逻辑 |
| `hooks.ts` | 钩子扩展 |
| `types.ts` | 类型定义 |
| `runtime/` | 运行时环境 |

### 5.5 CLI 模块 (命令行界面)

**路径**: `src/cli/`

**命令结构**:

```
openclaw
├── gateway          # 网关管理
├── plugins          # 插件管理
├── nodes            # 节点管理
├── channels         # 频道管理
├── models           # 模型配置
├── memory           # 记忆管理
├── skills           # 技能管理
├── sandbox          # 沙箱管理
├── hooks            # 钩子管理
├── logs             # 日志查看
├── update           # 更新管理
├── config           # 配置管理
└── completion       # 命令补全
```

---

## 6. 核心设计模式

### 6.1 依赖注入模式

```typescript
// src/cli/deps.ts
export function createDefaultDeps(): CliDeps {
  return {
    config: loadConfig(),
    sessionStore: loadSessionStore(),
    logger: createLogger(),
    // ...
  };
}
```

### 6.2 插件架构模式

```typescript
// src/plugins/loader.ts
export async function loadPlugins(config: PluginConfig): Promise<Plugin[]> {
  const manifests = await discoverPlugins();
  const plugins = await Promise.all(
    manifests.map(loadPluginEntry)
  );
  return plugins.filter(Boolean);
}
```

### 6.3 事件驱动模式

```typescript
// src/gateway/hooks.ts
export class HookEmitter {
  on(event: string, handler: HookHandler): void;
  emit(event: string, data: unknown): Promise<void>;
}
```

### 6.4 Provider 抽象模式

```typescript
// 统一的 AI Provider 接口
interface ModelProvider {
  chat(request: ChatRequest): Promise<ChatResponse>;
  stream(request: ChatRequest): AsyncIterator<Chunk>;
}
```

### 6.5 Channel Adapter 模式

```typescript
// src/channels/dock.ts
export class ChannelDock {
  async registerAdapter(platform: string, adapter: ChannelAdapter): void;
  async routeMessage(msg: InboundMessage): Promise<void>;
}
```

---

## 7. 数据流设计

### 7.1 消息处理主流程

```
┌──────────────┐
│  User Input  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Channel    │  ← Telegram/Discord/Slack/WhatsApp/...
│   Adapter     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Gateway    │  ← 消息路由、认证、会话管理
│   Server      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Agent     │  ← Pi Protocol Agent, 工具调用, LLM 调用
│   Runtime    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Gateway    │  ← 响应格式化
│   Server      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Channel    │  ← 发送回对应平台
│   Adapter     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    User      │
└──────────────┘
```

### 7.2 认证流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Auth Profile System                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Message                                               │
│      │                                                       │
│      ▼                                                       │
│  Resolve Auth Profile (Round-robin / Priority)             │
│      │                                                       │
│      ▼                                                       │
│  Check Credentials (API Key / Token / OAuth)               │
│      │                                                       │
│      ├─→ Valid → Route to Model Provider                    │
│      │                                                       │
│      └─→ Invalid → Try Next Profile / Fail                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 工具调用流程

```
Agent Decision
     │
     ▼
┌─────────────────┐
│ Tool Selection  │  ← 基于 schema 和 policy 选择工具
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Pre-call      │  ← 参数验证、权限检查、workspace 设置
│   Hooks         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tool Execute   │  ← Bash / File System / API / Custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Post-call     │  ← 结果处理、错误处理
│   Hooks         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return Result │
│  to Agent      │
└─────────────────┘
```

---

## 8. 依赖关系图

### 8.1 核心依赖

```
entry.ts
    │
    ├── cli/
    │   ├── program.ts → commands.ts → 各子命令
    │   └── deps.ts → createDefaultDeps()
    │
    ├── config/
    │   ├── schema.ts
    │   └── validation.ts
    │
    ├── gateway/
    │   ├── server/impl.ts
    │   └── session-utils.ts
    │
    └── infra/
        ├── errors.ts
        ├── dotenv.ts
        └── ports.ts
```

### 8.2 模块间依赖

```
agents/
    ├── pi-embedded-* → @mariozechner/pi-*
    ├── auth-profiles → config/
    ├── model-* → providers/
    ├── tools/ → sandbox/
    └── skills/ → workspace/

gateway/
    ├── server/ → infra/ (HTTP, WebSocket)
    ├── hooks/ → plugins/
    └── session-utils → config/

channels/
    ├── dock.ts → gateway/
    └── */ → channel-web.ts

plugins/
    ├── loader.ts → infra/
    └── runtime/ → gateway/hooks/

config/
    ├── schema.ts → zod-schema/
    └── legacy/ → current schema
```

---

## 9. 配置文件和构建系统

### 9.1 构建脚本

```json
// package.json scripts
{
  "build": "pnpm canvas:a2ui:bundle && tsdown && ...",
  "dev": "node scripts/run-node.mjs",
  "gateway:dev": "OPENCLAW_SKIP_CHANNELS=1 ...",
  "test": "vitest run",
  "lint": "oxlint --type-aware",
  "format": "oxfmt --check"
}
```

### 9.2 构建工具链

| 工具 | 用途 |
|------|------|
| `tsdown` | TypeScript 编译打包 |
| `oxfmt` | 代码格式化 |
| `oxlint` | 代码检查 |
| `vitest` | 单元/E2E 测试 |
| `tsx` | TypeScript 脚本执行 |
| `rolldown` | 模块打包 |

### 9.3 配置文件

| 文件 | 用途 |
|------|------|
| `tsconfig.json` | TypeScript 配置 |
| `vitest.config.ts` | 测试配置 |
| `.pre-commit-config.yaml` | Git 钩子 |
| `docker-compose.yml` | Docker 服务 |
| `Dockerfile` | 容器镜像 |

### 9.4 插件配置 (Schema)

```typescript
// src/config/schema.ts
export const PluginConfigSchema = z.object({
  id: z.string(),
  name: z.string(),
  version: z.string(),
  enabled: z.boolean(),
  permissions: z.array(z.string()),
  config: z.record(z.unknown()),
});
```

---

## 10. 入口点和主要组件

### 10.1 程序入口

**文件**: `openclaw.mjs` → `src/entry.ts`

```
openclaw.mjs
    │
    └── require('openclaw') → src/index.ts
         │
         └── buildProgram() → Commander CLI
              │
              └── runCli() → src/cli/run-main.ts
                   │
                   ├── loadConfig()
                   ├── createDeps()
                   ├── runGateway() / runTUI() / runAgent()
                   └── handleShutdown()
```

### 10.2 Gateway 服务器启动

```typescript
// src/gateway/server/impl.ts
export async function startGateway(config: GatewayConfig): Promise<GatewayServer> {
  const server = new GatewayServer(config);

  await server.initialize();
  await server.startBaileys();
  await server.startHttpServer();
  await server.startWsServer();

  return server;
}
```

### 10.3 CLI 命令注册

```typescript
// src/cli/program/build-program.ts
export function buildProgram(): Command {
  const program = new Command();

  program
    .name('openclaw')
    .version(getVersion())
    .addCommand(gatewayCli)
    .addCommand(pluginsCli)
    .addCommand(nodesCli)
    .addCommand(channelsCli)
    // ... 更多命令

  return program;
}
```

---

## 11. 扩展点和设计考量

### 11.1 扩展点

| 类型 | 扩展方式 | 示例 |
|------|----------|------|
| **Channel** | Channel Adapter | extensions/telegram, extensions/discord |
| **Plugin** | Plugin SDK | src/plugin-sdk/ |
| **Tool** | Tool Definition | pi-tools/create-*-tools |
| **Model Provider** | Provider Interface | providers/github-copilot-* |
| **Hook** | Hook Point | gateway/hooks, plugins/hooks |
| **UI Theme** | TUI Theme | tui/theme/ |

### 11.2 设计考量

1. **跨平台支持**
   - CLI 统一入口
   - 平台适配器抽象
   - 配置兼容层

2. **安全性**
   - 插件权限系统
   - 工具执行审批
   - 会话隔离
   - 输入验证

3. **可扩展性**
   - 插件热加载
   - Hook 机制
   - Provider 接口抽象

4. **性能**
   - WebSocket 持久连接
   - 并发会话管理
   - 缓存策略

5. **测试覆盖**
   - 70% 代码覆盖率要求
   - E2E 测试
   - Live 测试
   - Docker 测试套件

---

## 12. 关键文件路径

| 组件 | 路径 |
|------|------|
| 入口 | `openclaw.mjs` |
| CLI 程序 | `src/cli/program.ts` |
| Gateway | `src/gateway/server/impl.ts` |
| Agent | `src/agents/pi-embedded-runner.ts` |
| 配置 | `src/config/schema.ts` |
| 插件 | `src/plugins/loader.ts` |
| Channels | `src/channels/dock.ts` |
| TUI | `src/tui/` |
| Memory | `src/memory/` |

---

## 13. 总结

OpenClaw 是一个设计精良的即时通讯网关和 AI Agent 运行时系统，采用了以下核心设计原则：

1. **分层架构**: CLI → Core → Modules → Extensions
2. **插件化设计**: Channel、Plugin、Tool 均可扩展
3. **统一抽象**: 多平台消息统一处理
4. **AI 优先**: 内置 Pi Protocol Agent 支持
5. **跨平台**: macOS/iOS/Android + Web + CLI
6. **生产就绪**: 完善的测试、错误处理、配置管理

该项目适合作为企业级消息网关和 AI Agent 平台的参考架构。

---

*文档版本: 2026.2.6*
*项目位置: E:\agi_vault\JavaProject\openclaw-main*
