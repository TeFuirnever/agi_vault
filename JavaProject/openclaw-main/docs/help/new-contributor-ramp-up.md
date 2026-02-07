# New Contributor Ramp-Up (Core Path)

This guide is for new contributors who need to become productive on OpenClaw architecture, module design, and implementation quickly.

## Goal

Become able to:
- explain the end-to-end flow from CLI to Gateway to Agent to Channel
- place new code in the correct module without guessing
- ship a first low-risk change with tests

## Baseline

- Runtime: Node >= 22.12.0
- Package manager: pnpm
- Main quality gates:
  - `pnpm build`
  - `pnpm check`
  - `pnpm test`

## Architecture Snapshot

OpenClaw is a local-first assistant platform.

Primary control plane:
- Gateway (WS + HTTP, default port 18789)

Primary call path:
1. CLI entry (`openclaw.mjs`)
2. Runtime entry (`src/entry.ts`)
3. CLI bootstrap (`src/cli/run-main.ts`)
4. Command assembly (`src/cli/program/build-program.ts` + `src/cli/program/command-registry.ts`)
5. Gateway runtime (`src/gateway/server.impl.ts`)
6. Agent execution + session/routing + channel delivery

## First 5 Days (Systematic Reading)

### Day 1-2: CLI Boot and Command Routing

Read in this exact order:
1. `openclaw.mjs`
2. `src/entry.ts`
3. `src/cli/run-main.ts`
4. `src/cli/program/build-program.ts`
5. `src/cli/program/command-registry.ts`

Checkpoint:
- You can explain direct command routing (`tryRouteCli`) vs full Commander parse.
- You can explain lazy subcommand registration in `register.subclis`.

### Day 3-4: Gateway Assembly and Protocol Surface

Read in this exact order:
1. `src/gateway/server.ts`
2. `src/gateway/server.impl.ts`
3. `src/gateway/server-methods-list.ts`
4. `src/gateway/server-ws-runtime.ts`

Checkpoint:
- You can explain startup lifecycle: config snapshot, migration, validation, plugin loading, runtime state creation.
- You can explain WS handshake and where method/event lists are exposed.

### Day 5: Agent, Session, Routing, Channels

Read in this exact order:
1. `src/commands/agent.ts`
2. `src/infra/outbound/*`
3. `src/routing/*`
4. `src/channels/plugins/index.ts`
5. any 2 extension manifests (`extensions/*/openclaw.plugin.json`)

Checkpoint:
- You can explain session key resolution and route-to-agent mapping.
- You can identify built-in channels vs extension channels and where each is loaded.

## Module Boundary Rules (Design Guardrails)

- CLI wiring stays in `src/cli/*`; command behavior in `src/commands/*`.
- Gateway transport/state belongs to `src/gateway/*`.
- Channel runtime concerns belong to channel modules or extension packages.
- Shared infra (env, paths, retry, ports, guards) belongs to `src/infra/*`.
- Do not add plugin-only dependencies to root package unless core uses them.

## First Implementation Ticket (Low Risk)

Recommended first ticket shape:
- Add a small read-only diagnostics improvement in one command output path.
- Keep behavior backward compatible.
- Add/adjust colocated tests (`*.test.ts`).

Definition of done:
1. single-module scope
2. tests added/updated
3. all gates pass (`pnpm build && pnpm check && pnpm test`)
4. no protocol or default behavior break

## PR Checklist for New Contributors

- [ ] Scope is focused (one module/subsystem)
- [ ] Behavior change is intentional and documented in PR description
- [ ] Tests cover both happy path and one failure/edge path
- [ ] Ran build/lint/test gates locally
- [ ] Followed existing command and output formatting conventions

## Suggested Next Steps After First PR

1. Take one cross-module task that touches `commands + gateway`.
2. Then take one channel-related task touching built-in and extension awareness.
3. Finally, author a short module design note before each non-trivial implementation.
