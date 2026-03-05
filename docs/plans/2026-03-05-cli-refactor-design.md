# CLI-First Refactor Design

**Date:** 2026-03-05
**Status:** Completed

## Goal

Remove the Web UI (Next.js frontend) from VideoClaw, focusing on a CLI-first architecture while preserving all existing capabilities.

## Approach

Option 3: Remove `web/` directory and `claw ui` command only. Keep FastAPI server as optional headless API.

## Changes Made

| Action | Target | Rationale |
|--------|--------|-----------|
| Deleted | `web/` directory | Next.js frontend, no Python coupling |
| Removed | `claw ui` CLI command | Referenced removed `web/` directory |
| Cleaned | `.gitignore` web entries | `web/.next/`, `web/node_modules/`, `node_modules/` |
| Cleaned | `Makefile` `ui` target | Referenced `claw ui` |
| Updated | `README.md` | Architecture table, project structure, roadmap |
| Preserved | `src/videoclaw/server/` | Headless API, optional `[server]` dependency |
| Preserved | All CLI commands (except `claw ui`) | Core capabilities unchanged |

## Verification

- 36/37 tests pass (1 pre-existing failure unrelated to refactor)
- `claw ui` command removed; all other commands intact
- `pyproject.toml` dependency structure unchanged

## Existing Capabilities Summary

### CLI Commands
- `claw generate <prompt>` — full pipeline orchestration
- `claw flow run/validate <yaml>` — YAML pipeline execution
- `claw doctor` — system health check
- `claw model list/pull` — model adapter management
- `claw project list/show` — project management
- `claw template list/use` — flow templates

### Core Engine
- Director Agent (LLM-powered scene planning)
- DAG-based parallel executor with retry logic
- ClawFlow YAML engine with variable interpolation + cycle detection
- 5 video model adapters (OpenAI/Kling/MiniMax/Zhipu/Mock)
- LLM integration via LiteLLM (multi-provider)
- Cost tracking with budget guards
- Event system (async pub/sub)
- Project state persistence (JSON)

### Optional Headless API
- FastAPI REST + WebSocket (install with `pip install videoclaw[server]`)
