<div align="center">
  <h1>VideoClaw</h1>
  <p><strong>The Agent OS for AI Video Generation</strong></p>
  <p>
    Orchestrate multiple AI models. Automate entire video pipelines.<br/>
    From script to publish — one command, one flow, zero babysitting.
  </p>
  <p>
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#features">Features</a> &bull;
    <a href="#clawflow">ClawFlow</a> &bull;
    <a href="#architecture">Architecture</a> &bull;
    <a href="#supported-models">Models</a> &bull;
    <a href="#contributing">Contributing</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License" />
    <img src="https://img.shields.io/badge/python-3.12+-green" alt="Python" />
    <img src="https://img.shields.io/badge/tests-37%20passing-brightgreen" alt="Tests" />
    <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Docker-lightgrey" alt="Platform" />
  </p>
</div>

---

> **VideoClaw doesn't generate videos. It orchestrates the models that do.**
>
> Think Kubernetes for containers, but for AI video generation.

## Why VideoClaw?

You've tried Sora, Runway, Kling, CogVideo... Each is impressive alone.
But making a *real* video still means:

- Writing prompts for each shot manually
- Waiting, downloading, re-uploading between tools
- No idea what it costs until the bill arrives
- Starting from scratch when one shot fails
- Manually stitching, adding subtitles, music, voiceover

**VideoClaw fixes all of this.**

## Quick Start

```bash
# Install
pip install videoclaw

# Check system readiness
claw doctor

# Generate a video from a single prompt
claw generate "A 30-second product intro for a smart watch, cinematic style"

# Or run a YAML pipeline
claw flow run examples/product-promo.yaml
```

## Features

### ClawFlow — YAML Pipelines

Define your entire video pipeline in a version-controllable YAML file:

```yaml
name: product-promo
variables:
  product: "VideoClaw"

steps:
  - id: script
    type: script_gen
    params:
      prompt: "Write a promo for {{product}}"

  - id: storyboard
    type: storyboard
    depends_on: [script]

  - id: hero_shot
    type: video_gen
    depends_on: [storyboard]
    params:
      prompt: "{{product}} logo reveal, cinematic"
      model_id: sora

  - id: narration
    type: tts
    depends_on: [script]

  - id: compose
    type: compose
    depends_on: [hero_shot, narration]

  - id: render
    type: render
    depends_on: [compose]
```

Features: variable interpolation (`{{var}}`), dependency validation, cycle detection, parallel execution of independent steps.

```bash
claw flow validate my-pipeline.yaml   # Check without running
claw flow run my-pipeline.yaml        # Execute the pipeline
```

### Multi-Model Orchestration

One pipeline, multiple models. VideoClaw picks the best model for each shot based on your strategy — quality, speed, or cost.

```
Same 30s video:
  All Sora:              $2.50  ~3 min
  VideoClaw hybrid:      $0.47  ~2 min   <- auto-routes simple shots locally
  VideoClaw all-local:   $0.00  ~6 min
```

### Director Agent (LLM-Powered)

The Director takes your prompt and uses an LLM to produce a structured production plan: scene breakdown, visual descriptions, camera movements, voiceover script, and music style. Supports prompt refinement based on reviewer feedback.

### Video Agents

AI agents that think, act, and self-correct. The Director plans, the Cameraman generates, the Reviewer QA-checks. Bad shot? Auto-retry with improved prompts.

### Built-in Cost Tracking

Real-time per-node cost display. Budget guards. Optimization hints. Know exactly what every video costs.

### Smart DAG Executor

Dependency-aware parallel execution. Shots generate concurrently. If one fails, others keep running. Resume from any checkpoint.

### Apple Silicon Ready

Designed for local inference on Mac. MPS backend support for PyTorch-based models.

## Architecture

```
  You --> Director Agent --> Planner --> DAG Executor
              |                              |
              v                     +--------+--------+
         Scriptwriter              v        v        v
         Cameraman              [Sora]  [CogVideo] [Mock]
         Reviewer                  |        |        |
              |                    +--------+--------+
              v                             v
         Quality Gate --> Compose --> Output
```

Six-layer design:

| Layer | Purpose |
|-------|---------|
| Interface | CLI (`claw`) + Web UI + REST API |
| Gateway | FastAPI server, WebSocket progress |
| Agent Runtime | Director, Planner, DAG Executor |
| Generation | Script, Storyboard, Video, TTS, Music, Compose |
| Model Adapters | Protocol-based adapters (Sora, Mock, etc.) |
| Distribution | Publishers (YouTube, Bilibili, etc.) |

## Supported Models

| Category | Models | Mode |
|----------|--------|------|
| Video | Sora (OpenAI), Runway, Kling, CogVideoX, Wan2.2, AnimateDiff | Cloud + Local |
| LLM | Claude, GPT, Qwen, DeepSeek, Ollama (via LiteLLM) | Cloud + Local |
| TTS | Edge-TTS, Fish-Speech, ElevenLabs, ChatTTS | Cloud + Local |
| Music | Suno, Udio, MusicGen | Cloud + Local |

> Adding a new model? Implement the `VideoModelAdapter` protocol (4 async methods). No ABC inheritance needed.

## CLI Commands

```bash
claw generate <prompt>        # Full pipeline: script -> shots -> compose -> output
claw flow run <file.yaml>     # Execute a ClawFlow YAML pipeline
claw flow validate <file.yaml># Validate flow without running
claw doctor                   # System health check
claw model list               # List available model adapters
claw model pull <id>          # Download a local model
claw project list             # List all projects
claw project show <id>        # Show project details + cost
claw template list            # List flow templates
claw template use <name>      # Generate from template
```

## REST API

```bash
# Start the server
uvicorn videoclaw.server.app:create_app --factory

# Endpoints
GET  /health                  # Health check
POST /api/projects/           # Create project
GET  /api/projects/           # List projects
GET  /api/projects/{id}       # Get project details
DELETE /api/projects/{id}     # Delete project
POST /api/generate/           # Start generation pipeline
POST /api/generate/flow       # Run a ClawFlow pipeline
GET  /api/generate/{id}/status# Check generation status
WS   /ws/{project_id}        # Real-time progress updates
```

## Docker

```bash
docker compose up
# API available at http://localhost:8000
```

## Project Structure

```
videoclaw/
├── src/videoclaw/
│   ├── cli.py              # CLI entry point (Typer + Rich)
│   ├── config.py           # Configuration (Pydantic Settings)
│   ├── core/               # Director, DAG engine, state, events
│   ├── agents/             # Video Agent protocol + roles
│   ├── models/             # Model adapters, registry, LLM wrapper
│   ├── generation/         # Script, storyboard, video, audio, compose
│   ├── cost/               # Cost tracking + budget guards
│   ├── flow/               # ClawFlow YAML parser + runner
│   ├── server/             # FastAPI server + WebSocket
│   ├── storage/            # Local filesystem storage
│   ├── publishers/         # YouTube, Bilibili publishers
│   └── utils/              # FFmpeg helpers
├── examples/               # Example ClawFlow YAML pipelines
├── tests/                  # 37 tests (unit + integration)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Configuration

```bash
# Set environment variables or use .env file
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For Sora/GPT | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Claude | Anthropic API key |
| `VIDEOCLAW_DEFAULT_LLM` | No | Default LLM (default: `gpt-4o`) |
| `VIDEOCLAW_DEFAULT_VIDEO_MODEL` | No | Default video model (default: `mock`) |
| `VIDEOCLAW_PROJECTS_DIR` | No | Project storage path (default: `./projects`) |
| `VIDEOCLAW_BUDGET_DEFAULT_USD` | No | Default budget cap (default: `10.0`) |

## Development

```bash
git clone https://github.com/your-org/videoclaw.git
cd videoclaw
pip install -e ".[dev]"
pytest                    # Run 37 tests
ruff check src/ tests/    # Lint
```

## Roadmap

- [x] **Phase 1**: Core engine, DAG executor, model adapters, CLI, cost tracking
- [x] **Phase 2**: FastAPI server, WebSocket, storage, publishers, test suite
- [x] **Phase 3**: ClawFlow YAML engine, integration tests, Docker
- [x] **Phase 4**: Director LLM integration, GitHub Actions CI, flow templates
- [ ] **Phase 5**: Web UI (ClawFlow visual editor)
- [ ] **Phase 6**: Plugin marketplace (ClawHub) + industry templates
- [ ] **Phase 7**: Multi-agent collaboration + quality review loop

## License

Apache 2.0 — use it for anything.
