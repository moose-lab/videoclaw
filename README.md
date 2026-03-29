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
    <img src="https://img.shields.io/badge/license-Modified%20MIT-blue" alt="License" />
    <img src="https://img.shields.io/badge/python-3.12+-green" alt="Python" />
    <img src="https://img.shields.io/badge/tests-37%20passing-brightgreen" alt="Tests" />
    <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Docker-lightgrey" alt="Platform" />
  </p>
</div>

---

## Install

```bash
git clone https://github.com/moose-lab/videoclaw.git
cd videoclaw
uv sync            # Python 3.12+, installs deps + creates .venv
uv run claw --help # Done. No activation needed.
```

> Or activate the venv to use `claw` directly: `source .venv/bin/activate && claw --help`

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
# Clone and install (requires Python 3.12+ and uv)
git clone https://github.com/moose-lab/videoclaw.git
cd videoclaw
uv sync                          # Install dependencies + create .venv

# Option A: Use uv run (recommended, no activation needed)
uv run claw --help
uv run claw doctor               # Check system readiness

# Option B: Activate virtualenv, then use claw directly
source .venv/bin/activate
claw --help
claw doctor

# Generate a video from a single prompt
uv run claw generate "A 30-second product intro for a smart watch, cinematic style"

# Or run individual stages independently
uv run claw video "A cat riding a skateboard" -d 5 -o cat.mp4
uv run claw image "Character portrait" --provider gemini -o portrait.png
uv run claw tts "Hello world" --lang en -o hello.mp3
uv run claw storyboard "Product unboxing" -d 30 -o shots.json

# Agent-friendly: JSON output for programmatic use
uv run claw -j video "sunset over ocean" -o sunset.mp4
# → {"ok": true, "command": "video", "data": {"path": "...", "cost_usd": 0.05}, "error": null}

# Or run a YAML pipeline
uv run claw flow run examples/product-promo.yaml
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
| Interface | CLI (`claw`) + REST API (optional) |
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

> All commands support `--json / -j` for structured JSON output (agent-friendly).

```bash
# Full pipeline
claw generate <prompt>              # Script → shots → compose → render
claw generate <prompt> --dry-run    # Preview DAG without executing

# Single-stage commands (run each step independently)
claw video <prompt>                 # Generate a single video clip
claw image <prompt>                 # Generate a single image
claw tts <text>                     # Text-to-speech (supports stdin pipe)
claw storyboard <prompt>            # Decompose prompt into shot list
claw compose <v1.mp4> <v2.mp4> ...  # Compose multiple clips together
claw render <input.mp4>             # Encode/render final video
claw subtitle <scenes.json>         # Generate SRT/ASS subtitles

# ClawFlow YAML pipelines
claw flow run <file.yaml>           # Execute a pipeline
claw flow validate <file.yaml>      # Validate without running

# AI short drama series
claw drama new <synopsis>           # Create from concept
claw drama import <script.docx>     # Import complete script
claw drama list                     # List all series
claw drama show <id>                # Show series details
claw drama run <id>                 # Execute generation pipeline

# Management
claw config show                    # View all config (API keys masked)
claw config check                   # Validate config completeness
claw doctor                         # System health check
claw model list                     # List model adapters
claw project list                   # List all projects
claw project show <id>              # Show project details
claw project delete <id>            # Delete project and assets
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
│   ├── cli/                # CLI package (Typer + Rich)
│   │   ├── _app.py         # App definition, validators, helpers
│   │   ├── _output.py      # JSON output mode (OutputContext)
│   │   ├── stage.py        # Single-stage commands (video/image/tts/...)
│   │   ├── generate.py     # Full pipeline command
│   │   ├── drama.py        # Drama series commands
│   │   ├── config_cmd.py   # Config management
│   │   └── ...             # doctor, model, project, template, flow
│   ├── config.py           # Configuration (Pydantic Settings)
│   ├── core/               # Director, DAG engine, state, events
│   ├── agents/             # Video Agent protocol + roles
│   ├── models/             # Model adapters, registry, LLM wrapper
│   ├── generation/         # Script, storyboard, video, audio, compose
│   ├── drama/              # AI short drama orchestration
│   ├── cost/               # Cost tracking + budget guards
│   ├── flow/               # ClawFlow YAML parser + runner
│   ├── server/             # FastAPI REST API (optional, headless)
│   ├── storage/            # Local filesystem storage
│   ├── publishers/         # YouTube, Bilibili publishers
│   └── utils/              # FFmpeg helpers
├── examples/               # Example ClawFlow YAML pipelines
├── tests/                  # Unit + integration tests
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
git clone https://github.com/moose-lab/videoclaw.git
cd videoclaw
uv sync --all-extras          # Install all deps including dev/server
# or: make dev

uv run pytest tests/ -v       # Run tests
uv run ruff check src/ tests/ # Lint
# or: make test / make lint
```

## Roadmap

- [x] **Phase 1**: Core engine, DAG executor, model adapters, CLI, cost tracking
- [x] **Phase 2**: FastAPI server, WebSocket, storage, publishers, test suite
- [x] **Phase 3**: ClawFlow YAML engine, integration tests, Docker
- [x] **Phase 4**: Director LLM integration, GitHub Actions CI, flow templates
- [ ] **Phase 5**: AI Short Drama orchestration + CLI-driven workflows
- [ ] **Phase 6**: Plugin marketplace (ClawHub) + industry templates
- [ ] **Phase 7**: Multi-agent collaboration + quality review loop

## License

Modified MIT — see [LICENSE](LICENSE) for details.
