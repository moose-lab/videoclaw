<div align="center">
  <h1>🎬 VideoClaw</h1>
  <p><strong>The Agent OS for AI Video Generation</strong></p>
  <p>
    Orchestrate multiple AI models. Automate entire video pipelines.<br/>
    From script to publish — one command, one flow, zero babysitting.
  </p>
  <p>
    <a href="#quick-start">Quick Start</a> •
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#supported-models">Models</a> •
    <a href="#contributing">Contributing</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License" />
    <img src="https://img.shields.io/badge/python-3.12+-green" alt="Python" />
    <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Docker-lightgrey" alt="Platform" />
    <img src="https://img.shields.io/badge/status-Phase%201-orange" alt="Status" />
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

# → Script generated (Claude) ✓
# → 6 shots planned ✓
# → Shots generating... ████████ Done
# → Voice: Edge-TTS ████████ Done
# → Composed & rendered ✓
# → output/smart_watch_intro.mp4 (Cost: $0.47)
```

## Features

### Multi-Model Orchestration
One pipeline, multiple models. VideoClaw picks the best model for each shot based on your strategy — quality, speed, or cost.

```
Same 30s video:
  All Sora:              $2.50  ~3 min
  VideoClaw hybrid:      $0.47  ~2 min   ← auto-routes simple shots locally
  VideoClaw all-local:   $0.00  ~6 min
```

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
  You ──▶ Director Agent ──▶ Planner ──▶ DAG Executor
              │                              │
              ▼                     ┌────────┼────────┐
         Scriptwriter              ▼        ▼        ▼
         Cameraman              [Sora]  [CogVideo] [Mock]
         Reviewer                  │        │        │
              │                    └────────┼────────┘
              ▼                             ▼
         Quality Gate ──▶ Compose ──▶ Output
```

## Supported Models

| Category | Models | Mode |
|----------|--------|------|
| Video | Sora (OpenAI), Runway, Kling, CogVideoX, Wan2.2, AnimateDiff | Cloud + Local |
| LLM | Claude, GPT, Qwen, DeepSeek, Ollama (via LiteLLM) | Cloud + Local |
| TTS | Edge-TTS, Fish-Speech, ElevenLabs, ChatTTS | Cloud + Local |
| Music | Suno, Udio, MusicGen | Cloud + Local |

> Adding a new model? Implement 4 async methods. See [Adapter Guide](docs/adapter-guide.md).

## CLI Commands

```bash
claw generate <prompt>      # Full pipeline: script → shots → compose → output
claw doctor                 # System health check
claw model list             # List available model adapters
claw model pull <id>        # Download a local model
claw project list           # List all projects
claw project show <id>      # Show project details + cost
claw template list          # List flow templates
claw template use <name>    # Generate from template
```

## Project Structure

```
videoclaw/
├── src/videoclaw/
│   ├── cli.py              # CLI entry point (Typer)
│   ├── config.py           # Configuration (Pydantic)
│   ├── core/               # DAG engine, state, events
│   ├── agents/             # Video Agent definitions
│   ├── models/             # Model adapters + router
│   ├── generation/         # Script, storyboard, video, audio, compose
│   ├── cost/               # Cost tracking engine
│   └── utils/              # FFmpeg helpers
├── templates/              # Flow templates (YAML)
├── tests/
└── docs/
```

## Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For Sora/GPT | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Claude | Anthropic API key |
| `VIDEOCLAW_DEFAULT_LLM` | No | Default LLM (default: gpt-4o) |
| `VIDEOCLAW_DEFAULT_VIDEO_MODEL` | No | Default video model (default: mock) |

## Roadmap

- [x] **Phase 1**: Core engine + CLI + Mock/Sora adapters + Cost tracking
- [ ] **Phase 2**: Web UI (ClawFlow visual editor) + More adapters
- [ ] **Phase 3**: Plugin marketplace (ClawHub) + Industry templates
- [ ] **Phase 4**: Multi-agent collaboration + Quality review loop

## Contributing

We welcome contributions! Whether it's a new model adapter, flow template, or bug fix.

```bash
git clone https://github.com/your-org/videoclaw.git
cd videoclaw
make dev
make test
```

## License

Apache 2.0 — use it for anything.
