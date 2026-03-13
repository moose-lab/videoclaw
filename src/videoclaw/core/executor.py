"""Async DAG executor -- runs tasks respecting dependencies and concurrency limits."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from videoclaw.config import get_config
from videoclaw.core.events import (
    EventBus,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_STARTED,
    PROJECT_COMPLETED,
    event_bus as default_event_bus,
)
from videoclaw.core.planner import DAG, NodeStatus, TaskNode, TaskType
from videoclaw.core.state import ProjectState, ProjectStatus, ShotStatus, StateManager

logger = logging.getLogger(__name__)

# Type alias for an async handler that processes a single task node.
NodeHandler = Callable[[TaskNode, ProjectState], Coroutine[Any, Any, Any]]


class DAGExecutor:
    """Execute a :class:`DAG` asynchronously, honouring dependency order.

    Independent nodes run in parallel up to *max_concurrency*.  On completion
    of each node the project state is checkpointed to disk.
    """

    def __init__(
        self,
        dag: DAG,
        state: ProjectState,
        state_manager: StateManager | None = None,
        bus: EventBus | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.dag = dag
        self.state = state
        self.state_manager = state_manager or StateManager()
        self.bus = bus or default_event_bus
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._config = get_config()

        # Handler dispatch table -- maps TaskType to its async handler.
        # During Phase 1 every entry points at a placeholder.  Later phases
        # will register real generation / composition handlers.
        self._handlers: dict[TaskType, NodeHandler] = {
            TaskType.SCRIPT_GEN: self._handle_script_gen,
            TaskType.STORYBOARD: self._handle_storyboard,
            TaskType.VIDEO_GEN: self._handle_video_gen,
            TaskType.TTS: self._handle_tts,
            TaskType.MUSIC: self._handle_music,
            TaskType.COMPOSE: self._handle_compose,
            TaskType.RENDER: self._handle_render,
        }

    def register_handler(self, task_type: TaskType, handler: NodeHandler) -> None:
        """Override the handler for a specific task type."""
        self._handlers[task_type] = handler

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> ProjectState:
        """Execute the full DAG and return the final project state."""
        self.state.status = ProjectStatus.GENERATING
        self._checkpoint()

        while not self.dag.is_complete:
            ready = self.dag.get_ready_nodes()
            if not ready:
                # Safety valve: if no nodes are ready and the DAG isn't complete
                # it means all remaining nodes are blocked by failures.
                if not any(
                    n.status == NodeStatus.RUNNING for n in self.dag.nodes.values()
                ):
                    logger.error("DAG stalled -- remaining nodes blocked by failures")
                    break
                # Some nodes are still running; wait briefly and re-check.
                await asyncio.sleep(0.05)
                continue

            tasks = [self._run_node(node) for node in ready]
            await asyncio.gather(*tasks)

        # Final status
        if self.dag.has_failures:
            self.state.status = ProjectStatus.FAILED
        else:
            self.state.status = ProjectStatus.COMPLETED
            await self.bus.emit(PROJECT_COMPLETED, {"project_id": self.state.project_id})

        self._checkpoint()
        logger.info(
            "Project %s finished with status %s",
            self.state.project_id,
            self.state.status.value,
        )
        return self.state

    # ------------------------------------------------------------------
    # Per-node execution
    # ------------------------------------------------------------------

    async def _run_node(self, node: TaskNode) -> None:
        """Acquire the semaphore, then execute and finalise a single node."""
        async with self._semaphore:
            await self._execute_node(node)

    async def _execute_node(self, node: TaskNode) -> None:
        """Dispatch *node* to its handler, with retry logic."""
        handler = self._handlers.get(node.task_type)
        if handler is None:
            self.dag.mark_failed(node.node_id, f"No handler for {node.task_type}")
            await self.bus.emit(TASK_FAILED, {
                "node_id": node.node_id,
                "error": f"No handler for {node.task_type}",
            })
            self._checkpoint()
            return

        self.dag.mark_running(node.node_id)
        await self.bus.emit(TASK_STARTED, {
            "node_id": node.node_id,
            "task_type": node.task_type.value,
        })

        max_attempts = self._config.max_retries + 1
        last_error: str = ""

        for attempt in range(1, max_attempts + 1):
            try:
                result = await handler(node, self.state)
                self.dag.mark_complete(node.node_id, result)
                await self.bus.emit(TASK_COMPLETED, {
                    "node_id": node.node_id,
                    "task_type": node.task_type.value,
                    "result": result,
                })
                self._checkpoint()
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Node %s attempt %d/%d failed: %s",
                    node.node_id,
                    attempt,
                    max_attempts,
                    last_error,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(0.1 * attempt)  # simple back-off

        # Exhausted retries
        self.dag.mark_failed(node.node_id, last_error)
        await self.bus.emit(TASK_FAILED, {
            "node_id": node.node_id,
            "task_type": node.task_type.value,
            "error": last_error,
        })
        self._checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Persist current project state to disk."""
        try:
            self.state_manager.save(self.state)
        except Exception:
            logger.exception("Failed to checkpoint state for %s", self.state.project_id)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    # Each handler receives the TaskNode and the current ProjectState.
    # Handlers are "smart": they detect pre-populated state (e.g. from
    # DramaPlanner) and skip generation when data already exists.

    async def _handle_script_gen(self, node: TaskNode, state: ProjectState) -> Any:
        """Generate script via LLM, or skip if already populated."""
        if state.script:
            logger.info("[script_gen] Script already exists, skipping LLM call")
            return {"script": state.script}

        from videoclaw.generation.script import ScriptGenerator

        generator = ScriptGenerator()
        language = state.metadata.get("language", "zh")
        style = state.metadata.get("style", "cinematic")
        duration = (
            sum(s.duration_seconds for s in state.storyboard)
            if state.storyboard
            else 60.0
        )

        script = await generator.generate(
            topic=state.prompt,
            duration=duration,
            tone=style,
            language=language,
        )
        state.script = script.voice_over_text
        logger.info(
            "[script_gen] Generated script: %d sections, %.1fs",
            len(script.sections),
            script.total_duration,
        )
        return {
            "script": state.script,
            "title": script.title,
            "sections": len(script.sections),
        }

    async def _handle_storyboard(self, node: TaskNode, state: ProjectState) -> Any:
        """Decompose script into shots via LLM, or skip if already populated."""
        if state.storyboard:
            logger.info(
                "[storyboard] Storyboard already populated (%d shots), skipping",
                len(state.storyboard),
            )
            return {"shot_count": len(state.storyboard)}

        from videoclaw.generation.script import Script, ScriptSection
        from videoclaw.generation.storyboard import StoryboardGenerator

        # Build a Script object from state for the storyboard generator
        script = Script(
            title=state.prompt,
            voice_over_text=state.script or "",
            sections=[
                ScriptSection(text=state.script or "", duration_seconds=60.0)
            ],
            total_duration=60.0,
        )

        generator = StoryboardGenerator()
        aspect_ratio = state.metadata.get("aspect_ratio", "16:9")
        style = state.metadata.get("style", "cinematic")

        shots = await generator.decompose(script, style=style, aspect_ratio=aspect_ratio)
        state.storyboard = shots
        logger.info("[storyboard] Generated %d shots", len(shots))
        return {"shot_count": len(shots)}

    async def _handle_video_gen(self, node: TaskNode, state: ProjectState) -> Any:
        """Generate video for a single shot using VideoGenerator."""
        from pathlib import Path

        from videoclaw.generation.video import VideoGenerator
        from videoclaw.models.registry import get_registry
        from videoclaw.models.router import ModelRouter, RoutingStrategy

        shot_id = node.params.get("shot_id", "unknown")
        logger.info("[video_gen] Generating video for shot %s", shot_id)

        shot = next((s for s in state.storyboard if s.shot_id == shot_id), None)
        if not shot:
            raise ValueError(f"Shot {shot_id} not found in storyboard")

        # Load character reference images from params
        reference_images: dict[str, str] = node.params.get("reference_images", {})
        speaking_character: str = node.params.get("speaking_character", "")

        primary_ref_bytes: bytes | None = None
        extra_refs: dict[str, bytes] = {}

        if reference_images:
            # Determine primary character: speaking_character first, else first available
            primary_name = speaking_character if speaking_character in reference_images else None
            if primary_name is None:
                primary_name = next(iter(reference_images))

            for char_name, img_path in reference_images.items():
                img_file = Path(img_path)
                if not img_file.exists():
                    logger.warning(
                        "[video_gen] Reference image not found for %s: %s",
                        char_name,
                        img_path,
                    )
                    continue
                img_bytes = img_file.read_bytes()
                if char_name == primary_name:
                    primary_ref_bytes = img_bytes
                else:
                    extra_refs[char_name] = img_bytes

        registry = get_registry()
        registry.discover()

        router = ModelRouter(registry)
        generator = VideoGenerator(router=router)

        result = await generator.generate_shot(
            shot,
            strategy=RoutingStrategy.AUTO,
            aspect_ratio=node.params.get("aspect_ratio", state.metadata.get("aspect_ratio")),
            reference_image=primary_ref_bytes,
            extra_references=extra_refs if extra_refs else None,
        )

        # Update shot status
        shot.cost = result.cost_usd
        shot.status = ShotStatus.COMPLETED if result.video_data else ShotStatus.FAILED

        # Persist video data
        if result.video_data:
            import hashlib

            video_hash = hashlib.md5(result.video_data).hexdigest()[:8]
            output_dir = Path(self._config.projects_dir) / state.project_id / "shots"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{shot_id}_{video_hash}.mp4"
            output_path.write_bytes(result.video_data)
            shot.asset_path = str(output_path)
            logger.info("[video_gen] Saved video to %s", output_path)

        return {
            "asset_path": shot.asset_path,
            "cost_usd": result.cost_usd,
            "model_id": result.model_id,
        }

    async def _handle_tts(self, node: TaskNode, state: ProjectState) -> Any:
        """Synthesize dialogue/narration audio per scene via TTSManager."""
        import json as _json
        from pathlib import Path

        from videoclaw.generation.audio.tts import TTSManager

        tts = TTSManager()
        language = node.params.get("language", state.metadata.get("language", "zh"))
        project_dir = Path(self._config.projects_dir) / state.project_id
        audio_dir = project_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_paths: list[dict[str, str]] = []
        scenes = node.params.get("scenes", [])

        if scenes:
            # Drama mode: per-scene dialogue + narration
            for scene_data in scenes:
                scene_id = scene_data.get("scene_id", "unknown")
                dialogue = scene_data.get("dialogue", "").strip()
                narration = scene_data.get("narration", "").strip()
                voice = scene_data.get("voice")

                if dialogue:
                    path = audio_dir / f"{scene_id}_dialogue.mp3"
                    await tts.generate_voiceover(
                        dialogue, path, voice=voice, language=language,
                    )
                    audio_paths.append({
                        "scene_id": scene_id,
                        "type": "dialogue",
                        "path": str(path),
                    })

                if narration:
                    path = audio_dir / f"{scene_id}_narration.mp3"
                    await tts.generate_voiceover(
                        narration, path, language=language,
                    )
                    audio_paths.append({
                        "scene_id": scene_id,
                        "type": "narration",
                        "path": str(path),
                    })
        elif state.script:
            # Generic mode: single voiceover from full script
            path = audio_dir / "voiceover.mp3"
            await tts.generate_voiceover(state.script, path, language=language)
            audio_paths.append({"type": "voiceover", "path": str(path)})

        state.assets["tts_audio"] = _json.dumps(audio_paths)
        logger.info("[tts] Synthesized %d audio segments", len(audio_paths))
        return {"audio_paths": audio_paths, "count": len(audio_paths)}

    async def _handle_music(self, node: TaskNode, state: ProjectState) -> Any:
        """Background music generation (placeholder — no music API integrated yet)."""
        logger.info("[music] No music API configured, skipping BGM generation")
        state.assets["music"] = ""
        return {"status": "skipped", "reason": "no_music_api"}

    async def _handle_compose(self, node: TaskNode, state: ProjectState) -> Any:
        """Compose video clips + audio + subtitles into a single timeline."""
        import json as _json
        from pathlib import Path

        from videoclaw.generation.compose import AudioTrack, AudioType, VideoComposer
        from videoclaw.generation.subtitle import generate_srt

        project_dir = Path(self._config.projects_dir) / state.project_id
        composer = VideoComposer()

        # 1. Collect video paths from completed shots
        video_paths: list[Path] = []
        for shot in state.storyboard:
            if shot.asset_path and Path(shot.asset_path).exists():
                video_paths.append(Path(shot.asset_path))

        if not video_paths:
            raise ValueError("No video assets available for composition")

        # 2. Concatenate videos with transitions
        composed_path = project_dir / "composed.mp4"
        transition = node.params.get("transition", "dissolve")

        # Extract per-scene transitions when available
        per_scene_transitions: list[str] | None = None
        scenes = node.params.get("scenes", [])
        if scenes:
            per_scene_transitions = [
                s.get("transition", "") or "" for s in scenes
            ]

        await composer.compose(
            video_paths,
            composed_path,
            transition=transition,
            transitions=per_scene_transitions,
        )
        logger.info("[compose] Composed %d clips -> %s", len(video_paths), composed_path)

        # 3. Generate subtitles from scene dialogue
        scenes = node.params.get("scenes", [])
        subtitle_path = None
        if scenes and any(s.get("dialogue") for s in scenes):
            subtitle_path = project_dir / "subtitles.srt"
            generate_srt(scenes, subtitle_path)
            state.assets["subtitles"] = str(subtitle_path)
            logger.info("[compose] Generated subtitles -> %s", subtitle_path)

        # 4. Collect audio tracks (TTS + music)
        audio_tracks: list[AudioTrack] = []
        tts_data = state.assets.get("tts_audio")
        if tts_data:
            for entry in _json.loads(tts_data):
                audio_path = Path(entry["path"])
                if audio_path.exists():
                    audio_tracks.append(AudioTrack(
                        path=audio_path,
                        type=AudioType.VOICE,
                        volume=0.9,
                    ))

        # 5. Final render: audio mix + subtitle burn
        output_path = project_dir / "composed_final.mp4"
        if audio_tracks or subtitle_path:
            await composer.render_final(
                video_path=composed_path,
                audio_tracks=audio_tracks,
                subtitle_path=subtitle_path,
                output_path=output_path,
            )
        else:
            import shutil
            shutil.copy2(composed_path, output_path)

        state.assets["composed_video"] = str(output_path)
        logger.info("[compose] Final composed video -> %s", output_path)
        return {"composed_path": str(output_path)}

    async def _handle_render(self, node: TaskNode, state: ProjectState) -> Any:
        """Produce the final deliverable video file via FFmpeg render pipeline.

        Falls back to a simple file copy if FFmpeg encoding fails.
        """
        import shutil
        from pathlib import Path

        from videoclaw.generation.render import VideoRenderer, _ASPECT_TO_RENDER_RESOLUTION

        project_dir = Path(self._config.projects_dir) / state.project_id
        composed = state.assets.get("composed_video")
        if not composed or not Path(composed).exists():
            raise ValueError("No composed video found — compose step may have failed")

        output_path = project_dir / "final.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve render parameters from node.params and state.metadata
        aspect_ratio = state.metadata.get("aspect_ratio")
        resolution = _ASPECT_TO_RENDER_RESOLUTION.get(aspect_ratio) if aspect_ratio else None

        codec = node.params.get("codec", "libx264")
        preset = node.params.get("preset", "medium")
        crf = node.params.get("crf", 23)
        bitrate = node.params.get("bitrate", "8M")
        audio_bitrate = node.params.get("audio_bitrate", "192k")

        # Build metadata from state
        render_metadata: dict[str, str] = {}
        if state.metadata.get("series_id"):
            render_metadata["title"] = state.prompt
        if state.metadata.get("episode_number"):
            render_metadata["episode_id"] = str(state.metadata["episode_number"])

        try:
            renderer = VideoRenderer()
            await renderer.render(
                input_path=Path(composed),
                output_path=output_path,
                resolution=resolution,
                bitrate=bitrate,
                audio_bitrate=audio_bitrate,
                codec=codec,
                preset=preset,
                crf=crf,
                metadata=render_metadata if render_metadata else None,
            )
        except Exception as exc:
            logger.warning(
                "[render] FFmpeg render failed (%s), falling back to file copy", exc,
            )
            shutil.copy2(composed, output_path)

        state.assets["final_video"] = str(output_path)
        logger.info("[render] Final video -> %s", output_path)
        return {"output_path": str(output_path)}
