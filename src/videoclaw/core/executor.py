"""Async DAG executor -- runs tasks respecting dependencies and concurrency limits."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from videoclaw.config import get_config
from videoclaw.core.events import (
    PROJECT_COMPLETED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_STARTED,
    EventBus,
)
from videoclaw.core.events import (
    event_bus as default_event_bus,
)
from videoclaw.core.planner import DAG, NodeStatus, TaskNode, TaskType
from videoclaw.core.state import ProjectState, ProjectStatus, ShotStatus, StateManager

if TYPE_CHECKING:
    from videoclaw.cost.tracker import CostTracker

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
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self.dag = dag
        self.state = state
        self.state_manager = state_manager or StateManager()
        self.bus = bus or default_event_bus
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._config = get_config()
        self.cost_tracker = cost_tracker

        # Handler dispatch table -- maps TaskType to its async handler.
        # During Phase 1 every entry points at a placeholder.  Later phases
        # will register real generation / composition handlers.
        self._handlers: dict[TaskType, NodeHandler] = {
            TaskType.SCRIPT_GEN: self._handle_script_gen,
            TaskType.STORYBOARD: self._handle_storyboard,
            TaskType.SCENE_VALIDATE: self._handle_scene_validate,
            TaskType.VIDEO_GEN: self._handle_video_gen,
            TaskType.TTS: self._handle_tts,
            TaskType.PER_SCENE_TTS: self._handle_per_scene_tts,
            TaskType.SUBTITLE_GEN: self._handle_subtitle_gen,
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

        # Persist cost ledger to disk alongside project state.
        if self.cost_tracker is not None:
            cost_path = (
                Path(self._config.projects_dir)
                / self.state.project_id
                / "cost.json"
            )
            try:
                self.cost_tracker.save_ledger(cost_path)
            except Exception:
                logger.exception("Failed to save cost ledger for %s", self.state.project_id)

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
        """Dispatch *node* to its handler, with retry logic and cost tracking."""
        import time as _time

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
        t0 = _time.monotonic()
        retries = 0

        for attempt in range(1, max_attempts + 1):
            try:
                result = await handler(node, self.state)
                duration = _time.monotonic() - t0
                self._record_cost(node, result, duration, retries)
                self.dag.mark_complete(node.node_id, result)
                await self.bus.emit(TASK_COMPLETED, {
                    "node_id": node.node_id,
                    "task_type": node.task_type.value,
                    "result": result,
                })
                self._checkpoint()
                return
            except Exception as exc:
                retries += 1
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
    # Cost tracking
    # ------------------------------------------------------------------

    def _record_cost(
        self,
        node: TaskNode,
        result: Any,
        duration_seconds: float,
        retries: int,
    ) -> None:
        """Build a :class:`CostRecord` from a completed task and record it."""
        if self.cost_tracker is None:
            return

        from videoclaw.cost.tracker import CostRecord

        # Extract cost from handler result (dict with cost_usd key)
        cost_usd: float = 0.0
        model_id: str = "unknown"
        video_seconds: float = 0.0

        if isinstance(result, dict):
            cost_usd = float(result.get("cost_usd", 0.0))
            model_id = result.get("model_id", node.params.get("model_id", "unknown"))
            video_seconds = float(result.get("video_seconds", 0.0))

        # Map TaskType to task_type string for bucketing
        task_type_map = {
            TaskType.VIDEO_GEN: "video_gen",
            TaskType.TTS: "tts",
            TaskType.PER_SCENE_TTS: "tts",
            TaskType.SCRIPT_GEN: "llm",
            TaskType.STORYBOARD: "llm",
            TaskType.SCENE_VALIDATE: "validate",
            TaskType.SUBTITLE_GEN: "subtitle",
            TaskType.MUSIC: "music",
            TaskType.COMPOSE: "compose",
            TaskType.RENDER: "render",
        }

        record = CostRecord(
            task_id=node.node_id,
            model_id=model_id,
            execution_mode="cloud" if cost_usd > 0 else "local",
            api_cost_usd=cost_usd,
            compute_cost_usd=0.0,
            duration_seconds=duration_seconds,
            task_type=task_type_map.get(node.task_type, node.task_type.value),
            video_seconds=video_seconds,
            retries=retries,
        )
        self.cost_tracker.record(record)

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

    async def _handle_scene_validate(self, node: TaskNode, state: ProjectState) -> Any:
        """Validate scene data before committing to expensive generation.

        Runs locale-aware quality checks on scene-level data:
        - Emotion vocabulary presence
        - Speaker/character consistency
        - Shot scale distribution (vertical framing)
        - Subtitle readability (dialogue word count vs duration)
        - Structural role validation (hook/cliffhanger placement)
        - Time-of-day coverage
        - Duration bounds (Seedance 5-15s constraint)

        Logs violations as warnings but does not block the pipeline
        (lenient mode).
        """
        scenes = node.params.get("scenes", [])
        language = node.params.get("language", state.metadata.get("language", "zh"))

        if not scenes:
            logger.info("[scene_validate] No scenes to validate, skipping")
            return {"status": "skipped", "violations": []}

        # Per-scene quick checks
        violations: list[str] = []
        warnings: list[str] = []
        total_close = 0
        total_scenes = len(scenes)

        # Subtitle readability limits
        # English: ≤8 words per subtitle segment (readable at 9:16 720p)
        # Chinese: ≤16 characters per subtitle line
        max_subtitle_words_en = 8
        max_subtitle_chars_zh = 16

        for idx, scene in enumerate(scenes):
            scene_id = scene.get("scene_id", "?")

            # --- Emotion field required ---
            if not scene.get("emotion"):
                violations.append(f"Scene {scene_id}: missing emotion field")

            # --- speaking_character consistency ---
            speaker = scene.get("speaking_character", "")
            present = scene.get("characters_present", [])
            if speaker and present and speaker not in present:
                violations.append(
                    f"Scene {scene_id}: speaker '{speaker}' not in characters_present"
                )

            # --- Track shot scale distribution ---
            if scene.get("shot_scale") in ("close_up", "medium_close"):
                total_close += 1

            # --- Duration bounds (Seedance 5-15s) ---
            dur = float(scene.get("duration_seconds", 0))
            if dur > 0 and (dur < 5 or dur > 15):
                warnings.append(
                    f"Scene {scene_id}: duration {dur:.1f}s outside Seedance range [5-15s]"
                )

            # --- Subtitle readability ---
            dialogue = (scene.get("dialogue") or "").strip()
            if dialogue and dur > 0:
                if language == "zh":
                    char_count = len(dialogue)
                    if char_count > max_subtitle_chars_zh:
                        warnings.append(
                            f"Scene {scene_id}: dialogue"
                            f" {char_count} chars"
                            f" > {max_subtitle_chars_zh}"
                            " char subtitle limit"
                            " — may be hard to read on 9:16"
                        )
                else:
                    word_count = len(dialogue.split())
                    if word_count > max_subtitle_words_en:
                        warnings.append(
                            f"Scene {scene_id}: dialogue"
                            f" {word_count} words"
                            f" > {max_subtitle_words_en}"
                            " word subtitle limit"
                            " — consider splitting"
                        )

            # --- Time-of-day coverage ---
            if not scene.get("time_of_day"):
                warnings.append(
                    f"Scene {scene_id}: missing time_of_day — "
                    f"lighting may be inconsistent"
                )

            # --- Structural role validation ---
            shot_role = scene.get("shot_role", "normal")
            if idx == 0 and shot_role != "hook":
                warnings.append(
                    f"Scene {scene_id}: first scene should have shot_role='hook' "
                    f"(has '{shot_role}')"
                )
            if idx == total_scenes - 1 and shot_role != "cliffhanger":
                warnings.append(
                    f"Scene {scene_id}: last scene should have shot_role='cliffhanger' "
                    f"(has '{shot_role}')"
                )

        # Vertical framing ratio (close_up + medium_close >= 50%)
        if total_scenes > 0:
            ratio = total_close / total_scenes
            if ratio < 0.5:
                violations.append(
                    f"Vertical framing: {ratio:.0%} close shots < 50% minimum"
                )

        all_issues = violations + warnings
        if all_issues:
            logger.warning(
                "[scene_validate] %d violation(s), %d warning(s):",
                len(violations), len(warnings),
            )
            for issue in all_issues:
                logger.warning("  • %s", issue)
        else:
            logger.info("[scene_validate] All %d scenes passed validation", total_scenes)

        return {
            "status": "ok",
            "violations": violations,
            "warnings": warnings,
            "scene_count": total_scenes,
        }

    async def _handle_video_gen(self, node: TaskNode, state: ProjectState) -> Any:
        """Generate video for a single shot using VideoGenerator."""
        from videoclaw.generation.video import VideoGenerator
        from videoclaw.models.registry import get_registry
        from videoclaw.models.router import ModelRouter, RoutingStrategy

        shot_id = node.params.get("shot_id", "unknown")
        logger.info("[video_gen] Generating video for shot %s", shot_id)

        shot = next((s for s in state.storyboard if s.shot_id == shot_id), None)
        if not shot:
            raise ValueError(f"Shot {shot_id} not found in storyboard")

        # Build image_urls list for Seedance Universal Reference (全能参考).
        # Priority: HTTPS URLs (vectorspace.cn proxy rejects base64 data URIs).
        # Fallback: local file paths (converted to base64 by the adapter).
        reference_image_urls: dict[str, str] = node.params.get("reference_image_urls", {})
        reference_images: dict[str, str] = node.params.get("reference_images", {})

        extra: dict[str, Any] = {}
        image_urls: list[dict[str, str]] = []

        if reference_image_urls:
            # Preferred: HTTPS URLs for character turnaround sheets
            for char_name, url in reference_image_urls.items():
                if url and url.startswith("http"):
                    image_urls.append({"url": url, "role": "reference_image"})
            logger.info(
                "[video_gen] Passing %d character HTTPS URLs as reference for shot %s",
                len(image_urls), shot_id,
            )
        elif reference_images:
            # Fallback: local file paths (will be base64-encoded by adapter)
            image_paths = [
                {"path": path, "role": "reference_image"}
                for path in reference_images.values()
                if path
            ]
            if image_paths:
                extra["image_paths"] = image_paths
                logger.info(
                    "[video_gen] Passing %d local image paths as reference for shot %s",
                    len(image_paths), shot_id,
                )

        # Scene reference URLs (from ConsistencyManifest)
        scene_ref_urls: dict[str, str] = node.params.get("scene_reference_urls", {})
        for loc_key, url in scene_ref_urls.items():
            if url and url.startswith("http"):
                image_urls.append({"url": url, "role": "reference_image"})
            elif url:
                if "image_paths" not in extra:
                    extra["image_paths"] = []
                extra["image_paths"].append({"path": url, "role": "reference_image"})

        # Prop reference URLs (from ConsistencyManifest)
        prop_ref_urls: dict[str, str] = node.params.get("prop_reference_urls", {})
        for prop_key, url in prop_ref_urls.items():
            if url and url.startswith("http"):
                image_urls.append({"url": url, "role": "reference_image"})
            elif url:
                if "image_paths" not in extra:
                    extra["image_paths"] = []
                extra["image_paths"].append({"path": url, "role": "reference_image"})

        logger.info(
            "[video_gen] Total reference images for shot %s: %d URLs + %d paths",
            shot_id, len(image_urls), len(extra.get("image_paths", [])),
        )

        if image_urls:
            extra["image_urls"] = image_urls

        # --- Build structured segments for interleaved content (if ref markers present) ---
        if "[ref:" in shot.prompt:
            from videoclaw.drama.prompt_segments import (
                PromptSegmenter,
                ReferenceMedia,
                allocate_reference_slots,
            )
            from videoclaw.drama.prompt_enhancer import _to_ref_key

            # Build available refs for slot allocation
            all_available: dict[str, dict[str, str]] = {
                "characters": {},
                "scenes": node.params.get("scene_reference_urls", {}),
                "props": node.params.get("prop_reference_urls", {}),
            }
            # Character refs: prefer URLs, fallback to paths
            for char_name, url in reference_image_urls.items():
                all_available["characters"][char_name] = url
            if not all_available["characters"]:
                for char_name, path in reference_images.items():
                    all_available["characters"][char_name] = path

            # Allocate slots by shot type
            shot_scale = None
            try:
                from videoclaw.drama.models import ShotScale as _SS
                shot_scale_raw = node.params.get("shot_scale")
                if shot_scale_raw:
                    shot_scale = _SS(shot_scale_raw)
            except (ValueError, KeyError):
                pass

            speaking_char = node.params.get("speaking_character")
            allocated = allocate_reference_slots(
                shot_scale, all_available, speaking_character=speaking_char,
            )

            # Build ref_map for segmenter (key → ReferenceMedia)
            ref_map = {_to_ref_key(r.key): r for r in allocated}

            # Parse prompt into segments
            segments = PromptSegmenter.parse(shot.prompt, ref_map)
            extra["prompt_segments"] = segments
            # Clear old-style refs since segments handle everything
            extra.pop("image_urls", None)
            extra.pop("image_paths", None)
            logger.info(
                "[video_gen] Using structured segments for shot %s (%d segments, %d refs)",
                shot_id, len(segments), len(allocated),
            )

        registry = get_registry()
        registry.discover()

        router = ModelRouter(registry)
        generator = VideoGenerator(router=router)

        result = await generator.generate_shot(
            shot,
            strategy=RoutingStrategy.AUTO,
            aspect_ratio=node.params.get("aspect_ratio", state.metadata.get("aspect_ratio")),
            extra=extra if extra else None,
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
            # Session prefix for traceability across iterations
            session_tag = state.metadata.get("session", "")
            prefix = f"session{session_tag}_" if session_tag else ""
            output_path = output_dir / f"{prefix}{shot_id}_{video_hash}.mp4"
            output_path.write_bytes(result.video_data)
            shot.asset_path = str(output_path)
            logger.info("[video_gen] Saved video to %s", output_path)

        return {
            "asset_path": shot.asset_path,
            "cost_usd": result.cost_usd,
            "model_id": result.model_id,
        }

    async def _handle_tts(self, node: TaskNode, state: ProjectState) -> Any:
        """Synthesize dialogue/narration audio per scene via TTSManager.

        In drama mode (when ``scenes`` is present in node params), this builds
        :class:`DialogueLine` objects and delegates to
        :meth:`TTSManager.generate_multi_role` for per-character voice routing.
        An :class:`EpisodeAudioManifest` is built and stored in
        ``state.assets["audio_manifest"]``.

        In generic mode (no scenes), falls back to a single
        ``generate_voiceover()`` call for backward compatibility.
        """
        import json as _json
        from pathlib import Path

        from videoclaw.generation.audio.tts import TTSManager

        tts = TTSManager()
        language = node.params.get("language", state.metadata.get("language", "zh"))
        project_dir = Path(self._config.projects_dir) / state.project_id
        audio_dir = project_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        scenes = node.params.get("scenes", [])

        if scenes:
            # --- Drama mode: multi-role voice pipeline ---
            from videoclaw.drama.models import (
                AudioSegment as DramaAudioSegment,
            )
            from videoclaw.drama.models import (
                DialogueLine,
                EpisodeAudioManifest,
                LineType,
                VoiceProfile,
            )

            # 1. Reconstruct voice_map from node.params or state.metadata
            raw_voice_map: dict[str, dict] = (
                node.params.get("voice_map")
                or state.metadata.get("voice_map")
                or {}
            )
            voice_map: dict[str, VoiceProfile] = {
                name: VoiceProfile.from_dict(vp_data)
                for name, vp_data in raw_voice_map.items()
            }

            # 2. Build DialogueLine list from scenes_data
            lines: list[DialogueLine] = []
            for scene_data in scenes:
                scene_id = scene_data.get("scene_id", "unknown")
                dialogue = scene_data.get("dialogue", "").strip()
                narration = scene_data.get("narration", "").strip()
                speaking_character = scene_data.get("speaking_character", "")
                emotion = scene_data.get("emotion", "")
                dialogue_line_type_str = scene_data.get(
                    "dialogue_line_type", "dialogue",
                )

                # Map dialogue_line_type string to LineType enum
                try:
                    dialogue_line_type = LineType(dialogue_line_type_str)
                except ValueError:
                    dialogue_line_type = LineType.DIALOGUE

                if dialogue:
                    lines.append(DialogueLine(
                        text=dialogue,
                        speaker=speaking_character or "narrator",
                        line_type=dialogue_line_type,
                        scene_id=scene_id,
                        emotion_hint=emotion or None,
                    ))

                narration_type = scene_data.get("narration_type", "voiceover")
                if narration and narration_type != "title_card":
                    lines.append(DialogueLine(
                        text=narration,
                        speaker="narrator",
                        line_type=LineType.NARRATION,
                        scene_id=scene_id,
                        emotion_hint=None,
                    ))

            # 3. Call generate_multi_role
            segments: list[DramaAudioSegment] = await tts.generate_multi_role(
                lines, voice_map, audio_dir, language=language,
            )

            # 4. Build EpisodeAudioManifest
            total_duration = sum(s.duration_seconds for s in segments)
            manifest = EpisodeAudioManifest(
                episode_id=state.metadata.get("episode_id", ""),
                segments=segments,
                total_duration=total_duration,
            )
            state.assets["audio_manifest"] = _json.dumps(manifest.to_dict())

            # 5. Backward compat: populate tts_audio for compose handler
            audio_paths: list[dict[str, str]] = []
            for seg in segments:
                audio_paths.append({
                    "scene_id": seg.scene_id,
                    "type": seg.audio_type.value,
                    "path": seg.audio_path or "",
                })
            state.assets["tts_audio"] = _json.dumps(audio_paths)

            logger.info(
                "[tts] Multi-role: synthesized %d segments from %d lines",
                len(segments),
                len(lines),
            )
            return {
                "audio_paths": audio_paths,
                "count": len(segments),
                "manifest_segments": len(segments),
            }
        elif state.script:
            # --- Generic mode: single voiceover from full script ---
            path = audio_dir / "voiceover.mp3"
            await tts.generate_voiceover(state.script, path, language=language)
            audio_paths: list[dict[str, str]] = [
                {"type": "voiceover", "path": str(path)},
            ]
            state.assets["tts_audio"] = _json.dumps(audio_paths)
            logger.info("[tts] Synthesized 1 generic voiceover segment")
            return {"audio_paths": audio_paths, "count": 1}

        # No scenes and no script -- nothing to synthesize
        state.assets["tts_audio"] = _json.dumps([])
        logger.info("[tts] No content to synthesize")
        return {"audio_paths": [], "count": 0}

    async def _handle_per_scene_tts(self, node: TaskNode, state: ProjectState) -> Any:
        """Synthesize TTS audio for a single scene's dialogue and narration.

        Each per-scene TTS node runs independently, enabling parallel execution
        and per-scene fault isolation / checkpoint.  Results are stored in
        ``state.assets["tts_scene_{scene_id}"]`` as JSON-serialised AudioSegment
        dicts so the downstream subtitle_gen node can aggregate them.
        """
        import json as _json
        from pathlib import Path

        from videoclaw.generation.audio.tts import TTSManager

        tts = TTSManager()
        language = node.params.get("language", state.metadata.get("language", "zh"))
        project_dir = Path(self._config.projects_dir) / state.project_id
        audio_dir = project_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        scene_data: dict = node.params.get("scene", {})
        scene_id = scene_data.get("scene_id", "unknown")

        from videoclaw.drama.models import (
            AudioSegment as DramaAudioSegment,
        )
        from videoclaw.drama.models import (
            DialogueLine,
            LineType,
            VoiceProfile,
        )

        # Reconstruct voice_map
        raw_voice_map: dict[str, dict] = (
            node.params.get("voice_map")
            or state.metadata.get("voice_map")
            or {}
        )
        voice_map: dict[str, VoiceProfile] = {
            name: VoiceProfile.from_dict(vp_data)
            for name, vp_data in raw_voice_map.items()
        }

        # Build DialogueLine list for this scene
        lines: list[DialogueLine] = []
        dialogue = scene_data.get("dialogue", "").strip()
        narration = scene_data.get("narration", "").strip()
        speaking_character = scene_data.get("speaking_character", "")
        emotion = scene_data.get("emotion", "")
        dialogue_line_type_str = scene_data.get("dialogue_line_type", "dialogue")

        try:
            dialogue_line_type = LineType(dialogue_line_type_str)
        except ValueError:
            dialogue_line_type = LineType.DIALOGUE

        if dialogue:
            lines.append(DialogueLine(
                text=dialogue,
                speaker=speaking_character or "narrator",
                line_type=dialogue_line_type,
                scene_id=scene_id,
                emotion_hint=emotion or None,
            ))

        narration_type = scene_data.get("narration_type", "voiceover")
        if narration and narration_type != "title_card":
            lines.append(DialogueLine(
                text=narration,
                speaker="narrator",
                line_type=LineType.NARRATION,
                scene_id=scene_id,
                emotion_hint=None,
            ))

        # Synthesize
        segments: list[DramaAudioSegment] = await tts.generate_multi_role(
            lines, voice_map, audio_dir, language=language,
        )

        # --- Audio post-processing (inner_monologue echo, narration EQ, etc.) ---
        try:
            from videoclaw.generation.audio.audio_post import AudioPostProcessor

            post = AudioPostProcessor()
            for seg in segments:
                if seg.audio_path and Path(seg.audio_path).exists():
                    processed_path = audio_dir / f"{seg.segment_id}_post.mp3"
                    await post.process(
                        Path(seg.audio_path), processed_path, seg.line_type,
                    )
                    seg.audio_path = str(processed_path)
        except Exception as exc:
            logger.warning(
                "[per_scene_tts] Audio post-processing failed for scene %s: %s",
                scene_id, exc,
            )

        # Store per-scene result for downstream aggregation
        seg_dicts = [s.to_dict() for s in segments]
        state.assets[f"tts_scene_{scene_id}"] = _json.dumps(seg_dicts)

        logger.info(
            "[per_scene_tts] scene %s: synthesized %d segments (post-processed)",
            scene_id,
            len(segments),
        )
        return {
            "scene_id": scene_id,
            "segments": len(segments),
            "audio_paths": [s.audio_path for s in segments],
        }

    async def _handle_subtitle_gen(self, node: TaskNode, state: ProjectState) -> Any:
        """Generate subtitles as a standalone DAG node.

        Collects per-scene AudioSegments from ``state.assets["tts_scene_*"]``
        to build an :class:`EpisodeAudioManifest` for accurate timing, then
        delegates to :class:`SubtitleGenerator`.
        """
        import json as _json
        from pathlib import Path

        from videoclaw.generation.subtitle import SubtitleGenerator

        project_dir = Path(self._config.projects_dir) / state.project_id
        scenes = node.params.get("scenes", [])

        if not scenes:
            logger.info("[subtitle_gen] No scenes data, skipping")
            return {"status": "skipped"}

        # Aggregate per-scene audio segments into a manifest dict

        all_segments: list[dict] = []
        for scene_data in scenes:
            scene_id = scene_data.get("scene_id", "")
            raw = state.assets.get(f"tts_scene_{scene_id}")
            if raw:
                try:
                    seg_dicts = _json.loads(raw)
                    all_segments.extend(seg_dicts)
                except (TypeError, _json.JSONDecodeError):
                    pass

        audio_manifest: dict | None = None
        if all_segments:
            total_duration = sum(s.get("duration_seconds", 0.0) for s in all_segments)
            audio_manifest = {
                "episode_id": state.metadata.get("episode_id", ""),
                "segments": all_segments,
                "total_duration": total_duration,
            }
            state.assets["audio_manifest"] = _json.dumps(audio_manifest)

        # Also build backward-compat tts_audio list
        audio_paths: list[dict[str, str]] = []
        for seg in all_segments:
            audio_paths.append({
                "scene_id": seg.get("scene_id", ""),
                "type": seg.get("audio_type", "dialogue"),
                "path": seg.get("audio_path", ""),
            })
        state.assets["tts_audio"] = _json.dumps(audio_paths)

        # Extract character colors from metadata
        character_colors: dict[str, str] | None = state.metadata.get("character_colors")
        title = state.metadata.get("series_id", "Untitled")
        language = state.metadata.get("language", "zh")

        sub_gen = SubtitleGenerator()

        # Generate ASS (primary) with SRT fallback
        try:
            subtitle_path = project_dir / "subtitles.ass"
            sub_gen.generate_ass(
                scenes,
                subtitle_path,
                audio_manifest=audio_manifest,
                character_colors=character_colors,
                title=title,
                language=language,
            )
        except Exception:
            logger.warning("[subtitle_gen] ASS generation failed, falling back to SRT")
            subtitle_path = project_dir / "subtitles.srt"
            sub_gen.generate_srt(
                scenes,
                subtitle_path,
                audio_manifest=audio_manifest,
                language=language,
            )

        state.assets["subtitles"] = str(subtitle_path)
        logger.info("[subtitle_gen] Generated subtitles -> %s", subtitle_path)
        return {"subtitle_path": str(subtitle_path), "segments_used": len(all_segments)}

    async def _handle_music(self, node: TaskNode, state: ProjectState) -> Any:
        """Generate background music track for the episode."""
        from pathlib import Path

        from videoclaw.generation.audio.music import MusicManager

        project_dir = Path(self._config.projects_dir) / state.project_id
        music_dir = project_dir / "audio"
        music_dir.mkdir(parents=True, exist_ok=True)

        # Extract music params from node or defaults
        mood = node.params.get("mood", "neutral")
        style = node.params.get("style", "orchestral")

        # Calculate total duration from storyboard
        duration = sum(s.duration_seconds for s in state.storyboard) if state.storyboard else 60.0

        output_path = music_dir / "bgm.aac"
        manager = MusicManager()

        try:
            path = await manager.generate_bgm(
                mood=mood,
                style=style,
                duration_seconds=duration,
                output_path=output_path,
            )
            state.assets["music"] = str(path)
            logger.info("[music] Generated BGM -> %s", path)
            return {"music_path": str(path), "duration": duration}
        except Exception as exc:
            logger.warning("[music] BGM generation failed: %s, skipping", exc)
            state.assets["music"] = ""
            return {"status": "skipped", "reason": str(exc)}

    async def _handle_compose(self, node: TaskNode, state: ProjectState) -> Any:
        """Compose video clips + audio + subtitles into a single timeline.

        Subtitles are now generated by the upstream ``subtitle_gen`` node and
        read from ``state.assets["subtitles"]``.  Audio segments are aggregated
        by ``subtitle_gen`` into ``state.assets["tts_audio"]``.
        """
        import json as _json
        from pathlib import Path

        from videoclaw.generation.compose import (
            AlignmentReport,
            AudioTrack,
            AudioType,
            VideoComposer,
            align_clips,
        )

        project_dir = Path(self._config.projects_dir) / state.project_id
        composer = VideoComposer()

        # 1. Collect video paths from completed shots
        video_paths: list[Path] = []
        for shot in state.storyboard:
            if shot.asset_path and Path(shot.asset_path).exists():
                video_paths.append(Path(shot.asset_path))

        if not video_paths:
            raise ValueError("No video assets available for composition")

        # 2. Pre-compose alignment: probe actual durations and match to scenes
        composed_path = project_dir / "composed.mp4"
        transition = node.params.get("transition", "dissolve")
        scenes = node.params.get("scenes", [])

        per_scene_transitions: list[str] | None = None
        clip_durations: list[float] | None = None
        alignment: AlignmentReport | None = None

        if scenes and len(scenes) == len(video_paths):
            # Scene-anchored alignment: probe actual durations and compare
            alignment = await align_clips(video_paths, scenes)

            # ALWAYS use actual probed durations for xfade offset calculation
            # to prevent transition misalignment from Seedance duration drift
            clip_durations = [c.actual_duration for c in alignment.clips]
            per_scene_transitions = [c.transition for c in alignment.clips]

            if not alignment.is_aligned:
                logger.warning(
                    "[compose] Duration misalignment detected in %d scenes: %s "
                    "(using actual durations for transitions)",
                    len(alignment.misaligned_scene_ids),
                    ", ".join(alignment.misaligned_scene_ids),
                )
        elif scenes:
            # Fallback: video count != scene count — log and probe durations
            logger.warning(
                "[compose] Video count (%d) != scene count (%d), "
                "cannot scene-anchor — probing actual durations",
                len(video_paths), len(scenes),
            )
            per_scene_transitions = [
                s.get("transition", "") or "" for s in scenes
            ]
            # clip_durations=None → compose() will probe via ffprobe
        # else: no scenes — compose() probes durations automatically

        await composer.compose(
            video_paths,
            composed_path,
            transition=transition,
            transitions=per_scene_transitions,
            clip_durations=clip_durations,
        )

        # Store alignment report in state for downstream audit
        if alignment:
            import json as _alignment_json
            state.assets["alignment_report"] = _alignment_json.dumps({
                "is_aligned": alignment.is_aligned,
                "total_scripted": alignment.total_scripted,
                "total_actual": alignment.total_actual,
                "total_drift": alignment.total_drift,
                "misaligned_scene_ids": alignment.misaligned_scene_ids,
                "clips": [
                    {
                        "scene_id": c.scene_id,
                        "scripted": c.scripted_duration,
                        "actual": c.actual_duration,
                        "drift": c.drift,
                    }
                    for c in alignment.clips
                ],
            })

        logger.info("[compose] Composed %d clips -> %s", len(video_paths), composed_path)

        # 3. Read subtitles from upstream subtitle_gen node
        subtitle_path: Path | None = None
        raw_sub = state.assets.get("subtitles")
        if raw_sub and Path(raw_sub).exists():
            subtitle_path = Path(raw_sub)

        # 4. Collect audio tracks (TTS + music)
        #    Prefer audio_manifest (has start_time + line_type for proper mix)
        #    over the flat tts_audio list.
        audio_tracks: list[AudioTrack] = []
        raw_manifest = state.assets.get("audio_manifest")
        if raw_manifest:
            from videoclaw.drama.models import LineType
            manifest_data = _json.loads(raw_manifest)
            for seg in manifest_data.get("segments", []):
                seg_path = seg.get("audio_path", "")
                if seg_path and Path(seg_path).exists():
                    line_type = seg.get("line_type", "dialogue")
                    vol = 0.85 if line_type == LineType.NARRATION else 1.0
                    audio_tracks.append(AudioTrack(
                        path=Path(seg_path),
                        type=AudioType.VOICE,
                        volume=vol,
                        start_time=float(seg.get("start_time", 0.0)),
                    ))
        else:
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

        # Add music track if available
        music_path_str = state.assets.get("music", "")
        if music_path_str and Path(music_path_str).exists():
            audio_tracks.append(AudioTrack(
                path=Path(music_path_str),
                type=AudioType.MUSIC,
                volume=0.3,  # BGM lower than dialogue
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

        from videoclaw.generation.render import _ASPECT_TO_RENDER_RESOLUTION, VideoRenderer

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
