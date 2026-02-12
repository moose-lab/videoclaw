"""Parse and validate ClawFlow YAML files.

Example flow YAML::

    name: product-promo
    description: Generate a 30-second product promo
    version: "1.0"

    variables:
      product_name: "VideoClaw"
      duration: 5

    steps:
      - id: script
        type: script_gen
        params:
          prompt: "Write a promo for {{product_name}}"

      - id: storyboard
        type: storyboard
        depends_on: [script]

      - id: shot_hero
        type: video_gen
        depends_on: [storyboard]
        params:
          prompt: "Product hero shot, cinematic"
          model_id: sora
          duration: "{{duration}}"

      - id: narration
        type: tts
        depends_on: [script]
        params:
          voice: alloy

      - id: bgm
        type: music
        depends_on: [storyboard]

      - id: compose
        type: compose
        depends_on: [shot_hero, narration, bgm]

      - id: render
        type: render
        depends_on: [compose]
        params:
          resolution: 1080p
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from videoclaw.core.planner import TaskType

logger = logging.getLogger(__name__)

_VALID_TYPES = {t.value for t in TaskType}
_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FlowStep:
    """A single step in a flow definition."""

    id: str
    type: TaskType
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "depends_on": self.depends_on,
            "params": self.params,
        }


@dataclass
class FlowDef:
    """Complete flow definition parsed from YAML."""

    name: str
    steps: list[FlowStep]
    description: str = ""
    version: str = "1.0"
    variables: dict[str, Any] = field(default_factory=dict)

    @property
    def step_ids(self) -> set[str]:
        return {s.id for s in self.steps}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "variables": self.variables,
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Variable interpolation
# ---------------------------------------------------------------------------

def _interpolate(value: Any, variables: dict[str, Any]) -> Any:
    """Replace ``{{var}}`` placeholders in string values."""
    if not isinstance(value, str):
        return value
    if "{{" not in value:
        return value

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            raise ValueError(f"Undefined variable: {{{{{key}}}}}")
        return str(variables[key])

    result = _VAR_PATTERN.sub(_replace, value)

    # Try numeric conversion for pure-number results.
    try:
        if "." in result:
            return float(result)
        return int(result)
    except ValueError:
        return result


def _interpolate_params(params: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    """Interpolate all string values in a params dict."""
    return {k: _interpolate(v, variables) for k, v in params.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class FlowValidationError(ValueError):
    """Raised when a flow file is invalid."""


def _validate(flow: FlowDef) -> None:
    """Run validation checks and raise FlowValidationError on problems."""
    errors: list[str] = []
    ids = flow.step_ids

    if not flow.name:
        errors.append("Flow must have a 'name'")
    if not flow.steps:
        errors.append("Flow must have at least one step")

    seen: set[str] = set()
    for step in flow.steps:
        if step.id in seen:
            errors.append(f"Duplicate step id: {step.id!r}")
        seen.add(step.id)

        for dep in step.depends_on:
            if dep not in ids:
                errors.append(f"Step {step.id!r} depends on unknown step {dep!r}")
            if dep == step.id:
                errors.append(f"Step {step.id!r} depends on itself")

    # Simple cycle detection via topological ordering.
    in_degree: dict[str, int] = {s.id: 0 for s in flow.steps}
    adj: dict[str, list[str]] = {s.id: [] for s in flow.steps}
    for step in flow.steps:
        for dep in step.depends_on:
            if dep in adj:
                adj[dep].append(step.id)
                in_degree[step.id] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        nid = queue.pop()
        visited += 1
        for child in adj[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited != len(flow.steps):
        errors.append("Flow contains a dependency cycle")

    if errors:
        raise FlowValidationError("; ".join(errors))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_flow(raw: dict[str, Any]) -> FlowDef:
    """Parse a raw dict (from YAML) into a validated FlowDef.

    Raises :class:`FlowValidationError` on invalid input.
    """
    variables = raw.get("variables", {})

    steps: list[FlowStep] = []
    for s in raw.get("steps", []):
        step_type = s.get("type", "")
        if step_type not in _VALID_TYPES:
            raise FlowValidationError(
                f"Unknown step type {step_type!r} in step {s.get('id', '?')!r}. "
                f"Valid types: {sorted(_VALID_TYPES)}"
            )
        params = _interpolate_params(s.get("params", {}), variables)
        steps.append(FlowStep(
            id=s["id"],
            type=TaskType(step_type),
            depends_on=s.get("depends_on", []),
            params=params,
        ))

    flow = FlowDef(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        version=str(raw.get("version", "1.0")),
        variables=variables,
        steps=steps,
    )
    _validate(flow)
    return flow


def load_flow(path: str | Path) -> FlowDef:
    """Load and parse a ClawFlow YAML file from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Flow file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FlowValidationError(f"Flow file must be a YAML mapping, got {type(raw).__name__}")

    logger.info("Loaded flow %r from %s", raw.get("name", "?"), path)
    return parse_flow(raw)
