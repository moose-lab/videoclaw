/**
 * Flow Editor Utilities
 *
 * Utilities for converting between YAML Flow definitions and React Flow format.
 */

import YAML from "yaml";
import type { Node, Edge } from "@xyflow/react";
import {
  type FlowDef,
  type FlowStep,
  type FlowNode,
  type FlowEdge,
  type FlowNodeData,
  type TaskType,
  TASK_TYPE_COLORS,
} from "./types";

// Layout constants
const NODE_WIDTH = 160;
const NODE_HEIGHT = 80;
const HORIZONTAL_GAP = 60;
const VERTICAL_GAP = 40;

/**
 * Parse YAML string to FlowDef
 */
export function parseYamlToFlowDef(yaml: string): FlowDef | null {
  try {
    const data = YAML.parse(yaml);
    if (!data || !data.steps) {
      return null;
    }
    return data as FlowDef;
  } catch {
    return null;
  }
}

/**
 * Convert FlowDef to YAML string
 */
export function flowDefToYaml(flow: FlowDef): string {
  return YAML.stringify(flow);
}

/**
 * Calculate layout positions for nodes using layered approach
 */
function calculateNodePositions(steps: FlowStep[]): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const stepMap = new Map<string, FlowStep>();
  const levels = new Map<string, number>();

  // Build step map
  steps.forEach((step) => stepMap.set(step.id, step));

  // Calculate levels (topological sort)
  function getLevel(stepId: string, visited: Set<string>): number {
    if (levels.has(stepId)) {
      return levels.get(stepId)!;
    }
    if (visited.has(stepId)) {
      return 0; // Cycle detected, default to level 0
    }
    visited.add(stepId);

    const step = stepMap.get(stepId);
    if (!step || step.depends_on.length === 0) {
      levels.set(stepId, 0);
      return 0;
    }

    const maxDepLevel = Math.max(
      ...step.depends_on
        .filter((dep) => stepMap.has(dep))
        .map((dep) => getLevel(dep, visited))
    );
    const level = maxDepLevel + 1;
    levels.set(stepId, level);
    return level;
  }

  // Calculate levels for all steps
  steps.forEach((step) => getLevel(step.id, new Set()));

  // Group steps by level
  const levelGroups = new Map<number, string[]>();
  levels.forEach((level, stepId) => {
    if (!levelGroups.has(level)) {
      levelGroups.set(level, []);
    }
    levelGroups.get(level)!.push(stepId);
  });

  // Assign positions
  levelGroups.forEach((stepIds, level) => {
    const x = level * (NODE_WIDTH + HORIZONTAL_GAP);
    const totalHeight = stepIds.length * NODE_HEIGHT + (stepIds.length - 1) * VERTICAL_GAP;
    const startY = -totalHeight / 2;

    stepIds.forEach((stepId, index) => {
      positions.set(stepId, {
        x,
        y: startY + index * (NODE_HEIGHT + VERTICAL_GAP),
      });
    });
  });

  return positions;
}

/**
 * Convert FlowDef to React Flow nodes and edges
 */
export function flowDefToReactFlow(flow: FlowDef): { nodes: FlowNode[]; edges: FlowEdge[] } {
  const nodes: FlowNode[] = [];
  const edges: FlowEdge[] = [];
  const positions = calculateNodePositions(flow.steps);

  // Create nodes
  flow.steps.forEach((step) => {
    const pos = positions.get(step.id) || { x: 0, y: 0 };
    nodes.push({
      id: step.id,
      type: "flow",
      position: pos,
      data: {
        label: step.id,
        type: step.type,
        params: step.params,
      },
    });
  });

  // Create edges
  flow.steps.forEach((step) => {
    step.depends_on.forEach((depId) => {
      edges.push({
        id: `${depId}-${step.id}`,
        source: depId,
        target: step.id,
        type: "smoothstep",
        animated: false,
        style: { stroke: TASK_TYPE_COLORS[step.type] },
      });
    });
  });

  return { nodes, edges };
}

/**
 * Convert React Flow nodes and edges back to FlowDef
 */
export function reactFlowToFlowDef(
  nodes: FlowNode[],
  edges: FlowEdge[],
  existingFlow?: FlowDef
): FlowDef {
  const stepMap = new Map<string, FlowStep>();

  // Build steps from nodes
  nodes.forEach((node) => {
    const deps = edges
      .filter((e) => e.target === node.id)
      .map((e) => e.source);

    stepMap.set(node.id, {
      id: node.id,
      type: node.data.type,
      depends_on: deps,
      params: node.data.params,
    });
  });

  // Preserve order from existing flow if available
  let steps: FlowStep[];
  if (existingFlow?.steps) {
    steps = existingFlow.steps
      .map((s) => stepMap.get(s.id))
      .filter((s): s is FlowStep => s !== undefined);
    // Add any new nodes
    stepMap.forEach((step, id) => {
      if (!steps.find((s) => s.id === id)) {
        steps.push(step);
      }
    });
  } else {
    steps = Array.from(stepMap.values());
  }

  return {
    name: existingFlow?.name || "untitled",
    description: existingFlow?.description,
    version: existingFlow?.version || "1.0",
    variables: existingFlow?.variables,
    steps,
  };
}

/**
 * Validate flow for cycles and missing dependencies
 */
export function validateFlow(flow: FlowDef): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const stepIds = new Set(flow.steps.map((s) => s.id));

  // Check for duplicate IDs
  const seen = new Set<string>();
  flow.steps.forEach((step) => {
    if (seen.has(step.id)) {
      errors.push(`Duplicate step ID: ${step.id}`);
    }
    seen.add(step.id);
  });

  // Check for missing dependencies
  flow.steps.forEach((step) => {
    step.depends_on.forEach((dep) => {
      if (!stepIds.has(dep)) {
        errors.push(`Step "${step.id}" depends on unknown step "${dep}"`);
      }
      if (dep === step.id) {
        errors.push(`Step "${step.id}" depends on itself`);
      }
    });
  });

  // Check for cycles using topological sort
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();

  flow.steps.forEach((step) => {
    inDegree.set(step.id, 0);
    adj.set(step.id, []);
  });

  flow.steps.forEach((step) => {
    step.depends_on.forEach((dep) => {
      if (stepIds.has(dep)) {
        adj.get(dep)!.push(step.id);
        inDegree.set(step.id, (inDegree.get(step.id) || 0) + 1);
      }
    });
  });

  const queue = flow.steps.filter((s) => inDegree.get(s.id) === 0);
  let visited = 0;

  while (queue.length > 0) {
    const current = queue.shift()!;
    visited++;
    adj.get(current.id)?.forEach((next) => {
      const newDegree = (inDegree.get(next) || 1) - 1;
      inDegree.set(next, newDegree);
      if (newDegree === 0) {
        queue.push(flow.steps.find((s) => s.id === next)!);
      }
    });
  }

  if (visited !== flow.steps.length) {
    errors.push("Flow contains a dependency cycle");
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Generate a unique step ID
 */
export function generateStepId(type: TaskType, existingIds: string[]): string {
  const prefix = type.replace("_gen", "");
  let counter = 1;
  let id = `${prefix}_${counter}`;

  while (existingIds.includes(id)) {
    counter++;
    id = `${prefix}_${counter}`;
  }

  return id;
}
