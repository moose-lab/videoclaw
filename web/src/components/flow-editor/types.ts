/**
 * Flow Editor Type Definitions
 *
 * Types for the visual DAG editor, aligned with backend TaskType enum.
 */

import type { Node, Edge } from "@xyflow/react";

// Task types matching backend TaskType enum
export type TaskType =
  | "script_gen"
  | "storyboard"
  | "video_gen"
  | "tts"
  | "music"
  | "compose"
  | "render";

// Node colors by task type
export const TASK_TYPE_COLORS: Record<TaskType, string> = {
  script_gen: "#3b82f6", // blue
  storyboard: "#8b5cf6", // purple
  video_gen: "#22c55e", // green
  tts: "#f97316", // orange
  music: "#ec4899", // pink
  compose: "#eab308", // yellow
  render: "#06b6d4", // cyan
};

// Task type labels for display
export const TASK_TYPE_LABELS: Record<TaskType, string> = {
  script_gen: "Script",
  storyboard: "Storyboard",
  video_gen: "Video Gen",
  tts: "TTS",
  music: "Music",
  compose: "Compose",
  render: "Render",
};

// Task type descriptions
export const TASK_TYPE_DESCRIPTIONS: Record<TaskType, string> = {
  script_gen: "Generate script from prompt",
  storyboard: "Break down into shots",
  video_gen: "Generate video shot",
  tts: "Text-to-speech narration",
  music: "Generate background music",
  compose: "Combine all assets",
  render: "Final video output",
};

// Flow step from YAML
export interface FlowStep {
  id: string;
  type: TaskType;
  depends_on: string[];
  params: Record<string, unknown>;
}

// Complete flow definition
export interface FlowDef {
  name: string;
  description?: string;
  version?: string;
  variables?: Record<string, unknown>;
  steps: FlowStep[];
}

// Custom node data for React Flow
export interface FlowNodeData {
  label: string;
  type: TaskType;
  params: Record<string, unknown>;
  description?: string;
}

// Custom node type
export type FlowNode = Node<FlowNodeData, "flow">;

// Custom edge type
export type FlowEdge = Edge;

// Node template for palette
export interface NodeTemplate {
  type: TaskType;
  label: string;
  description: string;
  defaultParams: Record<string, unknown>;
}

// Default node templates for palette
export const NODE_TEMPLATES: NodeTemplate[] = [
  {
    type: "script_gen",
    label: "Script",
    description: "Generate script from prompt",
    defaultParams: { prompt: "" },
  },
  {
    type: "storyboard",
    label: "Storyboard",
    description: "Break down into shots",
    defaultParams: {},
  },
  {
    type: "video_gen",
    label: "Video Gen",
    description: "Generate video shot",
    defaultParams: { prompt: "", model_id: "mock", duration: 5 },
  },
  {
    type: "tts",
    label: "TTS",
    description: "Text-to-speech narration",
    defaultParams: { voice: "alloy" },
  },
  {
    type: "music",
    label: "Music",
    description: "Generate background music",
    defaultParams: { style: "orchestral", duration: 30 },
  },
  {
    type: "compose",
    label: "Compose",
    description: "Combine all assets",
    defaultParams: {},
  },
  {
    type: "render",
    label: "Render",
    description: "Final video output",
    defaultParams: { resolution: "1080p" },
  },
];
