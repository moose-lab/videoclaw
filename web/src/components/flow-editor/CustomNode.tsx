/**
 * Custom Flow Node Component
 *
 * Renders individual nodes in the DAG editor with styled appearance.
 */

"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import {
  type FlowNodeData,
  type TaskType,
  TASK_TYPE_COLORS,
  TASK_TYPE_LABELS,
} from "./types";
import {
  FileText,
  LayoutGrid,
  Video,
  Mic,
  Music,
  Layers,
  Film,
} from "lucide-react";

// Icon map for task types
const TASK_ICONS: Record<TaskType, React.ComponentType<{ className?: string }>> = {
  script_gen: FileText,
  storyboard: LayoutGrid,
  video_gen: Video,
  tts: Mic,
  music: Music,
  compose: Layers,
  render: Film,
};

interface CustomNodeProps extends NodeProps<FlowNodeData> {
  selected?: boolean;
}

function CustomNodeComponent({ data, selected }: CustomNodeProps) {
  const Icon = TASK_ICONS[data.type];
  const color = TASK_TYPE_COLORS[data.type];
  const label = TASK_TYPE_LABELS[data.type];

  return (
    <div
      className={cn(
        "relative min-w-[140px] rounded-lg border-2 bg-background/95 backdrop-blur-sm transition-all",
        "shadow-lg hover:shadow-xl",
        selected ? "ring-2 ring-primary ring-offset-2 ring-offset-background" : "",
        "cursor-pointer"
      )}
      style={{
        borderColor: color,
        boxShadow: selected ? `0 0 20px ${color}40` : undefined,
      }}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !border-2"
        style={{ 
          backgroundColor: color,
          borderColor: color,
        }}
      />

      {/* Node content */}
      <div className="p-3">
        <div className="flex items-center gap-2">
          <div
            className="flex h-8 w-8 items-center justify-center rounded-md"
            style={{ backgroundColor: `${color}20` }}
          >
            <Icon className="h-4 w-4" style={{ color }} />
          </div>
          <div className="flex flex-col">
            <span className="text-xs font-semibold" style={{ color }}>
              {label}
            </span>
            <span className="text-sm font-medium truncate max-w-[100px]">
              {data.label}
            </span>
          </div>
        </div>
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !border-2"
        style={{ 
          backgroundColor: color,
          borderColor: color,
        }}
      />
    </div>
  );
}

export const CustomNode = memo(CustomNodeComponent);
