/**
 * Node Palette Component
 *
 * Left sidebar with draggable node templates.
 */

"use client";

import { cn } from "@/lib/utils";
import {
  type TaskType,
  type NodeTemplate,
  NODE_TEMPLATES,
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
  GripVertical,
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

interface NodePaletteProps {
  onNodeAdd?: (type: TaskType) => void;
}

export function NodePalette({ onNodeAdd }: NodePaletteProps) {
  const handleDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    template: NodeTemplate
  ) => {
    event.dataTransfer.setData("application/reactflow", JSON.stringify(template));
    event.dataTransfer.effectAllowed = "move";
  };

  const handleDoubleClick = (template: NodeTemplate) => {
    onNodeAdd?.(template.type);
  };

  return (
    <div className="h-full flex flex-col border-r border-border bg-background">
      {/* Header */}
      <div className="p-3 border-b border-border">
        <h3 className="text-sm font-semibold">Nodes</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Drag to canvas or double-click
        </p>
      </div>

      {/* Node list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {NODE_TEMPLATES.map((template) => {
          const Icon = TASK_ICONS[template.type];
          const color = TASK_TYPE_COLORS[template.type];

          return (
            <div
              key={template.type}
              draggable
              onDragStart={(e) => handleDragStart(e, template)}
              onDoubleClick={() => handleDoubleClick(template)}
              className={cn(
                "group flex items-center gap-2 p-2 rounded-lg border cursor-grab",
                "hover:bg-muted/50 transition-colors",
                "active:cursor-grabbing"
              )}
              style={{
                borderColor: `${color}40`,
              }}
            >
              <div
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md"
                style={{ backgroundColor: `${color}20` }}
              >
                <Icon className="h-4 w-4" style={{ color }} />
              </div>
              <div className="flex-1 min-w-0">
                <div
                  className="text-xs font-semibold"
                  style={{ color }}
                >
                  {template.label}
                </div>
                <div className="text-xs text-muted-foreground truncate">
                  {template.description}
                </div>
              </div>
              <GripVertical className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-border text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <kbd className="px-1.5 py-0.5 rounded bg-muted border text-[10px]">
            Delete
          </kbd>
          <span>to remove node</span>
        </div>
      </div>
    </div>
  );
}

export { type NodeTemplate };
