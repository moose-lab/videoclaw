/**
 * Node Properties Panel Component
 *
 * Right sidebar for editing selected node properties.
 */

"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  type FlowNode,
  type TaskType,
  TASK_TYPE_COLORS,
  TASK_TYPE_LABELS,
} from "./types";
import {
  X,
  Settings,
  Link2,
  Trash2,
} from "lucide-react";

interface NodePropertiesProps {
  node: FlowNode | null;
  dependencies: string[];
  onUpdate?: (nodeId: string, updates: { label?: string; params?: Record<string, unknown> }) => void;
  onDelete?: (nodeId: string) => void;
  onClose?: () => void;
}

// Common parameter fields by task type
const PARAM_FIELDS: Record<TaskType, { key: string; label: string; type: "text" | "number" | "select" }[]> = {
  script_gen: [
    { key: "prompt", label: "Prompt", type: "text" },
  ],
  storyboard: [],
  video_gen: [
    { key: "prompt", label: "Prompt", type: "text" },
    { key: "model_id", label: "Model", type: "text" },
    { key: "duration", label: "Duration (s)", type: "number" },
  ],
  tts: [
    { key: "voice", label: "Voice", type: "text" },
  ],
  music: [
    { key: "style", label: "Style", type: "text" },
    { key: "duration", label: "Duration (s)", type: "number" },
  ],
  compose: [],
  render: [
    { key: "resolution", label: "Resolution", type: "text" },
  ],
};

export function NodeProperties({
  node,
  dependencies,
  onUpdate,
  onDelete,
  onClose,
}: NodePropertiesProps) {
  const [localLabel, setLocalLabel] = useState(node?.data.label || "");

  if (!node) {
    return (
      <div className="h-full flex flex-col border-l border-border bg-background">
        <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm p-4 text-center">
          Select a node to edit its properties
        </div>
      </div>
    );
  }

  const color = TASK_TYPE_COLORS[node.data.type];
  const label = TASK_TYPE_LABELS[node.data.type];
  const paramFields = PARAM_FIELDS[node.data.type];

  const handleLabelChange = (value: string) => {
    setLocalLabel(value);
    onUpdate?.(node.id, { label: value });
  };

  const handleParamChange = (key: string, value: string | number) => {
    const newParams = { ...(node.data.params || {}), [key]: value };
    onUpdate?.(node.id, { params: newParams });
  };

  const handleDelete = () => {
    onDelete?.(node.id);
    onClose?.();
  };

  return (
    <div className="h-full flex flex-col border-l border-border bg-background">
      {/* Header */}
      <div className="p-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings className="h-4 w-4" style={{ color }} />
          <h3 className="text-sm font-semibold">Properties</h3>
        </div>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Node info */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div
              className="px-2 py-1 rounded text-xs font-semibold"
              style={{ backgroundColor: `${color}20`, color }}
            >
              {label}
            </div>
            <span className="text-xs text-muted-foreground font-mono">
              {node.id}
            </span>
          </div>
        </div>

        {/* Label */}
        <div className="space-y-1.5">
          <Label className="text-xs">Label</Label>
          <Input
            value={localLabel}
            onChange={(e) => handleLabelChange(e.target.value)}
            className="h-8 text-sm"
          />
        </div>

        {/* Parameters */}
        {paramFields.length > 0 && (
          <div className="space-y-3">
            <Label className="text-xs text-muted-foreground">Parameters</Label>
            {paramFields.map((field) => (
              <div key={field.key} className="space-y-1.5">
                <Label className="text-xs">{field.label}</Label>
                {field.type === "text" ? (
                  <Textarea
                    value={String((node.data.params || {})[field.key] || "")}
                    onChange={(e) => handleParamChange(field.key, e.target.value)}
                    className="min-h-[60px] text-sm resize-none"
                  />
                ) : (
                  <Input
                    type="number"
                    value={(node.data.params || {})[field.key] || ""}
                    onChange={(e) => handleParamChange(field.key, Number(e.target.value))}
                    className="h-8 text-sm"
                  />
                )}
              </div>
            ))}
          </div>
        )}

        {/* Dependencies */}
        {dependencies.length > 0 && (
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground flex items-center gap-1.5">
              <Link2 className="h-3 w-3" />
              Depends on
            </Label>
            <div className="flex flex-wrap gap-1.5">
              {dependencies.map((dep) => (
                <span
                  key={dep}
                  className="px-2 py-1 rounded bg-muted text-xs font-mono"
                >
                  {dep}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-border">
        <Button
          variant="destructive"
          size="sm"
          className="w-full"
          onClick={handleDelete}
        >
          <Trash2 className="h-4 w-4 mr-2" />
          Delete Node
        </Button>
      </div>
    </div>
  );
}
