"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { api, FlowStep } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Workflow,
  Play,
  Check,
  AlertCircle,
  Loader2,
  ArrowDown,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import YAML from "yaml";

import {
  FlowCanvas,
  NodePalette,
  NodeProperties,
  type FlowDef,
  type FlowNode,
  type FlowEdge,
  type TaskType,
  NODE_TEMPLATES,
} from "@/components/flow-editor";
import {
  parseYamlToFlowDef,
  flowDefToYaml,
  flowDefToReactFlow,
  reactFlowToFlowDef,
  validateFlow,
  generateStepId,
} from "@/components/flow-editor/utils";

const EXAMPLE_FLOW = `name: my-video
description: A custom video pipeline
version: "1.0"

variables:
  topic: "AI video generation"

steps:
  - id: script
    type: script_gen
    params:
      prompt: "Write a script about {{topic}}"

  - id: storyboard
    type: storyboard
    depends_on: [script]

  - id: shot_1
    type: video_gen
    depends_on: [storyboard]
    params:
      prompt: "Opening shot for {{topic}}"
      model_id: mock
      duration: 5

  - id: narration
    type: tts
    depends_on: [script]

  - id: bgm
    type: music
    depends_on: [storyboard]

  - id: compose
    type: compose
    depends_on: [shot_1, narration, bgm]

  - id: render
    type: render
    depends_on: [compose]
    params:
      resolution: 1080p
`;

const typeColors: Record<string, string> = {
  script_gen: "bg-blue-500/10 text-blue-500 border-blue-500/30",
  storyboard: "bg-purple-500/10 text-purple-500 border-purple-500/30",
  video_gen: "bg-green-500/10 text-green-500 border-green-500/30",
  tts: "bg-orange-500/10 text-orange-500 border-orange-500/30",
  music: "bg-pink-500/10 text-pink-500 border-pink-500/30",
  compose: "bg-yellow-500/10 text-yellow-500 border-yellow-500/30",
  render: "bg-cyan-500/10 text-cyan-500 border-cyan-500/30",
};

export default function FlowPage() {
  const router = useRouter();
  const [yaml, setYaml] = useState(EXAMPLE_FLOW);
  const [parsed, setParsed] = useState<FlowDef | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<{
    project_id: string;
    message: string;
  } | null>(null);

  // Visual editor state
  const [nodes, setNodes] = useState<FlowNode[]>([]);
  const [edges, setEdges] = useState<FlowEdge[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Parse YAML on mount
  useEffect(() => {
    const flow = parseYamlToFlowDef(yaml);
    if (flow) {
      setParsed(flow);
      const { nodes: n, edges: e } = flowDefToReactFlow(flow);
      setNodes(n);
      setEdges(e);
      setParseError(null);
    }
  }, []);

  // Get selected node
  const selectedNode = selectedNodeId
    ? nodes.find((n) => n.id === selectedNodeId) || null
    : null;

  // Get dependencies for selected node
  const selectedDeps = selectedNodeId
    ? edges.filter((e) => e.target === selectedNodeId).map((e) => e.source)
    : [];

  // Handle YAML parse
  const handleParse = useCallback(() => {
    try {
      const flow = parseYamlToFlowDef(yaml);
      if (!flow) {
        setParseError("Invalid flow: missing 'steps'");
        setParsed(null);
        return false;
      }

      const validation = validateFlow(flow);
      if (!validation.valid) {
        setParseError(validation.errors.join("; "));
        setParsed(null);
        return false;
      }

      setParsed(flow);
      const { nodes: n, edges: e } = flowDefToReactFlow(flow);
      setNodes(n);
      setEdges(e);
      setParseError(null);
      return true;
    } catch (e) {
      setParseError(e instanceof Error ? e.message : "Parse error");
      setParsed(null);
      return false;
    }
  }, [yaml]);

  // Handle flow change from visual editor
  const handleFlowChange = useCallback(
    (flow: FlowDef) => {
      setParsed(flow);
      setYaml(flowDefToYaml(flow));
    },
    []
  );

  // Handle node add from palette
  const handleNodeAdd = useCallback(
    (type: TaskType) => {
      const existingIds = nodes.map((n) => n.id);
      const id = generateStepId(type, existingIds);
      const template = NODE_TEMPLATES.find((t) => t.type === type);

      const newNode: FlowNode = {
        id,
        type: "flow",
        position: {
          x: Math.random() * 300 + 100,
          y: Math.random() * 300 - 150,
        },
        data: {
          label: id,
          type,
          params: template?.defaultParams || {},
        },
      };

      setNodes((nds) => [...nds, newNode]);

      // Update flow
      const newFlow = reactFlowToFlowDef(
        [...nodes, newNode],
        edges,
        parsed || undefined
      );
      handleFlowChange(newFlow);
    },
    [nodes, edges, parsed, handleFlowChange]
  );

  // Handle node update
  const handleNodeUpdate = useCallback(
    (
      nodeId: string,
      updates: { label?: string; params?: Record<string, unknown> }
    ) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? {
                ...n,
                data: {
                  ...n.data,
                  ...(updates.label ? { label: updates.label } : {}),
                  ...(updates.params ? { params: updates.params } : {}),
                },
              }
            : n
        )
      );

      // Update flow
      const updatedNodes = nodes.map((n) =>
        n.id === nodeId
          ? {
              ...n,
              data: {
                ...n.data,
                ...(updates.label ? { label: updates.label } : {}),
                ...(updates.params ? { params: updates.params } : {}),
              },
            }
          : n
      );
      const newFlow = reactFlowToFlowDef(updatedNodes, edges, parsed || undefined);
      handleFlowChange(newFlow);
    },
    [nodes, edges, parsed, handleFlowChange]
  );

  // Handle node delete
  const handleNodeDelete = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId)
      );
      setSelectedNodeId(null);

      // Update flow
      const updatedNodes = nodes.filter((n) => n.id !== nodeId);
      const updatedEdges = edges.filter(
        (e) => e.source !== nodeId && e.target !== nodeId
      );
      const newFlow = reactFlowToFlowDef(
        updatedNodes,
        updatedEdges,
        parsed || undefined
      );
      handleFlowChange(newFlow);
    },
    [nodes, edges, parsed, handleFlowChange]
  );

  // Handle run
  const handleRun = async () => {
    const valid = handleParse();
    if (!valid) return;

    setRunning(true);
    setRunResult(null);
    try {
      const data = YAML.parse(yaml) as Record<string, unknown>;
      const res = await api.runFlow(data);
      setRunResult(res);
    } catch (e) {
      setParseError(e instanceof Error ? e.message : "Run failed");
    } finally {
      setRunning(false);
    }
  };

  const levels = buildLevels(parsed?.steps || []);

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Workflow className="h-6 w-6 text-primary" />
              ClawFlow Editor
            </h1>
            <p className="text-muted-foreground">
              Design and run video pipelines visually
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => handleParse()}>
              <Check className="mr-2 h-4 w-4" />
              Validate
            </Button>
            <Button onClick={handleRun} disabled={running}>
              {running ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Pipeline
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Status bar */}
        <div className="flex items-center gap-4 mt-3">
          {parsed && (
            <>
              <Badge variant="outline" className="bg-green-500/10 text-green-500">
                <Check className="mr-1 h-3 w-3" />
                {parsed.steps.length} steps
              </Badge>
              {parsed.variables && (
                <Badge variant="outline">
                  {Object.keys(parsed.variables).length} variables
                </Badge>
              )}
            </>
          )}
          {parseError && (
            <Badge variant="outline" className="bg-red-500/10 text-red-500">
              <AlertCircle className="mr-1 h-3 w-3" />
              {parseError}
            </Badge>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        <Tabs defaultValue="visual" className="flex-1 flex flex-col">
          <div className="px-6 pt-2 border-b border-border">
            <TabsList>
              <TabsTrigger value="visual">
                <Sparkles className="mr-1.5 h-4 w-4" />
                Visual Editor
              </TabsTrigger>
              <TabsTrigger value="yaml">YAML Editor</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="visual" className="flex-1 m-0 flex">
            <div className="w-[200px] shrink-0">
              <NodePalette onNodeAdd={handleNodeAdd} />
            </div>
            <div className="flex-1 relative">
              <FlowCanvas
                initialNodes={nodes}
                initialEdges={edges}
                onChange={handleFlowChange}
                existingFlow={parsed || undefined}
                onNodeSelect={setSelectedNodeId}
              />
            </div>
            <div className="w-[280px] shrink-0">
              <NodeProperties
                node={selectedNode}
                dependencies={selectedDeps}
                onUpdate={handleNodeUpdate}
                onDelete={handleNodeDelete}
                onClose={() => setSelectedNodeId(null)}
              />
            </div>
          </TabsContent>

          <TabsContent value="yaml" className="flex-1 m-0 p-6 overflow-auto">
            <Card className="h-full">
              <CardContent className="pt-6 h-full flex flex-col">
                <Textarea
                  value={yaml}
                  onChange={(e) => {
                    setYaml(e.target.value);
                    const flow = parseYamlToFlowDef(e.target.value);
                    if (flow) {
                      setParsed(flow);
                      const { nodes: n, edges: e } = flowDefToReactFlow(flow);
                      setNodes(n);
                      setEdges(e);
                      setParseError(null);
                    }
                  }}
                  className="font-mono text-sm flex-1 min-h-[400px] resize-none"
                  spellCheck={false}
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Run result */}
      {runResult && (
        <div className="px-6 py-4 border-t border-border bg-green-500/5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Badge className="bg-green-500/10 text-green-500">Started</Badge>
              <span className="text-sm text-muted-foreground">
                {runResult.message}
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => router.push(`/projects/${runResult.project_id}`)}
            >
              View Project
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function buildLevels(steps: FlowStep[]): FlowStep[][] {
  if (steps.length === 0) return [];

  const idSet = new Set(steps.map((s) => s.id));
  const assigned = new Set<string>();
  const levels: FlowStep[][] = [];

  for (let safety = 0; safety < 20; safety++) {
    const level = steps.filter(
      (s) =>
        !assigned.has(s.id) &&
        (s.depends_on || []).every((d) => assigned.has(d) || !idSet.has(d))
    );
    if (level.length === 0) break;
    levels.push(level);
    level.forEach((s) => assigned.add(s.id));
  }

  const remaining = steps.filter((s) => !assigned.has(s.id));
  if (remaining.length > 0) levels.push(remaining);

  return levels;
}
