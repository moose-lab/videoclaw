"use client";

import { useState, useCallback } from "react";
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
} from "lucide-react";
import YAML from "@/lib/yaml-parse";

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

interface ParsedFlow {
  name: string;
  description?: string;
  variables?: Record<string, unknown>;
  steps: FlowStep[];
}

export default function FlowPage() {
  const router = useRouter();
  const [yaml, setYaml] = useState(EXAMPLE_FLOW);
  const [parsed, setParsed] = useState<ParsedFlow | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<{
    project_id: string;
    message: string;
  } | null>(null);

  const handleParse = useCallback(() => {
    try {
      const data = YAML.parse(yaml);
      if (!data || !data.steps) {
        setParseError("Invalid flow: missing 'steps'");
        setParsed(null);
        return false;
      }
      setParsed(data as ParsedFlow);
      setParseError(null);
      return true;
    } catch (e) {
      setParseError(e instanceof Error ? e.message : "Parse error");
      setParsed(null);
      return false;
    }
  }, [yaml]);

  const handleRun = async () => {
    const valid = handleParse();
    if (!valid) return;

    setRunning(true);
    setRunResult(null);
    try {
      // Re-parse to get latest
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
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Workflow className="h-6 w-6 text-primary" />
          ClawFlow Editor
        </h1>
        <p className="text-muted-foreground">
          Define and run video pipelines with YAML
        </p>
      </div>

      <Tabs defaultValue="editor">
        <TabsList>
          <TabsTrigger value="editor">YAML Editor</TabsTrigger>
          <TabsTrigger value="visual">Visual DAG</TabsTrigger>
        </TabsList>

        <TabsContent value="editor" className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <Textarea
                value={yaml}
                onChange={(e) => setYaml(e.target.value)}
                className="font-mono text-sm min-h-[400px] resize-none"
                spellCheck={false}
              />
              <div className="flex gap-2 mt-4">
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
            </CardContent>
          </Card>

          {parseError && (
            <Card className="border-destructive">
              <CardContent className="pt-6 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
                <pre className="text-sm text-destructive whitespace-pre-wrap">
                  {parseError}
                </pre>
              </CardContent>
            </Card>
          )}

          {parsed && !parseError && (
            <Card className="border-green-500/50">
              <CardContent className="pt-6 text-sm">
                <div className="flex items-center gap-2 text-green-500 mb-2">
                  <Check className="h-4 w-4" />
                  Flow is valid
                </div>
                <div className="text-muted-foreground">
                  <strong>{parsed.name}</strong> &mdash; {parsed.steps.length} steps
                  {parsed.variables &&
                    `, ${Object.keys(parsed.variables).length} variables`}
                </div>
              </CardContent>
            </Card>
          )}

          {runResult && (
            <Card className="border-green-500/50">
              <CardContent className="pt-6 space-y-2">
                <Badge className="bg-green-500/10 text-green-500">Started</Badge>
                <p className="text-sm text-muted-foreground">{runResult.message}</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => router.push(`/projects/${runResult.project_id}`)}
                >
                  View Project
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="visual" className="space-y-4">
          {!parsed ? (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                Validate a flow first to see the visual DAG.
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{parsed.name}</CardTitle>
                <CardDescription>
                  {parsed.steps.length} steps &bull; {levels.length} levels
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center gap-3">
                  {levels.map((level, li) => (
                    <div key={li}>
                      {li > 0 && (
                        <div className="flex justify-center py-1">
                          <ArrowDown className="h-4 w-4 text-muted-foreground" />
                        </div>
                      )}
                      <div className="flex gap-3 justify-center flex-wrap">
                        {level.map((step) => (
                          <div
                            key={step.id}
                            className={`rounded-lg border px-4 py-2.5 text-sm min-w-[140px] text-center ${
                              typeColors[step.type] || "border-border"
                            }`}
                          >
                            <div className="font-medium">{step.id}</div>
                            <div className="text-xs opacity-70">{step.type}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
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
