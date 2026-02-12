"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Sparkles, Loader2, Wand2, ArrowRight } from "lucide-react";

const PRESETS = [
  {
    label: "Product Promo",
    prompt:
      "A sleek 30-second product promo for wireless earbuds, cinematic lighting, modern tech aesthetic",
  },
  {
    label: "Tutorial",
    prompt:
      "A step-by-step tutorial on how to make pour-over coffee, warm kitchen lighting, overhead camera",
  },
  {
    label: "Short Drama",
    prompt:
      "A mysterious traveler arrives at an ancient temple at dawn, cinematic wide shots, orchestral music",
  },
  {
    label: "Social Ad",
    prompt:
      "A vibrant 15-second social media ad for a fitness app, energetic transitions, bold typography",
  },
];

export default function GeneratePage() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ project_id: string; message: string } | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setGenerating(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.startGeneration(prompt, model || undefined);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="space-y-6 max-w-2xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Sparkles className="h-6 w-6 text-primary" />
          Generate Video
        </h1>
        <p className="text-muted-foreground">
          Describe your video and let the AI agents handle the rest.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <Badge
            key={preset.label}
            variant="outline"
            className="cursor-pointer hover:bg-primary/10 transition-colors"
            onClick={() => setPrompt(preset.prompt)}
          >
            <Wand2 className="mr-1 h-3 w-3" />
            {preset.label}
          </Badge>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Creative Prompt</CardTitle>
          <CardDescription>
            Describe the video you want. Be specific about style, mood, duration, and content.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="A 30-second cinematic product reveal for a smartwatch..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            className="resize-none"
          />

          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="text-sm text-muted-foreground mb-1.5 block">
                Model (optional)
              </label>
              <Input
                placeholder="auto (mock, sora, cogvideo...)"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
            </div>
            <Button
              onClick={handleGenerate}
              disabled={!prompt.trim() || generating}
              className="min-w-[140px]"
            >
              {generating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6 text-destructive text-sm">
            {error}
            <span className="block mt-1 text-muted-foreground text-xs">
              Make sure the API server is running.
            </span>
          </CardContent>
        </Card>
      )}

      {result && (
        <Card className="border-green-500/50">
          <CardContent className="pt-6 space-y-3">
            <div className="flex items-center gap-2">
              <Badge className="bg-green-500/10 text-green-500">Started</Badge>
              <span className="text-sm text-muted-foreground">{result.message}</span>
            </div>
            <div className="text-sm font-mono text-muted-foreground">
              Project ID: {result.project_id}
            </div>
            <Button
              variant="outline"
              onClick={() => router.push(`/projects/${result.project_id}`)}
            >
              View Project
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
