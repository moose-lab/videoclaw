"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, ProjectDetail } from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ArrowLeft,
  RefreshCw,
  Film,
  Clock,
  DollarSign,
  Layers,
} from "lucide-react";

const shotStatusColor: Record<string, string> = {
  pending: "bg-muted text-muted-foreground",
  generating: "bg-yellow-500/10 text-yellow-500",
  completed: "bg-green-500/10 text-green-500",
  failed: "bg-red-500/10 text-red-500",
};

export default function ProjectPage() {
  const params = useParams();
  const id = params.id as string;
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.getProject(id);
      setProject(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load project");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground">
        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
        Loading project...
      </div>
    );
  }

  if (error || !project) {
    return (
      <Card className="border-destructive">
        <CardContent className="pt-6 text-destructive">
          {error || "Project not found"}
          <div className="mt-4">
            <Link href="/">
              <Button variant="outline" size="sm">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Dashboard
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    );
  }

  const totalDuration = project.storyboard.reduce(
    (sum, s) => sum + s.duration_seconds,
    0
  );

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <Link
            href="/"
            className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1 mb-2"
          >
            <ArrowLeft className="h-3 w-3" />
            Back
          </Link>
          <h1 className="text-2xl font-bold">
            {project.prompt.length > 80
              ? project.prompt.slice(0, 80) + "..."
              : project.prompt}
          </h1>
          <p className="text-sm text-muted-foreground font-mono mt-1">{id}</p>
        </div>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="mr-1.5 h-4 w-4" />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6 flex items-center gap-3">
            <Badge
              variant="secondary"
              className={`text-base px-3 py-1 ${
                project.status === "completed"
                  ? "bg-green-500/10 text-green-500"
                  : project.status === "failed"
                  ? "bg-red-500/10 text-red-500"
                  : "bg-yellow-500/10 text-yellow-500"
              }`}
            >
              {project.status}
            </Badge>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6 flex items-center gap-3">
            <Layers className="h-5 w-5 text-muted-foreground" />
            <div>
              <div className="text-2xl font-bold">{project.storyboard.length}</div>
              <div className="text-xs text-muted-foreground">Shots</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6 flex items-center gap-3">
            <Clock className="h-5 w-5 text-muted-foreground" />
            <div>
              <div className="text-2xl font-bold">{totalDuration.toFixed(0)}s</div>
              <div className="text-xs text-muted-foreground">Duration</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6 flex items-center gap-3">
            <DollarSign className="h-5 w-5 text-muted-foreground" />
            <div>
              <div className="text-2xl font-bold font-mono">
                ${project.cost_total.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">Cost</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {project.script && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Script</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">
              {project.script}
            </p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Film className="h-4 w-4" />
            Storyboard
          </CardTitle>
          <CardDescription>
            {project.storyboard.length} shots, {totalDuration.toFixed(1)}s total
          </CardDescription>
        </CardHeader>
        <CardContent>
          {project.storyboard.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">
              No shots planned yet.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">#</TableHead>
                  <TableHead>Description</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Duration</TableHead>
                  <TableHead className="text-right">Cost</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {project.storyboard.map((shot, i) => (
                  <TableRow key={shot.shot_id}>
                    <TableCell className="text-muted-foreground">{i + 1}</TableCell>
                    <TableCell>
                      <div className="font-medium text-sm">
                        {(shot.description || shot.prompt || "").slice(0, 80)}
                      </div>
                      <div className="text-xs text-muted-foreground font-mono">
                        {shot.shot_id}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{shot.model_id}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="secondary"
                        className={shotStatusColor[shot.status] || ""}
                      >
                        {shot.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      {shot.duration_seconds.toFixed(1)}s
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      ${shot.cost.toFixed(4)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {project.metadata && Object.keys(project.metadata).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Metadata</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs text-muted-foreground overflow-auto max-h-48 rounded bg-muted p-3">
              {JSON.stringify(project.metadata, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
