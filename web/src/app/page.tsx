"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Project } from "@/lib/api";
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
import { Film, Plus, Trash2, RefreshCw } from "lucide-react";

const statusColor: Record<string, string> = {
  planning: "bg-blue-500/10 text-blue-500",
  generating: "bg-yellow-500/10 text-yellow-500",
  composing: "bg-purple-500/10 text-purple-500",
  completed: "bg-green-500/10 text-green-500",
  failed: "bg-red-500/10 text-red-500",
};

export default function Dashboard() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadProjects = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listProjects();
      setProjects(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load projects");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProjects();
  }, []);

  const handleDelete = async (id: string) => {
    try {
      await api.deleteProject(id);
      setProjects((prev) => prev.filter((p) => p.project_id !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Projects</h1>
          <p className="text-muted-foreground">Manage your video generation projects</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={loadProjects}>
            <RefreshCw className="mr-1.5 h-4 w-4" />
            Refresh
          </Button>
          <Link href="/generate">
            <Button size="sm">
              <Plus className="mr-1.5 h-4 w-4" />
              New Video
            </Button>
          </Link>
        </div>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6 text-destructive text-sm">
            {error}
            <span className="block mt-1 text-muted-foreground text-xs">
              Make sure the API server is running: uvicorn videoclaw.server.app:create_app --factory
            </span>
          </CardContent>
        </Card>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-20 text-muted-foreground">
          <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
          Loading projects...
        </div>
      ) : projects.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-20">
            <Film className="h-12 w-12 text-muted-foreground mb-4" />
            <CardTitle className="mb-2">No projects yet</CardTitle>
            <CardDescription>
              Create your first video with a single prompt or a ClawFlow pipeline.
            </CardDescription>
            <div className="flex gap-2 mt-4">
              <Link href="/generate">
                <Button>Generate Video</Button>
              </Link>
              <Link href="/flow">
                <Button variant="outline">Open ClawFlow</Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Project</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Shots</TableHead>
                <TableHead className="text-right">Cost</TableHead>
                <TableHead>Created</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {projects.map((p) => (
                <TableRow key={p.project_id}>
                  <TableCell>
                    <Link
                      href={`/projects/${p.project_id}`}
                      className="font-medium hover:underline"
                    >
                      {p.prompt.length > 60 ? p.prompt.slice(0, 60) + "..." : p.prompt}
                    </Link>
                    <div className="text-xs text-muted-foreground font-mono">
                      {p.project_id.slice(0, 12)}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary" className={statusColor[p.status] || ""}>
                      {p.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{p.shots_count}</TableCell>
                  <TableCell className="text-right font-mono">
                    ${p.cost_total.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm">
                    {new Date(p.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      onClick={() => handleDelete(p.project_id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      {/* Stats cards */}
      {projects.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Projects</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{projects.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Shots</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {projects.reduce((sum, p) => sum + p.shots_count, 0)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Cost</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold font-mono">
                ${projects.reduce((sum, p) => sum + p.cost_total, 0).toFixed(2)}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
