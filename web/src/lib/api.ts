const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Project {
  project_id: string;
  status: string;
  prompt: string;
  cost_total: number;
  created_at: string;
  updated_at: string;
  shots_count: number;
}

export interface ProjectDetail {
  project_id: string;
  status: string;
  prompt: string;
  script: string | null;
  cost_total: number;
  created_at: string;
  updated_at: string;
  storyboard: Shot[];
  assets: Record<string, string>;
  metadata: Record<string, unknown>;
}

export interface Shot {
  shot_id: string;
  description: string;
  prompt: string;
  duration_seconds: number;
  model_id: string;
  status: string;
  asset_path: string | null;
  cost: number;
}

export interface GenerateResponse {
  project_id: string;
  status: string;
  message: string;
}

export interface FlowStep {
  id: string;
  type: string;
  depends_on: string[];
  params: Record<string, unknown>;
}

export interface FlowDef {
  name: string;
  description: string;
  version: string;
  variables: Record<string, unknown>;
  steps: FlowStep[];
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  health: () => apiFetch<{ status: string }>("/health"),

  // Projects
  listProjects: () => apiFetch<Project[]>("/api/projects/"),
  getProject: (id: string) => apiFetch<ProjectDetail>(`/api/projects/${id}`),
  createProject: (prompt: string) =>
    apiFetch<Project>("/api/projects/", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    }),
  deleteProject: (id: string) =>
    apiFetch<{ deleted: string }>(`/api/projects/${id}`, { method: "DELETE" }),

  // Generation
  startGeneration: (prompt: string, model?: string) =>
    apiFetch<GenerateResponse>("/api/generate/", {
      method: "POST",
      body: JSON.stringify({ prompt, model }),
    }),
  runFlow: (flow: Record<string, unknown>, prompt?: string) =>
    apiFetch<GenerateResponse>("/api/generate/flow", {
      method: "POST",
      body: JSON.stringify({ flow, prompt }),
    }),
  getStatus: (id: string) =>
    apiFetch<{ project_id: string; status: string; cost_total: number; shots: Shot[] }>(
      `/api/generate/${id}/status`
    ),

  // WebSocket
  connectWs: (projectId: string): WebSocket => {
    const wsBase = API_BASE.replace(/^http/, "ws");
    return new WebSocket(`${wsBase}/ws/${projectId}`);
  },
};
