/** API client — Code Review System (FastAPI backend at localhost:8000) */

// ── Legacy types (from PersonalQA, kept for backward compat) ──

export interface Citation {
  number: number;
  text: string;
}

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Code Review API types ──

export type TaskStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
export type Severity = "Critical" | "Important" | "Minor";

export interface TaskResponse {
  task_id: string;
  status: TaskStatus;
  progress: string;
  created_at: string;
  updated_at: string;
  error?: string;
}

export interface Issue {
  severity: Severity;
  file_path: string;
  line: number;
  title: string;
  description: string;
  suggestion: string;
}

export interface ReportResponse {
  task_id: string;
  status: TaskStatus;
  score: number;
  issues: Issue[];
  research: string[];
  report_md: string;
  created_at: string;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
}

// ── Code Review API functions ──

export async function submitReview(
  code: string,
  language: string = "python",
  context: string = ""
): Promise<TaskResponse> {
  const res = await fetch(`${BASE_URL}/api/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, language, context }),
  });
  if (!res.ok) {
    const err: ErrorResponse = await res.json();
    throw new Error(err.detail || err.error || "Submit failed");
  }
  return res.json();
}

export async function submitPRReview(
  prUrl: string,
  token: string
): Promise<TaskResponse> {
  const res = await fetch(`${BASE_URL}/api/review/pr`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pr_url: prUrl, token }),
  });
  if (!res.ok) {
    const err: ErrorResponse = await res.json();
    throw new Error(err.detail || err.error || "Submit failed");
  }
  return res.json();
}

export async function getTask(taskId: string): Promise<TaskResponse> {
  const res = await fetch(`${BASE_URL}/api/task/${taskId}`);
  if (!res.ok) {
    const err: ErrorResponse = await res.json();
    throw new Error(err.detail || err.error || "Fetch failed");
  }
  return res.json();
}

export async function getReport(taskId: string): Promise<ReportResponse> {
  const res = await fetch(`${BASE_URL}/api/report/${taskId}`);
  if (!res.ok) {
    const err: ErrorResponse = await res.json();
    throw new Error(err.detail || err.error || "Fetch failed");
  }
  return res.json();
}

// ── Legacy PersonalQA API (kept for old chat components) ──

export interface Conversation {
  id: string;
  title: string;
  kb_name: string;
  created_at: string;
}

export async function uploadFile(file: File, kbName: string) {
  const form = new FormData();
  form.append("file", file);
  form.append("kb_name", kbName);
  const res = await fetch(`${BASE_URL}/api/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
  return res.json();
}

export async function listKBs(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/api/kb`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.knowledge_bases || [];
}

export async function createKB(name: string): Promise<void> {
  await fetch(`${BASE_URL}/api/kb`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function deleteKB(name: string): Promise<void> {
  await fetch(`${BASE_URL}/api/kb/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export async function listConversations(): Promise<Conversation[]> {
  const res = await fetch(`${BASE_URL}/api/conversations`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.conversations || [];
}

export async function deleteConversation(convId: string): Promise<void> {
  await fetch(`${BASE_URL}/api/conversations/${encodeURIComponent(convId)}`, { method: "DELETE" });
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
