"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getTask, type TaskStatus } from "@/lib/api";

const STATUS_LABELS: Record<TaskStatus, string> = {
  QUEUED: "排队中",
  RUNNING: "审查中...",
  COMPLETED: "已完成",
  FAILED: "失败",
};

const PROGRESS_STEPS = [
  { key: "QUEUED", label: "任务已排队" },
  { key: "RUNNING", label: "Agent 正在分析" },
  { key: "COMPLETED", label: "报告已生成" },
];

export default function TaskPage() {
  const params = useParams();
  const router = useRouter();
  const taskId = params.id as string;

  const [status, setStatus] = useState<TaskStatus>("QUEUED");
  const [error, setError] = useState("");
  const [pollCount, setPollCount] = useState(0);

  const poll = useCallback(async () => {
    try {
      const task = await getTask(taskId);
      setStatus(task.status);
      setPollCount((c) => c + 1);

      if (task.status === "COMPLETED") {
        router.push(`/report/${taskId}`);
      } else if (task.status === "FAILED") {
        setError(task.error || "审查失败，原因未知。");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "获取任务状态失败");
    }
  }, [taskId, router]);

  useEffect(() => {
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [poll]);

  const activeStepIndex = PROGRESS_STEPS.findIndex(
    (s) => s.key === status
  );

  return (
    <div className="min-h-screen bg-[oklch(0.975,0.008,85)] flex flex-col">
      <header className="border-b border-border/40 bg-card/80 backdrop-blur-sm">
        <div className="max-w-2xl mx-auto px-6 py-4">
          <h1 className="text-lg font-semibold text-foreground tracking-tight">
            Code Review Agent
          </h1>
        </div>
      </header>

      <main className="flex-1 max-w-2xl mx-auto px-6 py-16 w-full">
        <Card>
          <CardHeader>
            <CardTitle>审查进行中</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Task ID */}
            <p className="text-xs text-muted-foreground font-mono">
              任务编号: {taskId}
            </p>

            {/* Progress steps */}
            <div className="space-y-3">
              {PROGRESS_STEPS.map((step, i) => {
                const isActive = i === activeStepIndex;
                const isDone = i < activeStepIndex || status === "COMPLETED";
                const isFailed = status === "FAILED" && i === activeStepIndex;

                return (
                  <div key={step.key} className="flex items-center gap-3">
                    {/* 状态指示器 */}
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium shrink-0 transition-colors ${
                        isFailed
                          ? "bg-destructive/15 text-destructive"
                          : isDone
                          ? "bg-accent/15 text-accent"
                          : isActive
                          ? "bg-accent text-white animate-pulse"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {isFailed ? "!" : isDone ? "✓" : i + 1}
                    </div>
                    {/* 步骤说明 */}
                    <div className="flex-1">
                      <p
                        className={`text-sm ${
                          isActive || isFailed
                            ? "font-medium text-foreground"
                            : "text-muted-foreground"
                        }`}
                      >
                        {step.label}
                      </p>
                      {isActive && status === "RUNNING" && (
                        <p className="text-xs text-muted-foreground mt-0.5">
                          Reviewer 和 Researcher Agent 正在分析代码... (第 {pollCount} 次轮询)
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* 状态标签 */}
            <div className="text-center">
              <span
                className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${
                  status === "FAILED"
                    ? "bg-destructive/10 text-destructive"
                    : status === "COMPLETED"
                    ? "bg-accent/10 text-accent"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {STATUS_LABELS[status]}
              </span>
            </div>

            {/* 错误信息 */}
            {error && (
              <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
                <p className="text-sm text-destructive font-medium">出错了</p>
                <p className="text-sm text-muted-foreground mt-1">{error}</p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-3"
                  onClick={() => router.push("/")}
                >
                  返回首页
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
