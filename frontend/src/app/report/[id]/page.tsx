"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import MarkdownRenderer from "@/components/chat/markdown-renderer";
import { getReport, type ReportResponse } from "@/lib/api";

const SEVERITY_LABELS: Record<string, string> = {
  Critical: "严重",
  Important: "重要",
  Minor: "建议",
};

const SEVERITY_COLORS: Record<string, string> = {
  Critical: "bg-red-100 text-red-800 border-red-200",
  Important: "bg-amber-100 text-amber-800 border-amber-200",
  Minor: "bg-blue-100 text-blue-800 border-blue-200",
};

export default function ReportPage() {
  const params = useParams();
  const router = useRouter();
  const taskId = params.id as string;

  const [report, setReport] = useState<ReportResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const data = await getReport(taskId);
        setReport(data);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "加载报告失败");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [taskId]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[oklch(0.975,0.008,85)]">
        <p className="text-muted-foreground text-sm">正在加载报告...</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4 bg-[oklch(0.975,0.008,85)]">
        <p className="text-destructive text-sm">{error || "报告未找到"}</p>
        <Button variant="outline" onClick={() => router.push("/")}>
          返回首页
        </Button>
      </div>
    );
  }

  const criticalCount = report.issues.filter((i) => i.severity === "Critical").length;
  const importantCount = report.issues.filter((i) => i.severity === "Important").length;
  const minorCount = report.issues.filter((i) => i.severity === "Minor").length;

  return (
    <div className="min-h-screen bg-[oklch(0.975,0.008,85)]">
      {/* Header */}
      <header className="border-b border-border/40 bg-card/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-foreground tracking-tight">
              审查报告
            </h1>
            <p className="text-xs text-muted-foreground font-mono mt-0.5">
              {taskId}
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={() => router.push("/")}>
            新建审查
          </Button>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        {/* 总评分 */}
        <Card>
          <CardHeader>
            <CardTitle>总体评分</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <div
                className={`text-4xl font-bold font-mono ${
                  report.score < 0
                    ? "text-muted-foreground"
                    : report.score >= 8
                    ? "text-green-600"
                    : report.score >= 6
                    ? "text-amber-600"
                    : "text-red-600"
                }`}
              >
                {report.score >= 0 ? `${report.score}/10` : "N/A"}
              </div>
              <div className="flex gap-3 text-xs">
                <span className="px-2 py-1 rounded border border-red-200 bg-red-50 text-red-700">
                  {criticalCount} 严重
                </span>
                <span className="px-2 py-1 rounded border border-amber-200 bg-amber-50 text-amber-700">
                  {importantCount} 重要
                </span>
                <span className="px-2 py-1 rounded border border-blue-200 bg-blue-50 text-blue-700">
                  {minorCount} 建议
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 问题列表 */}
        {report.issues.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>发现的问题 ({report.issues.length})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {report.issues.map((issue, i) => (
                <div
                  key={i}
                  className="border border-border/50 rounded-lg p-4 space-y-2"
                >
                  <div className="flex items-start justify-between gap-3">
                    <h3 className="text-sm font-semibold text-foreground">
                      {issue.title}
                    </h3>
                    <span
                      className={`shrink-0 text-[11px] px-2 py-0.5 rounded border font-medium ${
                        SEVERITY_COLORS[issue.severity] || ""
                      }`}
                    >
                      {SEVERITY_LABELS[issue.severity] || issue.severity}
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {issue.description}
                  </p>
                  {issue.file_path && (
                    <p className="text-xs text-muted-foreground font-mono">
                      {issue.file_path}
                      {issue.line ? `:${issue.line}` : ""}
                    </p>
                  )}
                  {issue.suggestion && (
                    <div className="bg-accent/5 border border-accent/15 rounded-md p-3">
                      <p className="text-xs font-medium text-accent mb-1">
                        修复建议
                      </p>
                      <p className="text-sm text-foreground/80">
                        {issue.suggestion}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* 最佳实践 */}
        {report.research.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>最佳实践与参考</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {report.research.map((item, i) => (
                  <li key={i} className="text-sm text-muted-foreground flex gap-2">
                    <span className="text-accent shrink-0">-</span>
                    {item}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {/* 完整 Markdown 报告 */}
        {report.report_md && (
          <Card>
            <CardHeader>
              <CardTitle>完整报告</CardTitle>
            </CardHeader>
            <CardContent>
              <MarkdownRenderer>{report.report_md}</MarkdownRenderer>
            </CardContent>
          </Card>
        )}
      </main>

      <footer className="border-t border-border/40 py-4 text-center text-xs text-muted-foreground">
        生成时间: {new Date(report.created_at).toLocaleString("zh-CN")}
      </footer>
    </div>
  );
}
