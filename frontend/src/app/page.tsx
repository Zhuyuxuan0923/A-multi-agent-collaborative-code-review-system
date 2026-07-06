"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { submitReview, submitPRReview } from "@/lib/api";

type Tab = "code" | "pr";

export default function Home() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<Tab>("code");

  const [code, setCode] = useState("");
  const [language, setLanguage] = useState("python");

  const [prUrl, setPrUrl] = useState("");
  const [token, setToken] = useState("");

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleCodeSubmit = async () => {
    if (!code.trim()) {
      setError("请粘贴需要审查的代码。");
      return;
    }
    setError("");
    setIsSubmitting(true);
    try {
      const result = await submitReview(code, language);
      router.push(`/task/${result.task_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "提交失败");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePRSubmit = async () => {
    if (!prUrl.trim()) {
      setError("请输入 PR 地址。");
      return;
    }
    if (!token.trim()) {
      setError("请输入 GitHub Token。");
      return;
    }
    // 校验 PR URL 格式，防止非 GitHub URL 传入后端
    const prUrlPattern = /^https?:\/\/github\.com\/[^/]+\/[^/]+\/pull\/\d+\/?$/;
    if (!prUrlPattern.test(prUrl.trim())) {
      setError("PR 地址格式不正确。请输入合法的 GitHub PR URL，例如 https://github.com/owner/repo/pull/123");
      return;
    }
    setError("");
    setIsSubmitting(true);
    try {
      const result = await submitPRReview(prUrl, token);
      router.push(`/task/${result.task_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "提交失败");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[oklch(0.975,0.008,85)] flex flex-col">
      {/* Header */}
      <header className="border-b border-border/40 bg-card/80 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-foreground tracking-tight">
              Code Review Agent
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              多 Agent 协作的 AI 代码审查系统
            </p>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-4xl mx-auto px-6 py-12 w-full">
        <div className="mb-8">
          <h2 className="text-2xl font-serif font-bold text-foreground">
            提交代码审查
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            直接粘贴代码或提供 GitHub PR 地址，多 Agent 系统将自动分析并生成结构化审查报告。
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6 border-b border-border/40">
          <button
            onClick={() => { setActiveTab("code"); setError(""); }}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "code"
                ? "border-accent text-accent"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            粘贴代码
          </button>
          <button
            onClick={() => { setActiveTab("pr"); setError(""); }}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "pr"
                ? "border-accent text-accent"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            PR 地址
          </button>
        </div>

        {/* 粘贴代码 Tab */}
        {activeTab === "code" && (
          <Card>
            <CardHeader>
              <CardTitle>粘贴源代码</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                  编程语言
                </label>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="h-8 w-full min-w-0 rounded-lg border border-input bg-transparent px-2.5 py-1 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
                >
                  <option value="python">Python</option>
                  <option value="javascript">JavaScript</option>
                  <option value="typescript">TypeScript</option>
                  <option value="java">Java</option>
                  <option value="go">Go</option>
                  <option value="rust">Rust</option>
                  <option value="cpp">C++</option>
                  <option value="c">C</option>
                  <option value="csharp">C#</option>
                  <option value="ruby">Ruby</option>
                  <option value="php">PHP</option>
                  <option value="swift">Swift</option>
                  <option value="kotlin">Kotlin</option>
                  <option value="sql">SQL</option>
                  <option value="shell">Shell</option>
                  <option value="other">其他</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                  代码内容
                </label>
                <Textarea
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="在此粘贴需要审查的代码..."
                  rows={16}
                  className="font-mono text-sm"
                />
              </div>
              {error && (
                <p className="text-sm text-destructive bg-destructive/5 px-3 py-2 rounded-lg">
                  {error}
                </p>
              )}
              <Button
                onClick={handleCodeSubmit}
                disabled={isSubmitting}
                className="w-full"
              >
                {isSubmitting ? "提交中..." : "提交审查"}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* PR 地址 Tab */}
        {activeTab === "pr" && (
          <Card>
            <CardHeader>
              <CardTitle>提交 GitHub PR</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                  PR 地址
                </label>
                <Input
                  value={prUrl}
                  onChange={(e) => setPrUrl(e.target.value)}
                  placeholder="https://github.com/owner/repo/pull/123"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                  GitHub Token
                </label>
                <Input
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="ghp_xxxxxxxxxxxx"
                />
                <p className="text-[11px] text-muted-foreground mt-1">
                  Token 仅用于拉取 PR diff，存储前会做哈希处理。
                </p>
              </div>
              {error && (
                <p className="text-sm text-destructive bg-destructive/5 px-3 py-2 rounded-lg">
                  {error}
                </p>
              )}
              <Button
                onClick={handlePRSubmit}
                disabled={isSubmitting}
                className="w-full"
              >
                {isSubmitting ? "提交中..." : "提交 PR 审查"}
              </Button>
            </CardContent>
          </Card>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border/40 py-4 text-center text-xs text-muted-foreground">
        由多 Agent 系统驱动: Reviewer + Researcher + Reporter
      </footer>
    </div>
  );
}
