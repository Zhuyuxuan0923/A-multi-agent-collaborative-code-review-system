import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Code Review Agent — AI 代码审查系统",
  description: "提交代码或 PR URL，多 Agent 协作审查，生成结构化报告。",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
