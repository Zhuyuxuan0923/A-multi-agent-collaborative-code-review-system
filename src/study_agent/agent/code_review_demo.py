"""
代码审查多 Agent 系统演示。

运行方式: python -m src.study_agent.agent.code_review_demo

流程:
  1. Orchestrator 接收待审查代码
  2. 并行调用 Reviewer (找问题) + Researcher (查最佳实践)
  3. Reporter 整合两方结果生成 Markdown 报告
  4. 输出报告到控制台 + 保存到文件
"""

import os
import sys
from datetime import datetime

# 确保项目根目录在 sys.path 中
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from study_agent.agent.code_review_agents import CodeReviewOrchestrator
from study_agent.llm.client import LLMClient

# ============================================================
# 待审查的示例代码 (故意包含多种问题)
# ============================================================

SAMPLE_CODE_PYTHON = """
import sqlite3
import os

ADMIN_PASSWORD = "admin123"

def getUserData(userId):
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE id = " + userId
    result = conn.execute(query).fetchall()
    return result

def getUsersWithOrders():
    users = getUserData("1")
    result = []
    for user in users:
        orders = db.execute("SELECT * FROM orders WHERE user_id = " + str(user[0]))
        result.append({"user": user, "orders": orders})
    return result

def saveFile(filename, content):
    f = open("/tmp/" + filename, "w")
    f.write(content)

def processRequest(req):
    data = req["data"]
    value = data.upper()
    return {"status": "ok", "value": value}
"""

SAMPLE_CODE_JS = """
const express = require('express');
const mysql = require('mysql');
const app = express();

const DB_PASSWORD = "mypassword123";

app.get('/user/:id', (req, res) => {
    const id = req.params.id;
    const query = "SELECT * FROM users WHERE id = " + id;
    db.query(query, (err, result) => {
        res.json(result);
    });
});

app.post('/login', (req, res) => {
    const { username, password } = req.body;
    const html = "<h1>Welcome " + username + "</h1>";
    res.send(html);
});

app.listen(3000);
"""


# ============================================================
# 主流程
# ============================================================


def main():
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    print(f"使用 LLM Provider: {provider}")
    print()

    # 初始化 LLM 客户端 (三个 Agent 共享一个 client)
    llm = LLMClient(provider=provider)

    # 初始化编排器
    orchestrator = CodeReviewOrchestrator(llm)

    # 选择审查目标
    code = SAMPLE_CODE_PYTHON.strip()
    language = "python"

    print("待审查代码:")
    print("-" * 40)
    # 添加行号显示
    for i, line in enumerate(code.split("\n"), 1):
        print(f"  {i:2d} | {line}")
    print("-" * 40)
    print()

    # 执行审查
    report = orchestrator.review(code, language, verbose=True)

    print()
    print("=" * 60)
    print("审查报告")
    print("=" * 60)
    print(report)

    # 保存报告
    output_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "data",
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"code_review_report_{timestamp}.md")

    # 构建完整报告 (含元数据)
    full_report = f"""\
# 代码审查报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 审查语言: {language}
> 审查引擎: Multi-Agent (Reviewer + Researcher + Reporter)
> LLM Provider: {provider}

---

{report}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\n报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
