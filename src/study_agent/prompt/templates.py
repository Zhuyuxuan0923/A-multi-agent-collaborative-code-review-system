"""
Prompt 模板管理器 — 加载、管理、渲染 Jinja2 模板文件

核心概念：
  .j2 文件 = Jinja2 模板文件（和 .html 一样本质是文本，.j2 是约定俗成的后缀）
  FileSystemLoader = Jinja2 的"文件读取器"，从指定目录加载模板
  Environment = 模板引擎的"运行环境"，持有所有配置和已加载的模板

使用方式：
  manager = PromptManager("src/study_agent/prompt/templates")
  prompt = manager.render("qa_with_rag", role="技术专家", question="什么是 RAG？")
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


class PromptManager:
    """
    Prompt 模板管理器

    三大职责：
      1. 从 templates/ 目录加载所有 .j2 模板文件
      2. 提供模板级别的变量渲染
      3. 集中管理 so 你不需要每次手动 Environment → Loader → get_template
    """

    def __init__(self, template_dir: str | Path):
        """
        template_dir: 存放 .j2 模板文件的目录路径

        FileSystemLoader 做了什么？
          它告诉 Jinja2："去这个目录找模板文件"
          比如你调用 manager.render("qa_with_rag", ...)
          → Jinja2 去 template_dir/qa_with_rag.j2 找文件
          → 解析、渲染、返回结果
        """
        self.template_dir = Path(template_dir)
        if not self.template_dir.exists():
            raise FileNotFoundError(f"模板目录不存在: {self.template_dir}")

        # Environment 是 Jinja2 的"引擎核心"
        # FileSystemLoader 是它的"眼睛"——告诉它去哪找文件
        self._env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            # 下面三个参数你现在不需要改，知道有就行：
            trim_blocks=True,  # 删除 {% %} 后面的第一个换行符（让输出更整洁）
            lstrip_blocks=True,  # 删除 {% %} 前面的空白（同上）
            keep_trailing_newline=True,  # 保留文件末尾的换行符
        )

    def render(self, template_name: str, **variables: object) -> str:
        """
        加载并渲染一个模板

        参数：
          template_name: 模板文件名，不含 .j2 后缀（如 "qa_with_rag" 而不是 "qa_with_rag.j2"）
          **variables: 模板里 {{ }} 占位符的值（如 role="老师", task="翻译"）

        返回：渲染后的完整 prompt 字符串

        举例：
          manager.render("qa_with_rag", role="老师", task="翻译")
          → 加载 templates/qa_with_rag.j2，把 {{ role }} 替换成 "老师"，{{ task }} 替换成 "翻译"
        """
        template: Template = self._env.get_template(f"{template_name}.j2")
        result: str = template.render(**variables)
        return result

    def list_templates(self) -> list[str]:
        """列出所有可用的模板名（不含 .j2 后缀），用于调试和检查"""
        return [f.stem for f in self.template_dir.glob("*.j2")]

    def get_template_source(self, template_name: str) -> str:
        """查看某个模板的原始内容（不含 .j2 后缀），用于调试时看模板长什么样"""
        template_path = self.template_dir / f"{template_name}.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"模板文件不存在: {template_path}")
        return template_path.read_text(encoding="utf-8")
