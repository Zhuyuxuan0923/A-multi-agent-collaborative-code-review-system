"""
Few-Shot 示例管理器 — 存储、筛选、格式化示例

为什么需要单独管理示例？
  1. 示例和模板是两回事——模板是骨架，示例是血肉
  2. 同一个示例可以被多个模板复用
  3. 示例数量会累积（50+ 个），需要按标签筛选而非全部塞进去

一张表看懂 Few-Shot 的核心概念：

  概念       |  含义                          |  类比
  ──────────┼────────────────────────────────┼────────────
  Zero-Shot  |  不给例子，直接问                |  对新实习生说"去写个报告"
  One-Shot   |  给 1 个输入→输出范例            |  给他看一份"上次写了什么"
  Few-Shot   |  给 2-5 个范例                   |  给他看上个月的 3 份报告
  Many-Shot  |  给 10+ 个范例（可能塞满上下文）  |  把整个文件夹扔给他

我们主要用 Few-Shot（3 个例子），因为：
  - 1 个不够：模型可能模仿不到规律
  - 5+ 个太多：吃 token、可能混淆
  - 3 个刚好：给模型一个"明显的模式"，又不浪费上下文
"""


class FewShotExample:
    """单个 Few-Shot 示例：一对 input → output，附带标签用于筛选"""

    def __init__(self, input_text: str, output_text: str, tags: list[str] | None = None):
        self.input = input_text
        self.output = output_text
        self.tags = tags or []

    def __repr__(self) -> str:
        tags_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"FewShotExample(input={self.input[:30]}...){tags_str}"


class FewShotManager:
    """
    示例仓库 + 筛选器

    职责：
      1. 存储示例
      2. 按标签筛选（比如只选"代码审查"相关示例）
      3. 格式化输出（把示例列表转成 prompt 里能直接用的文本）
    """

    def __init__(self) -> None:
        self._examples: list[FewShotExample] = []

    def add(self, input_text: str, output_text: str, tags: list[str] | None = None) -> None:
        """添加一个示例到仓库"""
        self._examples.append(FewShotExample(input_text, output_text, tags))

    def add_batch(self, examples: list[tuple[str, str, list[str] | None]]) -> None:
        """
        批量添加示例

        examples 格式: [(input, output, [tags]), ...]
        例如:
          manager.add_batch([
              ("1+1=?", "2", ["math", "simple"]),
              ("导数定义", "极限定义式", ["math", "calculus"]),
          ])
        """
        for input_text, output_text, tags in examples:
            self.add(input_text, output_text, tags)

    @property
    def count(self) -> int:
        return len(self._examples)

    def filter_by_tags(self, tags: list[str]) -> list[FewShotExample]:
        """
        按标签筛选——只要示例的 tags 和给定 tags 有任意交集就选中

        为什么用"交集"而不是"完全匹配"？
          一个示例可能标了 ["代码审查", "Python", "安全"]
          你搜索 ["Python"] → 应该匹配它
          你搜索 ["安全", "Python"] → 也应该匹配它
        """
        if not tags:
            return list(self._examples)
        tag_set = set(tags)
        return [ex for ex in self._examples if tag_set & set(ex.tags)]

    def pick(self, tags: list[str] | None = None, max_count: int = 3) -> list[FewShotExample]:
        """
        筛选并取最多 max_count 个示例

        为什么限制数量？
          - 每个示例都在消耗 token（input + output 都是 token）
          - 上下文窗口有限，3 个示例通常是最优数量
          - 选太多 = token 费用高 + 模型可能被过多例子搞混
        """
        candidates = self.filter_by_tags(tags or [])
        return candidates[:max_count]

    def format(self, examples: list[FewShotExample]) -> str:
        """
        把示例列表格式化成 prompt 能用的纯文本

        输出格式:
          示例 1:
          输入: xxx
          输出: yyy

          示例 2:
          输入: xxx
          输出: yyy
        """
        if not examples:
            return ""
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"示例 {i}:")
            lines.append(f"输入: {ex.input}")
            lines.append(f"输出: {ex.output}")
            lines.append("")  # 空行分隔
        return "\n".join(lines)
