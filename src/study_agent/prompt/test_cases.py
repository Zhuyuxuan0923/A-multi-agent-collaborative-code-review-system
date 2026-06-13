"""Prompt 评测用测试用例 —— 50 条客服消息分类任务。

每条用例包含：
  - id:       用例编号（TC001 ~ TC050）
  - text:     模拟用户发来的客服消息
  - expected: 正确的分类结果（category / priority / sentiment）

分类体系：
  category:
    billing    → 账单、付款、退款、订阅扣费
    technical  → 技术故障、Bug、报错、兼容性问题
    account    → 登录、密码、绑定、注销、权限
    product    → 产品功能、价格、规格、使用方法咨询
    complaint  → 投诉、不满、要求赔偿、情绪宣泄

  priority:
    high   → 涉及金钱/安全/无法使用，需立即处理
    medium → 影响使用但不阻塞，正常处理
    low    → 一般咨询，不紧急

  sentiment:
    negative → 生气、沮丧、失望、焦虑
    neutral  → 客观描述，无明显情绪
    positive → 满意、感谢、喜欢、认可

用例设计覆盖：
  - 5 个类别各约 10 条
  - 短消息（< 10 字）、长消息（> 100 字）
  - 边界 case：模糊表述、混合情感、中英混杂
  - 优先级分布：high ~15 条 / medium ~20 条 / low ~15 条
"""

from study_agent.prompt.evaluator import EvalCase

CLASSIFICATION_CASES: list[EvalCase] = [
    # ═══════════════════════════════════════════════════════════
    # billing — 账单/付款/退款（10 条）
    # ═══════════════════════════════════════════════════════════
    EvalCase(
        "TC001",
        "我的账户刚才被扣了两次钱，能帮我查一下吗？",
        {"category": "billing", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC002",
        "这个月的账单比上个月多了300块，我没改过套餐啊",
        {"category": "billing", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC003",
        "请问怎么查看我的订阅到期时间？",
        {"category": "billing", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC004",
        "我想升级到高级版，请问年付有优惠吗？",
        {"category": "billing", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC005",
        "刚才那笔付款显示失败了，但钱已经从我卡里扣了，怎么回事？",
        {"category": "billing", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC006",
        "你们这个自动续费怎么关？我根本没同意续费",
        {"category": "billing", "priority": "medium", "sentiment": "negative"},
    ),
    EvalCase(
        "TC007",
        "刚付了年费，感受到专业版的功能确实强很多，值！",
        {"category": "billing", "priority": "low", "sentiment": "positive"},
    ),
    EvalCase(
        "TC008",
        "能开发票吗？我们公司需要报销",
        {"category": "billing", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC009",
        "我上个月明明取消了订阅，怎么这个月又扣费了？给我退回来！",
        {"category": "billing", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC010",
        "PayPal支付一直pending，换了两张信用卡都不行",
        {"category": "billing", "priority": "high", "sentiment": "negative"},
    ),
    # ═══════════════════════════════════════════════════════════
    # technical — 技术故障/Bug/报错（10 条）
    # ═══════════════════════════════════════════════════════════
    EvalCase(
        "TC011",
        "App 一直闪退，打开三秒就崩，重启手机也没用",
        {"category": "technical", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC012",
        "上传文件时提示Error 500，试了三次都是这样",
        {"category": "technical", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC013",
        "API 文档里这个 endpoint 的 base URL 是什么？没找到",
        {"category": "technical", "priority": "medium", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC014",
        "网页加载特别慢，其他网站都正常，就你们家这样",
        {"category": "technical", "priority": "medium", "sentiment": "negative"},
    ),
    EvalCase(
        "TC015",
        "新版界面找不到导出按钮了，之前明明在右上角的",
        {"category": "technical", "priority": "medium", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC016",
        "iOS 18 更新后你们的 App 打不开了，什么时候适配？",
        {"category": "technical", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC017",
        "用 Chrome 没问题，但 Safari 上排版全乱了",
        {"category": "technical", "priority": "medium", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC018",
        "数据导出的 CSV 文件中文全变成乱码了",
        {"category": "technical", "priority": "medium", "sentiment": "negative"},
    ),
    EvalCase(
        "TC019",
        "你们的搜索功能太好用了，响应速度也很快，赞一个",
        {"category": "technical", "priority": "low", "sentiment": "positive"},
    ),
    EvalCase(
        "TC020",
        "Webhook 不触发了，昨天的配置没动过，今天突然不work",
        {"category": "technical", "priority": "high", "sentiment": "negative"},
    ),
    # ═══════════════════════════════════════════════════════════
    # account — 账户管理/登录/密码（10 条）
    # ═══════════════════════════════════════════════════════════
    EvalCase(
        "TC021",
        "密码忘了，点了忘记密码但一直收不到重置邮件",
        {"category": "account", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC022",
        "怎么换绑手机号？之前那个号已经不用了",
        {"category": "account", "priority": "medium", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC023",
        "有人从陌生设备登录了我的账号，是不是被盗了？",
        {"category": "account", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC024",
        "我注册时用的微信登录，现在想改成邮箱登录可以吗？",
        {"category": "account", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC025",
        "为什么我的账号被冻结了？我没违反任何规定啊",
        {"category": "account", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC026",
        "帮我把这个账号注销掉，我不需要了",
        {"category": "account", "priority": "medium", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC027",
        "双因素认证怎么关？每次都输验证码太麻烦了",
        {"category": "account", "priority": "low", "sentiment": "negative"},
    ),
    EvalCase(
        "TC028",
        "换个头像怎么总是上传失败？文件大小和格式都符合要求",
        {"category": "account", "priority": "medium", "sentiment": "negative"},
    ),
    EvalCase(
        "TC029",
        "我想把团队其他成员加到我的 workspace 里，怎么操作？",
        {"category": "account", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC030",
        "刚注册成功，界面很清爽，引导也很友好，好评",
        {"category": "account", "priority": "low", "sentiment": "positive"},
    ),
    # ═══════════════════════════════════════════════════════════
    # product — 产品功能/使用方法咨询（10 条）
    # ═══════════════════════════════════════════════════════════
    EvalCase(
        "TC031",
        "你们支持批量导入吗？我们有2000多条数据要迁移过来",
        {"category": "product", "priority": "medium", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC032",
        "免费版和付费版具体差在哪里？有对比表吗？",
        {"category": "product", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC033",
        "这个功能有没有API接口？我想集成到我们的内部系统里",
        {"category": "product", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC034",
        "AI分析功能一次最多能处理多少条数据？有没有上限？",
        {"category": "product", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC035",
        "和竞品XX比起来，你们的核心优势是什么？",
        {"category": "product", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC036",
        "刚试用了智能推荐功能，准确率高得惊人，准备推荐给同事",
        {"category": "product", "priority": "low", "sentiment": "positive"},
    ),
    EvalCase(
        "TC037",
        "有没有数据备份功能？我怕数据丢了",
        {"category": "product", "priority": "medium", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC038",
        "支持钉钉集成吗？我们公司用的钉钉",
        {"category": "product", "priority": "low", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC039",
        "你们的移动端什么时候上线？等了好久了",
        {"category": "product", "priority": "medium", "sentiment": "neutral"},
    ),
    EvalCase(
        "TC040",
        "客服响应速度非常快，功能也很全，比我之前用的好太多了",
        {"category": "product", "priority": "low", "sentiment": "positive"},
    ),
    # ═══════════════════════════════════════════════════════════
    # complaint — 投诉/不满/要求赔偿（10 条）
    # ═══════════════════════════════════════════════════════════
    EvalCase(
        "TC041",
        "等了三天都没人回复我的工单，你们客服是摆设吗？",
        {"category": "complaint", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC042",
        "产品描述里说支持AI翻译，买了才发现根本没有，这是欺诈！",
        {"category": "complaint", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC043",
        "客服态度极其恶劣，工号1024，你查一下，我要投诉他",
        {"category": "complaint", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC044",
        "系统故障导致我丢了三个月的项目数据，你们怎么赔偿？",
        {"category": "complaint", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC045",
        "服务越来越差，以前响应很快现在一个工单要等一周",
        {"category": "complaint", "priority": "medium", "sentiment": "negative"},
    ),
    EvalCase(
        "TC046",
        "昨天在微博上看到有人吐槽你们泄露用户数据，是真的吗？",
        {"category": "complaint", "priority": "high", "sentiment": "negative"},
    ),
    EvalCase(
        "TC047",
        "功能砍了一个又一个，价格却一直涨，考虑换别的了",
        {"category": "complaint", "priority": "medium", "sentiment": "negative"},
    ),
    EvalCase(
        "TC048",
        "升级后界面变得好难用，能不能退回旧版？现在这个设计反人类",
        {"category": "complaint", "priority": "medium", "sentiment": "negative"},
    ),
    EvalCase(
        "TC049",
        "虽然之前有些问题，但这次处理得很好，谢谢你们的补偿方案",
        {"category": "complaint", "priority": "medium", "sentiment": "positive"},
    ),
    EvalCase(
        "TC050",
        "作为三年老用户，眼看着你们从精品变成现在这样，真的失望",
        {"category": "complaint", "priority": "medium", "sentiment": "negative"},
    ),
]
