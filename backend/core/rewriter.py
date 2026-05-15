import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class IntentRewriter:
    def __init__(self):
        """初始化重写引擎，使用低 temperature 保证重写的稳定性"""
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("❌ 未找到 ZHIPUAI_API_KEY，请检查 .env 文件！")

        # 这里依然使用 flash 版本，因为它速度极快，不会给系统的总体响应时间增加太多负担
        self.llm = ChatZhipuAI(model="glm-4-flash", temperature=0.1, api_key=api_key)
        self._build_chain()

    def _build_chain(self):
        """构建重写提示词链条 (Prompt Engineering 的核心体现)"""

        template = """你是一个专业的垂直领域 RAG 检索词优化专家（Query Optimizer）。
你的任务是将用户随意的、口语化的提问，转化为最容易在官方说明书中检索到的“书面化、结构化术语”。

【当前上下文】
用户正在查询的手册名称是：《{doc_name}》

【重写规则】
1. 语言对齐：请通过手册名称推断该手册的主要语言。如果用户的提问语言与手册语言不同，请务必将其翻译为手册的语言。
2. 消除口语：将诸如“咋办”、“怎么弄”、“坏了”等口语，替换为“操作方法”、“故障排除”、“维修指南”等书面词汇。
3. 术语扩展：提取核心实体，并补充 1-2 个高度相关的同义专业术语，用空格隔开，增加检索命中率。（例如：“大灯”重写为“前照灯 大灯 灯光控制”）。
4. 格式要求：直接输出重写后的检索关键词字符串，**严禁输出任何解释性文字或标点符号**。

【示例】
手册：《比亚迪汉DM说明书》 | 用户提问：How to turn on the headlights?
输出：前照灯 开启操作 大灯控制 灯光开关

手册：《Tesla Model 3 Owner's Manual》 | 用户提问：雨刮器咋喷水？
输出：Windshield wipers washer fluid activation spray

【实际任务】
用户提问：{user_query}
输出："""

        prompt = ChatPromptTemplate.from_template(template)

        # 构建轻量级的 LCEL (LangChain Expression Language) 链条
        self.rewrite_chain = prompt | self.llm | StrOutputParser()

    def rewrite(self, user_query: str, doc_name: str) -> str:
        """
        执行重写操作
        :param user_query: 用户的原始提问
        :param doc_name: 当前查询的手册名称
        :return: 优化后的检索词
        """
        try:
            # 调用大模型进行意图改写
            optimized_query = self.rewrite_chain.invoke({
                "doc_name": doc_name,
                "user_query": user_query
            })
            return optimized_query.strip()
        except Exception as e:
            print(f"⚠️ Query Rewrite 失败，降级使用原始提问。错误信息: {e}")
            return user_query


# ================= 测试模块 =================
if __name__ == "__main__":
    print("🚀 正在启动 Query Rewriter 测试引擎...\n")
    rewriter = IntentRewriter()

    # 准备测试用例：涵盖跨语言、极度口语化、模糊意图三种极限场景
    test_cases = [
        {
            "doc_name": "比亚迪汉DM用户手册",
            "query": "How to turn on the headlights?",
            "desc": "场景 1: 英文提问 -> 中文手册 (跨语言)"
        },
        {
            "doc_name": "比亚迪汉DM用户手册",
            "query": "车子方向盘咋这么沉，是不是坏了？",
            "desc": "场景 2: 极度口语化/情绪化 -> 剥离情绪，转书面语"
        },
        {
            "doc_name": "Volvo XC90 Owner's Manual",
            "query": "自动泊车怎么开？",
            "desc": "场景 3: 中文提问 -> 英文手册 (跨语言)"
        }
    ]

    for i, tc in enumerate(test_cases):
        print(f"--- 测试用例 {i + 1}: {tc['desc']} ---")
        print(f"📘 手册语境: {tc['doc_name']}")
        print(f"🙋 原始提问: {tc['query']}")

        optimized = rewriter.rewrite(user_query=tc['query'], doc_name=tc['doc_name'])

        print(f"✨ 重写结果: {optimized}\n")