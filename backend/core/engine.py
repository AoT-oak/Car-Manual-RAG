import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .rewriter import IntentRewriter
from .retriever import ChromaHybridRetriever


class RAGEngine:
    def __init__(self):
        # 🌟 修改点 1：抽离硬编码，使用 os.getenv 读取环境变量
        # 如果其他人部署时没配置 .env，系统会默认拉取线上的 BAAI 模型库和当前目录建库
        self.bge_model_path = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-large-zh-v1.5")
        self.reranker_model_path = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
        self.db_base_dir = os.getenv("VECTOR_DB_DIR", "./vector_dbs")

        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("未找到 ZHIPUAI_API_KEY，请检查 .env 文件！")

        self.rewriter = IntentRewriter()
        self.llm = ChatZhipuAI(model="glm-4-flash", temperature=0.1, api_key=api_key)

        print("🧠 正在加载本地 BGE 向量模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.bge_model_path,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.active_db = None
        self.active_doc_name = ""
        self.hybrid_retriever = None

        if not os.path.exists(self.db_base_dir):
            os.makedirs(self.db_base_dir)

    def get_available_manuals(self) -> list:
        """获取所有已缓存的手册列表"""
        if not os.path.exists(self.db_base_dir):
            return []
        manuals = []
        for folder in os.listdir(self.db_base_dir):
            if folder.startswith("db_") and os.path.isdir(os.path.join(self.db_base_dir, folder)):
                manuals.append(folder.replace("db_", ""))
        return manuals

    def clean_text(self, text: str) -> str:
        """文本清洗"""
        text = re.sub(r'([^\n。！？])\n([^\n])', r'\1 \2', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def process_and_build_db(self, pdf_path: str, doc_name: str):
        """解析上传的 PDF 并构建 Chroma 向量库"""
        db_path = os.path.join(self.db_base_dir, f"db_{doc_name}")

        # 如果已经存在就不重复构建了
        if os.path.exists(db_path) and os.listdir(db_path):
            return

        print(f"⏳ 正在解析并构建知识库: {doc_name}...")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        for doc in docs:
            doc.page_content = self.clean_text(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100,
            separators=["\n\n", "。", "！", "？", "\n", " ", ""]
        )

        raw_chunks = text_splitter.split_documents(docs)
        valid_chunks = [c for c in raw_chunks if len(c.page_content) >= 20]

        Chroma.from_documents(
            documents=valid_chunks,
            embedding=self.embeddings,
            persist_directory=db_path
        )
        print(f"✅ 知识库构建完成！")

    def load_knowledge_base(self, doc_name: str):
        if self.active_doc_name == doc_name and self.active_db is not None:
            return
        db_path = os.path.join(self.db_base_dir, f"db_{doc_name}")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"未找到对应的知识库: {doc_name}")

        self.active_db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.active_doc_name = doc_name
        self.hybrid_retriever = ChromaHybridRetriever(self.active_db, self.reranker_model_path)

    def generate_response(self, query: str, doc_name: str) -> dict:
        self.load_knowledge_base(doc_name)
        optimized_query = self.rewriter.rewrite(query, doc_name)
        reranked_results = self.hybrid_retriever.search_and_rerank(optimized_query, top_k=10, final_k=3)

        context_str = ""
        for doc, score in reranked_results:
            context_str += f"[相关度打分: {score:.4f}]\n{doc}\n\n"

        # 🌟 修改点 2：补充了全局已知信息注入，解决丢失常识背景的问题
        template = f"""你是一个专业的汽车技术专家，正在为《{doc_name}》提供咨询服务。
请结合【全局已知信息】和【背景知识】进行回答。

【全局已知信息】
当前用户咨询的汽车具体车型即为：《{doc_name}》。

【输出约束 - 极其重要 (CRITICAL)】
1. 强制语言对齐 (Strict Language Alignment)：你必须严格检测【车主提问】的语种！
   - IF the user asks in English, you MUST answer entirely in English.
   - 如果用户用中文提问，你必须全程使用中文回答。
2. 真实性兜底：若背景知识和全局已知信息中都无法推断出答案，请直接回复：“在《{doc_name}》中未找到相关细节。”（需使用与提问相同的语言翻译这句话）。

【背景知识】
{{context}}

【车主提问】
{{question}}
"""
        prompt = ChatPromptTemplate.from_template(template)
        gen_chain = prompt | self.llm | StrOutputParser()

        answer = gen_chain.invoke({"context": context_str, "question": query})

        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "answer": answer,
            "reference_docs": [{"content": doc, "score": float(score)} for doc, score in reranked_results]
        }