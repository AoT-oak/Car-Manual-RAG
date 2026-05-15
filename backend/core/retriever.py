import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import time


class ChromaHybridRetriever:
    def __init__(self, vector_db, reranker_model_path: str):
        """
        专为 Chroma 向量库定制的混合检索与重排管道
        """
        self.vector_db = vector_db

        print("⚙️ 正在加载 Reranker 重排模型到 GPU...")
        self.reranker = CrossEncoder(reranker_model_path, device='cuda')

        print("📦 正在从 Chroma 提取知识库切片，构建 BM25 词频索引...")
        # 直接从 Chroma 底层获取所有持久化的文档文本
        db_data = self.vector_db.get()
        self.documents = db_data['documents']

        # 构建 BM25 索引
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ 多路召回引擎初始化完成！(共挂载 {len(self.documents)} 个知识切片)")

    def vector_search(self, query: str, top_k: int = 10) -> dict:
        """复用现有的 Chroma 进行语义向量检索"""
        # 返回的是 Document 对象列表
        docs = self.vector_db.similarity_search(query, k=top_k)
        # 用文档内容作为唯一键，记录排名
        return {doc.page_content: rank for rank, doc in enumerate(docs)}

    def bm25_search(self, query: str, top_k: int = 10) -> dict:
        """BM25 词频精准检索"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {self.documents[i]: rank for rank, i in enumerate(top_indices)}

    def rrf_fusion(self, bm25_ranks: dict, vector_ranks: dict, k: int = 60) -> list[str]:
        """RRF 融合逻辑，返回融合后的文档文本列表"""
        rrf_scores = {}
        all_docs = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        for doc in all_docs:
            score = 0.0
            if doc in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[doc])
            if doc in vector_ranks:
                score += 1.0 / (k + vector_ranks[doc])
            rrf_scores[doc] = score

        # 按得分从高到低排序，返回文本内容
        sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_docs

    def search_and_rerank(self, query: str, top_k: int = 10, final_k: int = 3):
        """核心业务逻辑：双路召回 -> RRF融合 -> Rerank重排"""
        start_time = time.time()

        # 1. 双路召回 (扩大池子，保证高覆盖)
        bm25_ranks = self.bm25_search(query, top_k)
        vector_ranks = self.vector_search(query, top_k)

        # 2. RRF 融合去重
        candidate_docs = self.rrf_fusion(bm25_ranks, vector_ranks)

        # 3. Cross-Encoder 精准重排
        cross_inp = [[query, doc] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(cross_inp)

        # 排序并截取最终结果
        best_indices = np.argsort(rerank_scores)[::-1][:final_k]
        final_results = [
            (candidate_docs[i], rerank_scores[i])
            for i in best_indices
        ]

        cost_time = time.time() - start_time
        print(f"⏱️ 混合检索耗时: {cost_time:.3f} 秒")
        return final_results