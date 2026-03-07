"""
检索优化模块
"""

import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document], model_name: str = 'BAAI/bge-reranker-base'):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        logger.info(f"正在加载重排模型: {model_name}...")
        try:
            # 必须将模型赋值给 self.rerank_model
            self.rerank_model = CrossEncoder(model_name)
            logger.info(f"重排模型加载成功")
        except Exception as e:
            logger.error(f"重排模型加载失败: {e}")
            raise
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )



        logger.info("检索器设置完成")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF/Cross-Encoder 重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        # reranked_docs = self._rrf_rerank(vector_docs, bm25_docs) # RRF 重排
        reranked_docs = self._cross_encoder_rerank(query, vector_docs, bm25_docs) # Cross-Encoder 重排
        return reranked_docs[:top_k]


    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """
        # 先进行混合检索，获取更多候选
        docs = self.hybrid_search(query, top_k * 3)
        
        # 应用元数据过滤
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        
        return filtered_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs


    def _cross_encoder_rerank(self, query: str, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        """
        使用 Cross-Encoder 对文档进行深度语义重排
        """
        # 合并结果并去重（基于内容哈希）
        unique_docs_dict = {}
        for doc in vector_docs + bm25_docs:
            doc_id = hash(doc.page_content)
            if doc_id not in unique_docs_dict:
                unique_docs_dict[doc_id] = doc
        
        unique_docs = list(unique_docs_dict.values())
        
        if not unique_docs:
            return []

        # 准备 Cross-Encoder 的输入格式: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in unique_docs]
        
        # 计算得分
        logger.info(f"开始对 {len(unique_docs)} 个文档进行 Cross-Encoder 重排...")
        scores = self.rerank_model.predict(pairs)

        # 将得分存入文档元数据并排序
        for i, score in enumerate(scores):
            unique_docs[i].metadata['rerank_score'] = float(score)

        # 按得分从大到小排序
        reranked_docs = sorted(unique_docs, key=lambda x: x.metadata['rerank_score'], reverse=True)

        logger.info("Cross-Encoder 重排完成")
        return reranked_docs


