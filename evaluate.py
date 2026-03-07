"""
评估模块
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import pandas as pd
from datasets import Dataset
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main import RecipeRAGSystem

class RagasEmbeddingAdapter:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    # --- 异步方法 (用于 Testset 生成和部分评估) ---
    async def embed_text(self, text: str):
        return await self.langchain_embeddings.aembed_query(text)

    async def embed_texts(self, texts: list[str]):
        return await self.langchain_embeddings.aembed_documents(texts)

    # --- 同步方法 (用于 AnswerRelevancy 等指标计算) ---
    def embed_query(self, text: str):
        return self.langchain_embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]):
        return self.langchain_embeddings.embed_documents(texts)

# -----------------------------------
# 环境与 API 初始化
# -----------------------------------

if not os.getenv("MOONSHOT_API_KEY"):
    raise ValueError("请设置 MOONSHOT_API_KEY")

openai_client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

judge_llm = llm_factory(
    model="moonshot-v1-32k", 
    client=openai_client,
    temperature=0,
    max_tokens=8192
)

# 适配版本地 Embedding
base_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
ragas_embeddings = RagasEmbeddingAdapter(base_embeddings)

# -----------------------------------
# 评估核心逻辑
# -----------------------------------
def start_eval():
    print("🔄 正在初始化 RAG 系统并构建知识库...")
    rag_system = RecipeRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base()

    # --- 加载testset.csv ---
    csv_path = "./evaluation/testset.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 错误：未找到测试集文件 {csv_path}，请先运行生成脚本。")
        return

    test_df = pd.read_csv(csv_path)
    print(f"✅ 成功加载测试集，共 {len(test_df)} 道题目。")

    results_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": []
    }

    print("🏃 正在调用你的 RAG 系统生成回答...")
    for index, row in test_df.iterrows():
        q = row['user_input']
        gt = row['reference']

        print(f"[{index+1}/{len(test_df)}] 🔍 正在提问: {q}")

        # 调用main.py 里的 RAG 逻辑
        out = rag_system.get_answer_and_context(q)
        answer = out["answer"]

        # 提取检索到的原始内容片段
        # 修复后的代码：增加兼容性检查
        contexts = []
        for doc in out["contexts"]:
            if hasattr(doc, 'page_content'):
                # 如果是 LangChain 的 Document 对象
                contexts.append(doc.page_content)
            else:
                # 如果已经是字符串字符串或其他格式
                contexts.append(str(doc))

        results_data["user_input"].append(q)
        results_data["response"].append(answer)
        results_data["retrieved_contexts"].append(contexts)
        results_data["reference"].append(gt)

    # 转换为 Ragas 评估所需的数据格式
    dataset = Dataset.from_dict(results_data)
 
    # 实例化指标 
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    print("⚖️ RAGAS 正在进行智能评分...")
    
    # 配置LLM，防止因RPM太小报错
    config = RunConfig(max_workers=1, timeout=300)
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=ragas_embeddings, 
        run_config=config
    )

    # --- 报告导出 ---
    print("\n" + "="*120)
    print("📊 RAG 评估得分总览：")
    print(result)
    print("="*120)

    # 保存详细结果到 CSV
    report_df = result.to_pandas()
    report_path = "./evaluation/eval_report.csv"
    report_df.to_csv(report_path, index=False, encoding='utf_8_sig')
    print(f"💾 详细评分报告已保存至: {report_path}")

if __name__ == "__main__":
    start_eval()