"""
生成测试集
"""

import os
import sys
import random
from openai import OpenAI

# 1. 处理路径：确保能导入父目录下的 rag_modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 导入自定义模块
try:
    from rag_modules import DataPreparationModule
except ImportError as e:
    print(f"❌ 导入失败：请确保上一级目录存在 rag_modules 文件夹")
    raise e

from langchain_huggingface import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory
from ragas.run_config import RunConfig

# ---------------------------------------------------------
# 3. 核心组件适配器
# ---------------------------------------------------------
class RagasEmbeddingAdapter:
    """解决异步接口不匹配：'HuggingFaceEmbeddings' -> 'embed_text'"""
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    async def embed_text(self, text: str):
        return await self.langchain_embeddings.aembed_query(text)

    async def embed_texts(self, texts: list[str]):
        return await self.langchain_embeddings.aembed_documents(texts)

# ---------------------------------------------------------
# 4. 初始化 API 与 模型
# ---------------------------------------------------------
if not os.getenv("MOONSHOT_API_KEY"):
    raise ValueError("请在环境变量中设置 MOONSHOT_API_KEY")

openai_client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

# 初始化 Kimi 裁判 (LLM)
judge_llm = llm_factory(
    model="moonshot-v1-32k", 
    client=openai_client,
    temperature=0.3 
)

# 初始化适配版 Embedding
base_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
ragas_embeddings = RagasEmbeddingAdapter(base_embeddings)

# ---------------------------------------------------------
# 5. 加载与生成逻辑
# ---------------------------------------------------------
def get_structured_chunks(path):
    print(f"🛠️ 正在调用 DataPreparationModule 处理结构化数据: {path}")
    prep_module = DataPreparationModule(data_path=path)
    prep_module.load_documents()
    chunks = prep_module.chunk_documents()
    stats = prep_module.get_statistics()
    print(f"📊 统计：加载了 {stats.get('total_documents')} 份食谱，生成了 {len(chunks)} 个分块")
    return chunks

def run_auto_generate():
    knowledge_path = "../data/dishes" 
    all_chunks = get_structured_chunks(knowledge_path)

    if not all_chunks:
        print("❌ 未找到文档块")
        return
    
    chunk_count = len(all_chunks)
    sample_size = 20
    start_idx = random.randint(0, chunk_count - sample_size)
    sample_chunks = all_chunks[start_idx: start_idx + sample_size]

    print("🤖 正在初始化 Ragas 出题器...")
    generator = TestsetGenerator(
        llm=judge_llm,
        embedding_model=ragas_embeddings
    )

    config = RunConfig(max_workers=1, timeout=240)
    
    
    print(f"✍️ 开始自动出题...")
    
    testset = generator.generate_with_langchain_docs(
        documents=sample_chunks, 
        testset_size=5,  
        run_config=config
    )

    # ---------------------------------------------------------
    # 6. 数据清洗：过滤掉非中文内容
    # ---------------------------------------------------------
    df = testset.to_pandas()
    
    # 定义一个简单的中文字符检测函数
    def is_chinese(text):
        if not isinstance(text, str): return False
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    # 过滤掉那些 user_input 或 reference 中不含中文的行
    initial_count = len(df)
    df = df[df['user_input'].apply(is_chinese)].copy()
    
    if len(df) < initial_count:
        print(f"🧹 已自动剔除 {initial_count - len(df)} 条纯英文/无效数据")

    # 提取并保存
    q_col = 'user_input' if 'user_input' in df.columns else 'question'
    gt_col = 'reference' if 'reference' in df.columns else 'ground_truth'
    
    final_df = df[[q_col, gt_col]].copy()
    final_df.columns = ['user_input', 'reference']


    output_file = "testset.csv"
    final_df.to_csv(output_file, index=False, encoding="utf_8_sig")
    
    print("\n" + "="*60)
    print(f"🎉 纯中文测试集生成成功！")
    print(f"📁 保存路径: {os.path.abspath(output_file)}")
    print(f"📝 结果预览（仅展示前2条）:\n{final_df.head(2)}")
    print("="*60)

if __name__ == "__main__":
    run_auto_generate()