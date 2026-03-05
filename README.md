# ASK-RECIPE

## 项目简介
这是一个提供做菜食谱相关问答的项目，主要基于食谱项目[HowToCook](https://github.com/Anduin2017/HowToCook/tree/master)和RAG（检索增强生成）初学者项目[all-in-rag](https://github.com/datawhalechina/all-in-rag/)。

### 项目内容
- 准备数据：基于markdown文档进行结构分块，元数据增强，并构建父子文本块平衡检索准确率和完整性。 
- 构建索引：利用BEG模型对文档做embedding，结合FAISS向量数据库构建和储存索引。
- 优化检索：结合向量检索和BM25检索同时捕捉语义相似度和关键词匹配度，并使用RRF对检索结果重排。
- 集成生成：使用LLM进行意图识别完成查询路由，分发为列表、详细、一般模式，一般模式下进行查询重写优化。
- 评估系统：******

## 项目意义
本项目是为了记录自己对RAG的技术全栈学习，同时为所有学习RAG的人分享经验。

## 项目亮点
- 系统的搭建了一套RAG，结合了相关理论和实践学习。
- 掌握相关技术栈，如langchain库、BGE模型、FAISS向量数据库。
- 学习了Advanced RAG技术，如父子文本块，向量检索和BM25检索的混合检索，RRF重排，查询重写和路由。

## 未来计划
- 多模态检索。
- 利用知识图谱构建Graph RAG系统。
- 使用Agentic RAG和强化学习提升RAG系统性能。

## 快速开始
### 环境配置
1. 创建虚拟环境
使用conda创建环境
conda create -n ask-recipe python=3.12.7
conda activate ask-recipe

2. 安装核心依赖
pip install -r requirements.txt

3. 配置API key
在rag_modules文件夹中按照.env.example文件创建一个.env文件，配置其中包括MOONSHOT_API_KEY在内变量

4. python main.py 即可打开RAG系统进行交互式问答。

