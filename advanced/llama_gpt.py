import openai
import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, StorageContext, load_index_from_storage

openai.api_key = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

index.set_index_id("vector_index")
index.storage_context.persist('./data/index_mr_fujino')

# 基于索引回答问题
storage_context = StorageContext.from_defaults(persist_dir='./data/index_mr_fujino')
# load index
index = load_index_from_storage(storage_context, index_id="vector_index")

# build query index
query_engine = index.as_query_engine()
response = query_engine.query("鲁迅先生在日本学习医学的老师是谁？")
print(response)
