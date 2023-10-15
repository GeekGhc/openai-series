import openai, os
import faiss
from llama_index import SimpleDirectoryReader, LangchainEmbedding, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from transformers import AutoTokenizer, AutoModel

# 基于langchain实现llama index外链知识库的能力

openai.api_key = ""

text_splitter = SentenceSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)

parser = SimpleNodeParser(text_splitter=text_splitter, chunk_size=1024, chunk_overlap=20)
documents = SimpleDirectoryReader('../data/faq/').load_data()
nodes = parser.get_nodes_from_documents(documents)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
))
service_context = ServiceContext.from_defaults(embed_model=embed_model)

dimension = 768
faiss_index = faiss.IndexFlatIP(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context,
                         service_context=service_context)
query_engine = index.as_query_engine()

openai.api_key = os.environ.get("OPENAI_API_KEY")

# 问题1
# response = query_engine.query(
#     "请问你们海南能发货吗？",
#     # mode=QueryMode.EMBEDDING,
#     # verbose=True,
# )
# print(response)
#
# # 问题2
# response = query_engine.query(
#     "你们用哪些快递公司送货？",
# )
# print(response)
#
# # 问题3
# response = query_engine.query(
#     "你们的退货政策是怎么样的？",
# )
# print(response)


# 适用于运行在Colab
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()

question = """
自收到商品之日起7天内，如产品未使用、包装完好，您可以申请退货。某些特殊商品可能不支持退货，请在购买前查看商品详情页面的退货政策。

根据以上信息，请回答下面的问题：

Q: 你们的退货政策是怎么样的？
"""
response, history = model.chat(tokenizer, question, history=[])
print(response)
