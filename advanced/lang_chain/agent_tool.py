from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.agents import tool
from langchain.agents import initialize_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader
import json, re
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# 封装几种LLMChain作为Tool


# 商品问答LLMChain
llm = OpenAI(temperature=0)
loader = TextLoader('./data/ecommerce_faq.txt')
documents = loader.load()
text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

faq_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=docsearch, verbose=True)

# 商品推荐的LLMChain
product_loader = CSVLoader('./data/ecommerce_products.csv')
product_documents = product_loader.load()
product_text_splitter = CharacterTextSplitter(chunk_size=1024, separator="\n")
product_texts = product_text_splitter.split_documents(product_documents)
product_search = FAISS.from_documents(product_texts, OpenAIEmbeddings())
product_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=product_search, verbose=True)

# 商品查询LLMChain
ORDER_1 = "20230101ABC"
ORDER_2 = "20230101EFG"

ORDER_1_DETAIL = {
    "order_number": ORDER_1,
    "status": "已发货",
    "shipping_date": "2023-01-03",
    "estimated_delivered_date": "2023-01-05",
}

ORDER_2_DETAIL = {
    "order_number": ORDER_2,
    "status": "未发货",
    "shipping_date": None,
    "estimated_delivered_date": None,
}

# 商品查询的Tool
answer_order_info = PromptTemplate(
    template="请把下面的订单信息回复给用户： \n\n {order}?", input_variables=["order"]
)
answer_order_llm = LLMChain(llm=ChatOpenAI(temperature=0), prompt=answer_order_info)


# return_direct:不要再经过 Thought 那一步思考，直接把回答给到用户
@tool("Search Order", return_direct=True)
def search_order(input: str) -> str:
    """useful for when you need to answer questions about customers orders"""
    pattern = r"\d+[A-Z]+"
    match = re.search(pattern, input)

    order_number = input
    if match:
        order_number = match.group(0)
    else:
        return "请问您的订单号是多少？"
    if order_number == ORDER_1:
        return answer_order_llm.run(json.dumps(ORDER_1_DETAIL))
    elif order_number == ORDER_2:
        return answer_order_llm.run(json.dumps(ORDER_2_DETAIL))
    else:
        return f"对不起，根据{input}没有找到您的订单"


# 商品问题FAQ的Tool
@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)


# 商品推荐的Tool
@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.run(input)


tools = [
    search_order,
    recommend_product,
    faq
]

# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


# conversational-react-description:支持多轮对话
# ChatOpenAI的成本更低，需要指定memory
chatllm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_agent = initialize_agent(tools, chatllm,
                                      agent="conversational-react-description",
                                      memory=memory, verbose=True)
