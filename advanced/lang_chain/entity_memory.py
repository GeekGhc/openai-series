from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

# EntityMemory帮助我们记住整个对话里面的“命名实体”（Entity），保留实际在对话中我们最关心的信息

llm = OpenAI(temperature=0)

# 命名实体识别
entityMemory = ConversationEntityMemory(llm=llm)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=entityMemory
)

answer = conversation.predict(
    input="我叫张老三，在你们这里下了一张订单，订单号是 2023ABCD，我的邮箱地址是 customer@abc.com，但是这个订单十几天了还没有收到货")
print(answer)

# 打印存储的信息
print(conversation.memory.entity_store.store)
