from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 通过 SummaryMemory 概括对话的历史并记下来

llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=OpenAI())

prompt_template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内

{history}
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=prompt_template
)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)
conversation_with_summary.predict(input="你好")
