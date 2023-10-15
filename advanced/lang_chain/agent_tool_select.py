from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# 定义agent的几类Tools

llm = OpenAI(temperature=0)


def search_order(input: str) -> str:
    return "订单状态：已发货；发货日期：2023-01-01；预计送达时间：2023-01-10"


def recommend_product(input: str) -> str:
    return "红色连衣裙"


def faq(intput: str) -> str:
    return "7天无理由退货"


tools = [
    Tool(
        name="Search Order", func=search_order,
        description="useful for when you need to answer questions about customers orders"
    ),
    Tool(name="Recommend Product", func=recommend_product,
         description="useful for when you need to answer questions about product recommendations"
         ),
    Tool(name="FAQ", func=faq,
         description="useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."
         )
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
result = agent.run(question)
print(result)
question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
result = agent.run(question)
print(result)
question = "请问你们的货，能送到三亚吗？大概需要几天？"
result = agent.run(question)
print(result)

# 限制思考的次数(重试次数),让AI不用强行回答
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", max_iterations = 2, verbose=True)
question = "请问你们的货，能送到三亚吗？大概需要几天？"
result = agent.run(question)
print("===")
print(result)
print("===")