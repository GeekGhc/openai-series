import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.utilities import PythonREPL
from langchain import LLMMathChain

openai.api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)
multiply_prompt = PromptTemplate(template="请计算一下{question}是多少?", input_variables=["question"])
math_chain = LLMChain(llm=llm, prompt=multiply_prompt, output_key="answer")
answer = math_chain.run({"question": "352乘以493"})
print("OpenAI API 说答案是:", answer)

python_answer = 352 * 493
print("Python 说答案是:", python_answer)

multiply_by_python_prompt = PromptTemplate(template="请写一段Python代码，计算{question}?", input_variables=["question"])
math_chain = LLMChain(llm=llm, prompt=multiply_by_python_prompt, output_key="answer")
answer_code = math_chain.run({"question": "352乘以493"})

# 追加python解释器，基于langchain的内置utilities包
# https://python.langchain.com/docs/modules/agents/tools.html
python_repl = PythonREPL()
result = python_repl.run(answer_code)
print(result)

# 效果和上面一样，封装了LLMMathChain
llm_math = LLMMathChain(llm=llm, verbose=True)
result = llm_math.run("请计算一下352乘以493是多少?")
print(result)
