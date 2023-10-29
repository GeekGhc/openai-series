import os, requests, json

API_TOKEN = os.environ.get("HUGGINGFACE_API_KEY")
headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

model = "google/flan-t5-xxl"
API_URL = f"https://api-inference.huggingface.co/models/{model}"


def query(payload, api_url=API_URL, headers=headers):
    data = json.dumps(payload)
    response = requests.request("POST", api_url, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


# question = "Please answer the following question. What is the capital of France?"
# data = query({"inputs": question})
# print(data)


# 文本embedding
# model = "hfl/chinese-pert-base"
# API_URL = f"https://api-inference.huggingface.co/models/{model}"
#
# question = "今天天气真不错！"
# data = query({"inputs": question, "wait_for_model": True}, api_url=API_URL)
#
# print(data)

# 测试自己部署的模型
API_URL = "https://xxxx.aws.endpoints.huggingface.cloud"

text = "My name is WangWei and I like to"
data = query({"inputs": text}, api_url=API_URL)

print(data)
