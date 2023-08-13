# 通过moderate接口检测文本内容分类
# moderate的接口返回的是一个 JSON，里面包含是否应该对输入的内容进行标记的 flag 字段
# 也包括具体是什么类型的问题的 categories 字段
# 以及对应每个 categories 的分数的 category_scores 字段
import openai


def chatgpt(text):
    messages = []
    messages.append({"role": "system", "content": "You are a useful AI assistant"})
    messages.append({"role": "user", "content": text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
    )
    message = response["choices"][0]["message"]["content"]
    return message


# threaten = "你不听我的我就拿刀砍死你"
# print(chatgpt(threaten))


threaten = "你不听我的我就拿刀砍死你"


def moderation(text):
    response = openai.Moderation.create(
        input=text
    )
    output = response["results"][0]
    return output


print(moderation(threaten))
