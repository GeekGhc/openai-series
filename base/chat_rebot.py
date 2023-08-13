import openai
import os

# 获取open ai的秘钥
openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"

prompt = '请你用朋友的语气回复给到客户，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2021AEDG，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。'


def get_response(prompt, temperature=1.0):
    # api：https://platform.openai.com/docs/api-reference/completions/create
    completions = openai.Completion.create(engine=COMPLETION_MODEL,
                                           prompt=prompt,
                                           max_tokens=1024,  # 调用生成的内容允许的最大 token 数量,token：分词之后的一个字符序列里的一个单元
                                           n=1,
                                           stop=None,
                                           temperature=temperature,  # 控制温度，[0,2]值越高则输出越随机
                                           )
    message = completions.choices[0].text
    print(completions.choices)
    return message


# print(get_response(prompt))


# question = """
# Q : 鱼香肉丝怎么做？
# A :
# """
# print(get_response(question))


# question = """
# Q : 鱼香肉丝怎么做？
# A :
# 1.准备好食材：500克猪里脊肉，2个青椒，2个红椒，1个洋葱，2勺蒜蓉，3勺白糖，适量料酒，半勺盐，2勺生抽，2勺酱油，2勺醋，少许花椒粉，半勺老抽，适量水淀粉。
# 2.将猪里脊肉洗净，沥干水分，放入料酒、盐，抓捏抓匀，腌制20分钟。
# 3.将青红椒洗净，切成丝，洋葱洗净，切成葱花，蒜末拌入小苏打水中腌制。
# 4.将猪里脊肉切成丝，放入锅中，加入洋葱，炒制至断生，加入青红椒，炒匀，加入腌制好的蒜末，炒制至断生。
# 5.将白糖、生抽、酱油、醋、花椒粉、老抽、水淀粉倒入锅中，翻炒匀，用小火收汁，调味即可。
#
# Q : 那蚝油牛肉呢？
# A :
# """
# print(get_response(question))


prompts = """判断一下用户的评论情感上是正面的还是负面的
评论：买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质
情感：正面

评论：随意降价，不予价保，服务态度差
情感：负面
"""

# good_case = prompts + """
# 评论：外形外观：苹果审美一直很好，金色非常漂亮
# 拍照效果：14pro升级的4800万像素真的是没的说，太好了，
# 运行速度：苹果的反应速度好，用上三五年也不会卡顿的，之前的7P用到现在也不卡
# 其他特色：14pro的磨砂金真的太好看了，不太高调，也不至于没有特点，非常耐看，很好的
# 情感：
# """
# print(get_response(good_case))

bad_case = prompts + """
评论：信号不好电池也不耐电不推荐购买
情感
"""

print(get_response(bad_case))
