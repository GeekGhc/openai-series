import os, openai, backoff
import pandas as pd
import subprocess
import os
import openai

# 基于组合关系生成几段故事用于模型微调

openai.api_key = os.getenv("OPENAI_API_KEY")
dynasties = ['唐', '宋', '元', '明', '清', '汉', '魏', '晋', '南北朝']
super_powers = ['隐形', '飞行', '读心术', '瞬间移动', '不死之身', '喷火']
story_types = ['轻松', '努力', '艰难']


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gpt35(prompt, max_tokens=2048, temperature=0.5, top_p=1, frequency_penalty=0, presence_penalty=0):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)
    return response["choices"][0]["text"]


def prepare_stories(dynasties, super_powers, story_types, output_file="data/ultraman_stories.csv"):
    df = pd.DataFrame()
    repeat = 3
    for dynasty in dynasties:
        for super_power in super_powers:
            for story_type in story_types:
                for i in range(repeat):
                    prompt = f"""请你用中文写一段300字的故事，情节跌宕起伏，讲述一位{dynasty}朝时期的英雄人物，穿越到现代，拥有了{super_power}这样的超能力，通过{story_type}的战斗，帮助奥特曼一起打败了怪兽的故事。"""
                    story = gpt35(prompt)
                    row = {"dynasty": dynasty, "super_power": super_power, "story_type": story_type, "story": story}
                    row = pd.DataFrame([row])
                    df = pd.concat([df, row], axis=0, ignore_index=True)

    df.to_csv("data/ultraman_stories.csv")


prepare_stories(dynasties, super_powers, story_types)

df = pd.read_csv("data/ultraman_stories.csv")
df['sub_prompt'] = df['dynasty'] + "," + df['super_power'] + "," + df['story_type']
prepared_data = df.loc[:, ['sub_prompt', 'story']]
prepared_data.rename(columns={'sub_prompt': 'prompt', 'story': 'completion'}, inplace=True)
prepared_data.to_csv('data/prepared_data.csv', index=False)

# 把故事的csv文件转化成JSONL格式的文件
subprocess.run('openai tools fine_tunes.prepare_data --file data/prepared_data.csv --quiet'.split())
# 数据生成完成后，提交微调指令给模型
# 指定了三个参数，分别是用来训练的数据文件、一个基础模型，以及生成模型的后缀
subprocess.run(
    'openai api fine_tunes.create --training_file data/prepared_data_prepared.jsonl --model curie --suffix "ultraman"'.split())
# 找出微调后的模型
subprocess.run('openai api fine_tunes.list'.split())
# 提供微调任务id查看模型微调的效果
subprocess.run('openai api fine_tunes.results -i ft-3oxkr1zBVB4fJWogJDDjQbr0'.split())
# 在之前的微调模型上继续优化
# 模型换成之前的微调模型，learning_rate_multiplier根据样本数量在 0.05 到 0.2 不等。如果继续微调的样本数要比之前微调的数据量小很多，就可以调得大一点
subprocess.run(
    'openai api fine_tunes.create --training_file data/prepared_data_more_prepared.jsonl --model curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26 --suffix "ultraman" --learning_rate_multiplier 0.2'.split())


# 使用微调后的新模型进行回答
def write_a_story(prompt):
    response = openai.Completion.create(
        model="curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2000,
        stream=True,  # 拿到一个可以通过迭代器访问的一系列 events，每一个 event 都包含了一部分新生成的文,而不是直接返回全部结果文本
        top_p=1,
        stop=["."])
    return response["choices"][0]["text"]


story = write_a_story("宋,发射激光,艰难 ->\n")
print(story)
