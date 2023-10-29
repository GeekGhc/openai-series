import openai, os

# 基于OpenAI的whisper接口完成语音转文本功能

openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file = open("./data/podcast_clip.mp3", "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)

# 添加prompt辅助模型识别
transcript = openai.Audio.transcribe("whisper-1", audio_file,
                                     prompt="这是一段Onboard播客，里面会聊到ChatGPT以及PALM这个大语言模型。这个模型也叫做Pathways Language Model。")
print(transcript['text'])

# translate进行英文翻译
translated_prompt = """This is a podcast discussing ChatGPT and PaLM model. 
The full name of PaLM is Pathways Language Model."""
transcript = openai.Audio.translate("whisper-1", audio_file,
                                    prompt=translated_prompt)
print(transcript['text'])
