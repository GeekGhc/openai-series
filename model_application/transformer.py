from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 默认采用distilbert-base-uncased-finetuned-sst-2-english 情感分析类模型
# classifier = pipeline(task="sentiment-analysis", device=-1)
# preds = classifier("I am really happy today!")

# 模型支持中文
# classifier = pipeline(model="uer/roberta-base-finetuned-jd-binary-chinese", task="sentiment-analysis", device=0)
# preds = classifier("这个餐馆太难吃了。")
# print(preds)

# 英译中
# translation = pipeline(task="translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh", device=-1)
# text = "I like to learn data science and AI."
# translated_text = translation(text)
# print(translated_text)

# 语音识别类的任务
# transcriber = pipeline(model="openai/whisper-medium", device=-1)
# result = transcriber("../speech_image/data/podcast_clip.mp3")
# 中文处理
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
transcriber = pipeline(model="openai/whisper-medium", device=0,
                       generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
result = transcriber("./data/podcast_clip.mp3")
print(result)
