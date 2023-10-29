import os
import azure.cognitiveservices.speech as speechsdk

# 基于Azure进行语音合成

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_SPEECH_KEY'),
                                       region=os.environ.get('AZURE_SPEECH_REGION'))
# 指定播放的语言风格
# speech_config.speech_synthesis_voice_name = 'zh-CN-XiaohanNeural'
speech_config.speech_synthesis_voice_name = 'zh-CN-YunfengNeural'
# 文本
text = "今天天气真不错，ChatGPT真好用。"

# 默认异步播放
# audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
# 存储对应的语音
# audio_config = speechsdk.audio.AudioOutputConfig(filename="./data/tts.wav")
# speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
# speech_synthesizer.speak_text_async(text)

# 存储mp3
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
result = speech_synthesizer.speak_text_async(text).get()
stream =speechsdk.AudioDataStream(result)
stream.save_to_wav_file("./data/tts.mp3")

# 指定风格与角色
ssml = """<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
    <voice name="zh-CN-YunyeNeural">
        儿子看见母亲走了过来，说到：
        <mstts:express-as role="Boy" style="cheerful">
            “妈妈，我想要买个新玩具”
        </mstts:express-as>
    </voice>
    <voice name="zh-CN-XiaomoNeural">
        母亲放下包，说：
        <mstts:express-as role="SeniorFemale" style="angry">
            “我看你长得像个玩具。”
        </mstts:express-as>
    </voice>
</speak>"""
# speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()

# 一个 voice_name 在不同的场景片段下，用不同的语气和角色来说话
ssml = """<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="en-US-JennyNeural">
        <mstts:express-as style="excited">
            That'd be just amazing!
        </mstts:express-as>
        <mstts:express-as style="friendly">
            What's next?
        </mstts:express-as>
    </voice>
</speak>"""
# speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()
