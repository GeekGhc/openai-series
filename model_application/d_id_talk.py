import requests
import os
from IPython.display import display, HTML


# 创建一段小视频
def generate_talk(input, avatar_url,
                  voice_type="microsoft",
                  voice_id="zh-CN-XiaomoNeural",
                  api_key=os.environ.get('DID_API_KEY')):
    url = "https://api.d-id.com/talks"
    payload = {
        "script": {
            "type": "text",
            "provider": {
                "type": voice_type,
                "voice_id": voice_id
            },
            "ssml": "false",
            "input": input
        },
        "config": {
            "fluent": "false",
            "pad_audio": "0.0"
        },
        "source_url": avatar_url
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Basic " + api_key
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()


avatar_url = "https://cdn.discordapp.com/attachments/1065596492796153856/1095617463112187984/John_Carmack_Potrait_668a7a8d-1bb0-427d-8655-d32517f6583d.png"
text = "董玲，爱你哦，祝你每天开开心心"

response = generate_talk(input=text, avatar_url=avatar_url)
print(response)


# 获取生成的视频talk
def get_a_talk(id, api_key=os.environ.get('DID_API_KEY')):
    url = "https://api.d-id.com/talks/" + id
    headers = {
        "accept": "application/json",
        "authorization": "Basic " + api_key
    }
    response = requests.get(url, headers=headers)
    return response.json()


# 播放视频
def play_mp4_video(url):
    video_tag = f"""
    <video width="640" height="480" controls>
        <source src="{url}" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    """
    return HTML(video_tag)


talk = get_a_talk(response['id'])
print(talk)

result_url = talk['audio_url']
play_mp4_video(result_url)
