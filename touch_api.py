import os
import time
import google.generativeai as genai

# 配置 API 密钥
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
    """
    上传指定文件到 Gemini，并返回上传后的文件对象。
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_file_active(file):
    """
    等待上传的文件处理完成，直到状态变为 ACTIVE（失败则抛出异常）。
    """
    print("Waiting for file processing...", end="")
    while True:
        current_file = genai.get_file(file.name)
        if current_file.state.name == "ACTIVE":
            print(" done.")
            return current_file
        elif current_file.state.name == "FAILED":
            raise Exception(f"File {file.name} failed to process")
        else:
            print(".", end="", flush=True)
            time.sleep(10)

# 上传视频文件，确保使用正确的 MIME 类型（视频文件一般使用 "video/mp4"）
video_path = "/home/cjr/WorkSpace/VLLM-action-count/downloads/54cd3d4802f41a8f9195d165159f07cc/clips/0839_side stretch(R)_[7].mp4"
uploaded_video = upload_to_gemini(video_path, mime_type="video/mp4")
uploaded_video = wait_for_file_active(uploaded_video)

# 创建 Gemini 模型实例
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config={
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)

# 启动聊天会话（不需要预先将视频放入历史中）
chat_session = model.start_chat()

# 将文本提示和视频文件一起传入当前消息（注意：将文本与媒体文件组合成一个列表）
prompt_message = [
    uploaded_video,
    "This is a video of a video blogger performing the Wall Bridge movement. Please count how many Wall Bridge movements the blogger has completed in total. Note that the blogger's movement frequency may not remain constant, so please count each movement carefully one by one. Analyze the video frame by frame and provide the reasoning behind your answer."    
]

# 发送消息并获得回答
response = chat_session.send_message(prompt_message)
print("\nGenerated Response:\n", response.text)
