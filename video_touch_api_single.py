import os
import time # Re-import time for sleeping
# import google.generativeai as genai # Using client directly
from google import genai
from google.genai import types # Import types for config

# 配置 API 密钥
# genai.configure(api_key=os.environ["GEMINI_API_KEY"]) # Deprecated
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Remove old upload_to_gemini function
# def upload_to_gemini(path, mime_type=None):
#     """
#     上传指定文件到 Gemini，并返回上传后的文件对象。
#     """
#     file = genai.upload_file(path, mime_type=mime_type)
#     print(f"Uploaded file '{file.display_name}' as: {file.uri}")
#     return file

# Remove old wait_for_file_active function
# def wait_for_file_active(file):
#     """
#     等待上传的文件处理完成，直到状态变为 ACTIVE（失败则抛出异常）。
#     """
#     print("Waiting for file processing...", end="")
#     while True:
#         current_file = genai.get_file(file.name)
#         if current_file.state.name == "ACTIVE":
#             print(" done.")
#             return current_file
#         elif current_file.state.name == "FAILED":
#             raise Exception(f"File {file.name} failed to process")
#         else:
#             print(".", end="", flush=True)
#             time.sleep(10)

# 上传视频文件 - New method
video_path = "downloads/54cd3d4802f41a8f9195d165159f07cc/clips/0839_side stretch(R)_[7].mp4"
print(f"Uploading file: {video_path}")
# Use client.files.upload
uploaded_file = client.files.upload(file=video_path)
print(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")

# No explicit wait needed <- Incorrect, re-adding wait logic
# uploaded_video = wait_for_file_active(uploaded_video)

# Wait for the file to be processed.
print("Waiting for file processing...")
while uploaded_file.state.name == "PROCESSING":
    print(".", end="", flush=True)
    time.sleep(10)
    # Call files.get to check the processing state.
    uploaded_file = client.files.get(name=uploaded_file.name)

if uploaded_file.state.name == "FAILED":
  raise ValueError(f"File processing failed: {uploaded_file.state.name} - {uploaded_file.uri}")

if uploaded_file.state.name != "ACTIVE":
    raise ValueError(f"File processing did not complete successfully: {uploaded_file.state.name} - {uploaded_file.uri}")

print(" File processing complete.")

# # 创建 Gemini 模型实例 - Not needed separately
# model = genai.GenerativeModel(
#     model_name="gemini-2.0-flash-exp",
#     generation_config={
#         "temperature": 0,
#         "top_p": 0.95,
#         "top_k": 40,
#         "max_output_tokens": 8192,
#         "response_mime_type": "text/plain",
#     },
# )

# # 启动聊天会话 - Not needed
# chat_session = model.start_chat()

# Prepare prompt text
movement = "side stretch"
prompt_text = f"This is a video of a video blogger performing the {movement} movement. Please count how many {movement} movements the blogger has completed in total. Note that the blogger's movement frequency may not remain constant, so please count each movement carefully one by one. Analyze the video frame by frame and provide the reasoning behind your answer. Please reason step by step, and put your final answer within \\boxed{{}}"

# # 将文本提示和视频文件一起传入当前消息 - Replaced by generate_content
# prompt_message = [
#     uploaded_video,
#     prompt_text
# ]

# 发送消息并获得回答 - New method
# response = chat_session.send_message(prompt_message)
print("Generating content...")
model_name = "gemini-2.0-flash-exp" # Or "gemini-2.0-flash"
# Store config values in a dict first
generation_config_dict = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
response = client.models.generate_content(
    model=model_name,
    contents=[uploaded_file, prompt_text], # Pass file object and text directly
    # generation_config=generation_config # Incorrect
    # Use config=types.GenerateContentConfig(...) instead
    config=types.GenerateContentConfig(**generation_config_dict)
)
print("\nGenerated Response:\n", response.text)
