import os
import time
import google.generativeai as genai
import re
import json

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

def extract_movement_info(filename):
    # Extract movement name and side (if any)
    movement_match = re.search(r'_(.*?)(?:\((L|R)\))?_', filename)
    if movement_match:
        movement = movement_match.group(1)
        side = movement_match.group(2)
        if side == 'L':
            movement = f'left {movement}'
        elif side == 'R':
            movement = f'right {movement}'
    else:
        movement = None

    # Extract ground truth numbers
    ground_truth_match = re.search(r'\[(\d+(-\d+)*)\]', filename)
    if ground_truth_match:
        ground_truth = ground_truth_match.group(1)
    else:
        ground_truth = None

    return movement, ground_truth

def traverse_and_extract(directory):
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
    for root, dirs, files in os.walk(directory):
        if 'clips' in dirs:
            clips_path = os.path.join(root, 'clips')
            for file in os.listdir(clips_path):
                if file.endswith('.mp4'):
                    movement, ground_truth = extract_movement_info(file)
                    print(f'File: {file}')
                    print(f'Movement: {movement}')
                    print(f'Ground Truth: {ground_truth}')
                    # touch API operation
                    video_path = os.path.join(clips_path, file)
                    uploaded_video = upload_to_gemini(video_path, mime_type="video/mp4")
                    uploaded_video = wait_for_file_active(uploaded_video)
                    # 启动新聊天会话
                    chat_session = model.start_chat()
                    # 将文本提示和视频文件一起传入当前消息（注意：将文本与媒体文件组合成一个列表）
                    prompt_message = [
                        uploaded_video,
                        f"This is a video of a video blogger performing the {movement} movement. Please count how many {movement} movements the blogger has completed in total. Note that the blogger's movement frequency may not remain constant, so please count each movement carefully one by one. Analyze the video frame by frame and provide the reasoning behind your answer."    
                    ]

                    # 发送消息并获得回答
                    response = chat_session.send_message(prompt_message)
                    print("\nGenerated Response:\n", response.text)
                    # 这里要在原始视频clips文件夹下创建一个文件记录每个视频的movement和ground truth以及生成的response
                    response_data = {
                        "file": file,
                        "movement": movement,
                        "ground_truth": ground_truth,
                        "generated_response": response.text
                    }
                    with open(os.path.join(clips_path, 'response.jsonl'), 'a') as f:
                        f.write(json.dumps(response_data) + '\n')
                    # 要在根目录下创建一个文件记录所有视频的movement和ground truth以及生成的response
                    response_data = {
                        "file": os.path.join(clips_path, file),
                        "movement": movement,
                        "ground_truth": ground_truth,
                        "generated_response": response.text
                    }
                    with open(os.path.join(directory, 'all-videos-response.jsonl'), 'a') as f:
                        f.write(json.dumps(response_data) + '\n')

if __name__ == "__main__":
    downloads_directory = '/home/cjr/WorkSpace/VLLM-action-count/downloads'
    traverse_and_extract(downloads_directory)
