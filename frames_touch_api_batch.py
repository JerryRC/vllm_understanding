import os
import base64
import requests
import json
import re

# API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# 支持的模型
MODELS = {
    "openai": "openai/gpt-4o-2024-11-20",
    "claude": "anthropic/claude-3-opus-20240229",
    "qwen": "qwen/qwen2.5-vl-72b-instruct:free"
}

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

def encode_image(image_path):
    """将图像编码为base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_openrouter_api(model, frames, movement):
    """调用OpenRouter API"""
    prompt = f"These are a series of frames of a video blogger performing the {movement} movement. Please count how many {movement} movements the blogger has completed in total. Note that the blogger's movement frequency may not remain constant, so please count each movement carefully one by one. Analyze the video frame by frame and provide the reasoning behind your answer."
    
    # 构造消息
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    # 添加所有帧
    for frame_path in frames:
        base64_image = encode_image(frame_path)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    # 构造请求
    payload = {
        "model": MODELS[model],
        "messages": messages
    }
    # 发送请求
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )
    
    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")

def call_gemini_api(frames, movement):
    """调用Google Gemini API"""
    prompt = f"These are a series of frames of a video blogger performing the {movement} movement. Please count how many {movement} movements the blogger has completed in total. Note that the blogger's movement frequency may not remain constant, so please count each movement carefully one by one. Analyze the video frame by frame and provide the reasoning behind your answer."
    
    # 构造消息
    contents = [{"text": prompt}]
    
    # 添加所有帧
    for frame_path in frames:
        base64_image = encode_image(frame_path)
        contents.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64_image
            }
        })
    
    # 发送请求
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={GOOGLE_API_KEY}",
        json={"contents": contents}
    )
    
    result = response.json()
    return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

def traverse_and_extract(directory):
    for root, dirs, files in os.walk(directory):
        if 'frames' in dirs:
            frames_path = os.path.join(root, 'frames')
            for item in os.listdir(frames_path):
                if os.path.isdir(os.path.join(frames_path, item)):
                    movement, ground_truth = extract_movement_info(item)
                    print(f'Directory: {item}')
                    print(f'Movement: {movement}')
                    print(f'Ground Truth: {ground_truth}')

                    # movement path 下面是所有的此动作的帧，需要遍历所有帧encode到base64
                    movement_frame_path = os.path.join(frames_path, item)
                    frames = sorted([
                        os.path.join(movement_frame_path, f) for f in os.listdir(movement_frame_path)
                        if f.endswith(('.png', '.jpg', '.jpeg'))
                    ])
                    
                    response = call_openrouter_api("qwen", frames, movement)
                    print("\nGenerated Response:\n", response)
                    # 这里要在原始视频frames文件夹下创建一个文件记录每个视频的movement和ground truth以及生成的response
                    response_data = {
                        "file": item,
                        "movement": movement,
                        "ground_truth": ground_truth,
                        "generated_response": response
                    }
                    with open(os.path.join(frames_path, 'frames-response.jsonl'), 'a') as f:
                        f.write(json.dumps(response_data) + '\n')
                    # 要在根目录下创建一个文件记录所有视频的movement和ground truth以及生成的response
                    response_data = {
                        "file": os.path.join(frames_path, item),
                        "movement": movement,
                        "ground_truth": ground_truth,
                        "generated_response": response
                    }
                    with open(os.path.join(directory, 'frames-all-videos-response.jsonl'), 'a') as f:
                        f.write(json.dumps(response_data) + '\n')

if __name__ == "__main__":
    downloads_directory = '/home/cjr/WorkSpace/VLLM-action-count/downloads'
    traverse_and_extract(downloads_directory)
