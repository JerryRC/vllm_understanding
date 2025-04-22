import os
import json
import hashlib
from yt_dlp import YoutubeDL

# 读取JSON数据
def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# 下载视频
def download_videos(json_data, download_dir="downloads"):
    # 创建下载目录
    os.makedirs(download_dir, exist_ok=True)

    # 遍历每个视频
    for video in json_data:
        v_id = video["v_id"]
        v_name = video["v_name"]
        url = video["url"]
        # if "youtube" in url:
        #     continue
        actions = video["actions"]

        # 创建视频文件夹
        video_dir = os.path.join(download_dir, v_id)
        os.makedirs(video_dir, exist_ok=True)

        # 下载视频
        video_path = os.path.join(video_dir, f"{v_name}.mp4")
        ydl_opts = {
            'format': 'bestvideo[height<=720]',  # 下载720p视频
            'outtmpl': video_path,  # 视频保存路径
            'cookiefile': "~/tmp-cookies",  # 使用cookies文件
        }
        if os.path.exists(video_path):
            print(f"Skipping {url} because it already exists")
            continue
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Downloaded {url} to {video_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

        # 保存动作信息到metadata.json
        metadata_path = os.path.join(video_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(actions, f, indent=4)
        print(f"Saved metadata for {v_id} to {metadata_path}")

# 主函数
def main():
    # JSON文件路径
    json_file = "video_list.json"  # 替换为你的JSON文件路径

    # 加载JSON数据
    json_data = load_json(json_file)

    # 下载视频
    download_videos(json_data)

if __name__ == "__main__":
    main()
