import os
import json
import hashlib
from yt_dlp import YoutubeDL
import re
import shutil

# 读取JSON数据
def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# 读取强制覆盖列表
def load_force_download_list(force_download_file="force_download.txt"):
    if not os.path.exists(force_download_file):
        return []
    with open(force_download_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 下载视频
def download_videos(json_data, download_dir="downloads"):
    # 创建下载目录
    os.makedirs(download_dir, exist_ok=True)
    
    # 读取强制覆盖列表
    force_download_list = load_force_download_list()
    
    # 记录新创建的下载目录
    new_download_dirs = []

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
        
        # 检查文件夹是否已存在，如果存在且不在强制覆盖列表中，则跳过
        if os.path.exists(video_dir) and v_id not in force_download_list:
            # print(f"目录 {video_dir} 已存在，跳过下载")
            continue
        
        # 创建新目录或重新下载
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            new_download_dirs.append(v_id)
        else:
            new_download_dirs.append(v_id)
            print(f"目录 {video_dir} 已存在，但在强制覆盖列表中，将删除该目录下所有内容并重新下载")
            shutil.rmtree(video_dir)
            os.makedirs(video_dir)
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
            original_url = url # 保留原始 URL 以便打印或记录

            # 使用正则表达式移除 &list=...&index=... 部分
            cleaned_url = re.sub(r"&list=[^&]+&index=\d+", "", url)

            # 检查 URL 是否被修改
            if cleaned_url != url:
                print(f"检测到播放列表信息，使用正则清理 URL。原始 URL: {original_url}")
                url = cleaned_url # 更新 url 为清理后的版本
                print(f"用于下载的 URL: {url}")

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
    
    # 返回新创建的下载目录列表
    return new_download_dirs

# 主函数
def main():
    # JSON文件路径
    json_file = "video_list.json"  # 替换为你的JSON文件路径

    # 加载JSON数据
    json_data = load_json(json_file)

    # 下载视频并获取新创建的目录
    new_dirs = download_videos(json_data)
    
    # 输出新创建的目录
    print("\n新创建的下载目录:")
    for dir_path in new_dirs:
        print(dir_path)
    print("新创建的文件夹名称如上，请手动保存至force_download.txt文件，便于后续强制覆写使用。")

if __name__ == "__main__":
    main()
