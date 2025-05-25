import os
import json
import re
import subprocess

# 清理文件名中的非法字符
def clean_filename(name):
    # 替换非法字符为下划线
    name = str(name)
    return re.sub(r'[\\/:*?"<>|]', '_', name)

# 将时间字符串（HH:MM:SS 或 MM:SS）转换为秒数
def time_to_seconds(time_str):
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 3:  # MM:SS:DROP
        return parts[0] * 60 + parts[1]
    elif len(parts) == 2:  # MM:SS
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:  # SS
        return parts[0]
    else:
        raise ValueError(f"Invalid time format: {time_str}")

# 将秒数转换为MMSS格式
def seconds_to_mmss(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}{seconds:02d}"

# 裁剪视频
def cut_video(input_path, output_path, start_time, end_time):
    duration = end_time - start_time
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-ss', str(start_time),  # 起始时间（秒）
        '-i', input_path,  # 输入视频文件
        '-t', str(duration),  # 持续时间（秒）
        '-c:v', 'copy',  # 不重新编码视频流
        '-an',  # 禁用音频流
        '-map', '0:v',  # 仅处理视频流
        output_path  # 输出视频文件
    ]
    subprocess.run(command, check=True)

def slow_down_video(input_path, output_path, slow_factor):
    """
    通过简单拉长时间的方式实现慢放
    :param slow_factor: 2 倍慢放, 3 倍慢放, ...
    """
    assert slow_factor > 1
    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-vf', f"setpts={slow_factor}*PTS",
        '-an',
        '-c:v', 'libx264',
        '-preset', 'fast',  # 选择中等质量
        '-crf', '23',       # 选择中等质量
        output_path
    ]
    subprocess.run(command, check=True)

def extract_frames(video_path, output_dir, fps):
    """
    使用 ffmpeg 从视频中均匀抽帧，并保存到 `frames/{视频名称}/` 目录。
    
    :param video_path: 输入视频路径
    :param output_dir: 存放帧的文件夹路径
    :param fps: 每秒抽取的帧数
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # 获取视频名（无扩展名）
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    # FFmpeg 命令
    command = [
        'ffmpeg', '-y', '-i', video_path,  # 输入文件
        '-vf', f"fps={fps}",  # 每秒 `fps` 帧
        '-vsync', 'vfr',  # 确保不会跳帧
        '-frame_pts', '1',  # 保留帧时间信息
        os.path.join(frame_dir, "%06d.png")  # 输出格式
    ]

    subprocess.run(command, check=True)
    print(f"已完成 {video_name} 的抽帧，保存至 {frame_dir}")

# 处理单个视频
def process_video(video_dir, metadata_path):
    # 加载metadata.json
    with open(metadata_path, 'r') as f:
        actions = json.load(f)

    # 视频文件路径
    video_path = None
    for file in os.listdir(video_dir):
        if file.endswith('.mp4'):
            video_path = os.path.join(video_dir, file)
            break

    if not video_path:
        print(f"No video file found in {video_dir}")
        return

    # 创建clips文件夹
    clips_dir = os.path.join(video_dir, "clips")
    #delete the clips_dir no empty
    if os.path.exists(clips_dir):
        for file in os.listdir(clips_dir):
            os.remove(os.path.join(clips_dir, file))
        
    os.makedirs(clips_dir, exist_ok=True)

    # 遍历每个动作
    for action in actions:
        name = clean_filename(action["name"])  # 清理动作名称
        start_time = time_to_seconds(action["start_time"]) - 1  # 开始时间提前1秒
        end_time = time_to_seconds(action["end_time"]) + 1  # 结束时间延后1秒
        label = "[" + "-".join(map(str, action["count"])) + "]"  # 动作次数

        # 将开始时间格式化为MMSS
        start_time_mmss = seconds_to_mmss(start_time)

        # 输出文件路径
        output_path = os.path.join(clips_dir, f"{start_time_mmss}_{name}_{label}.mp4")

        # 裁剪视频
        try:
            cut_video(video_path, output_path, start_time, end_time)
            print(f"Cut {name} from {start_time}s to {end_time}s: {output_path}")
        except Exception as e:
            print(f"Failed to cut {name}: {e}")

# 读取要处理的文件夹列表
def read_folder_list(list_file):
    folder_list = []
    if os.path.exists(list_file):
        with open(list_file, 'r') as f:
            folder_list = [line.strip() for line in f if line.strip()]
    return folder_list

def first_cut(download_dir, folder_list=None):
    # 遍历每个视频文件夹, 根据metadata.json进行裁剪
    for v_id in os.listdir(download_dir):
        # 如果指定了文件夹列表，则只处理列表中的文件夹
        if folder_list and v_id not in folder_list:
            continue
            
        video_dir = os.path.join(download_dir, v_id)
        metadata_path = os.path.join(video_dir, "metadata.json")

        if os.path.exists(metadata_path):
            print(f"Processing video: {v_id}")
            process_video(video_dir, metadata_path)
        else:
            print(f"No metadata found for video: {v_id}")

def second_slow(download_dir, folder_list=None):
    # 遍历所有文件夹的clips文件夹，将其下文件逐个慢放，保存到与clip同级的slow文件夹
    for v_id in os.listdir(download_dir):
        # 如果指定了文件夹列表，则只处理列表中的文件夹
        if folder_list and v_id not in folder_list:
            continue
            
        video_dir = os.path.join(download_dir, v_id)
        clips_dir = os.path.join(video_dir, "clips")
        slow_dir = os.path.join(video_dir, "slow")
        os.makedirs(slow_dir, exist_ok=True)
        for file in os.listdir(clips_dir):
            input_path = os.path.join(clips_dir, file)
            output_path = os.path.join(slow_dir, file)
            slow_down_video(input_path, output_path, 3)
            print(f"Slow down {file} to {output_path}")

def third_extract(download_dir, folder_list=None):
    """
    遍历 `clips/` 目录中的视频，按每秒 `n` 帧均匀抽取，并保存到 `frames/` 目录。
    """
    for v_id in os.listdir(download_dir):
        # 如果指定了文件夹列表，则只处理列表中的文件夹
        if folder_list and v_id not in folder_list:
            continue
            
        video_dir = os.path.join(download_dir, v_id)
        clips_dir = os.path.join(video_dir, "clips")
        frames_dir = os.path.join(video_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for file in os.listdir(clips_dir):
            if file.endswith(".mp4"):
                input_path = os.path.join(clips_dir, file)
                extract_frames(input_path, frames_dir, fps=4)

    print("所有视频的帧提取完成！")

# 主函数
def main():
    # 下载目录
    # download_dir = "downloads"
    download_dir = "downloads_repcount"
    
    # 读取要处理的文件夹列表
    force_download_file = "force_download.txt"
    folder_list = read_folder_list(force_download_file)
    folder_list = []
    
    if folder_list:
        print(f"将只处理以下{len(folder_list)}个文件夹: {', '.join(folder_list)}")
    else:
        print("未指定处理文件列表，将处理所有文件夹")
    
    # first_cut(download_dir, folder_list)
    # second_slow(download_dir, folder_list)
    third_extract(download_dir, folder_list)


if __name__ == "__main__":
    main()