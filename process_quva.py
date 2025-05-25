import os
import shutil
import json
import numpy as np
from pathlib import Path

def create_directories():
    """创建必要的目录结构"""
    os.makedirs('downloads_quva/test/clips', exist_ok=True)

def get_new_filename(video_file, count):
    """生成新的文件名格式：0000_action_name_[count].mp4"""
    # 获取动作名称
    action_name = get_action_name(video_file)
    # 生成四位编号（从0000开始）
    index = int(video_file.split('_')[0])
    new_index = f"{index:04d}"
    # 组合新文件名
    new_filename = f"{new_index}_{action_name}_[{count}].mp4"
    return new_filename

def copy_videos():
    """复制视频文件到新目录并重命名"""
    source_dir = 'quvadataset/videos'
    target_dir = 'downloads_quva/test/clips'
    
    for video_file in os.listdir(source_dir):
        if video_file.endswith('.mp4'):
            source_path = os.path.join(source_dir, video_file)
            # 获取count
            count = get_count_from_npy(video_file)
            # 生成新文件名
            new_filename = get_new_filename(video_file, count)
            target_path = os.path.join(target_dir, new_filename)
            # 复制并重命名文件
            shutil.copy2(source_path, target_path)

def get_action_name(filename):
    """从文件名中提取动作名称"""
    # 去掉序号和.mp4后缀
    name = filename.split('_', 1)[1].replace('.mp4', '')
    # 将下划线替换为空格
    name = name.replace('_', ' ')
    return name

def get_count_from_npy(filename):
    """从对应的npy文件中读取count"""
    npy_file = filename.replace('.mp4', '.npy')
    npy_path = os.path.join('quvadataset/annotations', npy_file)
    
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        # 计算关键帧的数量
        return len(data)
    return 0

def generate_metadata():
    """生成metadata.json文件"""
    metadata = []
    clips_dir = 'downloads_quva/test/clips'
    
    for video_file in os.listdir(clips_dir):
        if video_file.endswith('.mp4'):
            # 从新文件名中提取信息
            parts = video_file.split('_', 1)
            action_name = parts[1].rsplit('_', 1)[0]  # 去掉最后的[count].mp4
            count = int(video_file.split('[')[1].split(']')[0])  # 提取count
            
            entry = {
                "name": action_name,
                "start_time": "00:00:00",
                "end_time": "00:00:00",
                "count": [count],
                "fuzzy_action": False,
                "complex_action": False,
                "original_filename": video_file
            }
            metadata.append(entry)
    
    # 将metadata写入JSON文件
    with open('downloads_quva/test/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

def main():
    print("开始处理QUVA数据集...")
    
    print("1. 创建目录结构...")
    create_directories()
    
    print("2. 复制并重命名视频文件...")
    copy_videos()
    
    print("3. 生成metadata.json...")
    generate_metadata()
    
    print("处理完成！")

if __name__ == "__main__":
    main() 