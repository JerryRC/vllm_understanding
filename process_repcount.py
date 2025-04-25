"""
处理RepCount数据集

1. 清理动作名称，主要将下划线替换为空格避免影响后续的正则识别。
2. 处理RepCount数据集的单个划分（train, test, valid）分别对应于之前downloads中的“一个视频”。
3. 处理后的数据集（train, test, valid）下各自有clips文件夹，因为原始文件已经是一个动作所以这里就是直接复制过来的。
   metadata.json文件基本按照原始格式记录了各“动作”，但是<开始时间>和<结束时间>都设置为00:00:00因为本质上并不存在大视频裁切成小视频。
4. 后续流程依然去原来的clip.py里执行，只是略过 clip 这一步即可。
"""

import os
import shutil
import pandas as pd
import json
import re

def clean_action_name(name):
    """清理动作名称，使其适合用于文件名。"""
    name = name.lower()
    name = re.sub(r'_', ' ', name)  # 将下划线替换为空格
    # name = re.sub(r'[^a-z0-9_]', '', name)  # 移除其他非法字符
    return name

def process_repcount_split(repcount_base_dir, downloads_base_dir, split):
    """处理 RepCount 数据集的单个划分（train, test, valid）。"""
    
    annotation_file = os.path.join(repcount_base_dir, 'annotation', f'{split}.csv')
    video_dir = os.path.join(repcount_base_dir, 'video', split)
    
    output_split_dir = os.path.join(downloads_base_dir, split)
    output_clips_dir = os.path.join(output_split_dir, 'clips')
    # output_frames_dir = os.path.join(output_split_dir, 'frames')
    # output_slow_dir = os.path.join(output_split_dir, 'slow')
    metadata_file = os.path.join(output_split_dir, 'metadata.json')

    # 创建输出目录
    os.makedirs(output_clips_dir, exist_ok=True)
    # os.makedirs(output_frames_dir, exist_ok=True)
    # os.makedirs(output_slow_dir, exist_ok=True)

    # metadata 现在是一个列表，而不是字典
    metadata_list = []

    # 检查标注文件是否存在
    if not os.path.exists(annotation_file):
        print(f"警告：找不到'{split}'的标注文件: {annotation_file}")
        return

    print(f"处理分割: {split}")
    try:
        # 读取标注文件
        df = pd.read_csv(annotation_file)
        if df.columns[0].startswith('Unnamed'):
            # 如果第一列是无名索引列，则将其作为索引读取
            df = pd.read_csv(annotation_file, index_col=0)
        
        # 确保使用正确的列名 - CSV中的列名是 type, name, count
        # type：动作类型/名称，name：原始文件名，count：计数
    except Exception as e:
        print(f"读取标注文件时出错 {annotation_file}: {e}")
        return

    # 处理每一个视频文件
    for index, row in df.iterrows():
        # 正确使用CSV中的列名
        original_filename = row['name']  # 原始文件名在'name'列
        action_label_raw = row['type']   # 动作标签在'type'列
        count = row['count']            # 计数仍在'count'列

        # 验证数据
        if pd.isna(original_filename) or pd.isna(action_label_raw) or pd.isna(count):
            print(f"由于数据缺失跳过行 {index}: {row.to_dict()}")
            continue
            
        try:
            count = int(count)
        except ValueError:
            print(f"由于计数无效跳过行 {index}: {count}")
            continue

        action_label_clean = clean_action_name(str(action_label_raw))
        
        source_video_path = os.path.join(video_dir, original_filename)
        
        # 检查源视频是否存在
        if not os.path.exists(source_video_path):
            print(f"警告：找不到源视频: {source_video_path}")
            continue

        # 使用当前 metadata 列表长度作为序列号，格式化为4位数字
        sequence_number = f"{len(metadata_list):04d}"
        
        # 新的文件名使用序列号、清理后的动作名称和计数
        new_filename = f"{sequence_number}_{action_label_clean}_[{count}].mp4"
        dest_video_path = os.path.join(output_clips_dir, new_filename)

        try:
            # 复制并重命名视频文件
            print(f"  复制 '{original_filename}' -> '{new_filename}'")
            shutil.copy2(source_video_path, dest_video_path)  # copy2 保留元数据

            # 添加条目到 metadata 列表
            metadata_entry = {
                "name": action_label_clean,  # 存储动作标签
                "start_time": "00:00:00",
                "end_time": "00:00:00",
                "count": [count],        # 将计数作为单元素数组存储
                "fuzzy_action": False,
                "complex_action": False,
                "original_filename": original_filename  # 保存原始文件名以供参考
            }
            metadata_list.append(metadata_entry)
        except Exception as e:
            print(f"处理文件时出错 {original_filename}: {e}")

    # 将 metadata 列表写入 JSON 文件
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
        print(f"元数据已写入 {metadata_file}")
    except Exception as e:
        print(f"写入元数据文件时出错 {metadata_file}: {e}")

# --- 主执行 ---
if __name__ == "__main__":
    repcount_base = 'RepCountDataset'  # 数据集原始位置
    downloads_base = 'downloads_repcount'  # 输出目录名
    splits_to_process = ['train', 'test', 'valid']

    if not os.path.isdir(repcount_base):
        print(f"错误：在'{repcount_base}'找不到RepCountDataset基本目录")
    else:
        os.makedirs(downloads_base, exist_ok=True)
        for split_name in splits_to_process:
            process_repcount_split(repcount_base, downloads_base, split_name)
        print("\n处理完成。") 