import os
import shutil
import re
import csv
from statistics import median

# 1. 复制所有clips下的视频到LLSP_form/video/test/
downloads_dir = 'downloads'
target_dir = 'LLSP_form/video/test'
os.makedirs(target_dir, exist_ok=True)

video_files = []
original_to_new = {}
for root, dirs, files in os.walk(downloads_dir):
    # if os.path.basename(root) == 'clips':
    if os.path.basename(root) == 'slow':
        for file in files:
            if file.lower().endswith('.mp4'):
                # 去掉逗号
                new_file = file.replace(',', '')
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, new_file)
                shutil.copy2(src_path, dst_path)
                video_files.append(new_file)
                original_to_new[file] = new_file

# 2. 生成csv
csv_path = 'LLSP_form/annotation/test.csv'
os.makedirs('LLSP_form/annotation', exist_ok=True)

def extract_label_from_filename(filename):
    # 匹配最后的中括号内容
    match = re.search(r'\[([0-9\-]+)\]', filename)
    if match:
        nums = match.group(1).split('-')
        nums = [int(n) for n in nums if n.isdigit()]
        if nums:
            # 取中位数
            return int(median(nums))
    return None

with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['', 'type', 'name', 'count', 'L1', 'L2'])
    idx = 0
    for file in video_files:
        label = extract_label_from_filename(file)
        if label is not None:
            writer.writerow([idx, 'skip', file, label, 0, 1])
            idx += 1
        else:
            print(f"警告：未能从文件名提取label: {file}")

print("全部完成！")