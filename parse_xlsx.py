import pandas as pd
import json
import hashlib

def parse_excel_to_json(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

    # 初始化结果列表
    result = []

    # 遍历每一列，每四列代表一个视频
    # for i in range(1, len(df.columns), 4):
    for i in range(1, len(df.columns), 2):
        # 提取视频URL
        url = df.iloc[1, i]
        if pd.isna(url):
            continue  # 如果URL为空，跳过这一列

        # 生成v_id（使用URL的哈希值）
        v_id = hashlib.md5(url.encode()).hexdigest()

        # 生成v_name
        if url.endswith('/'):
            v_name = url.split('/')[-2]
        else:
            v_name = url.split('/')[-1]  # 取URL的最后一部分作为名称

        # 初始化动作列表
        actions = []

        # 遍历动作信息
        for j in range(2, len(df), 4):
            # 提取动作信息
            name = df.iloc[j, i]
            start_time = df.iloc[j+1, i]
            end_time = df.iloc[j+2, i]
            count = df.iloc[j+3, i]
            

            # 如果动作信息不完整，跳过
            if pd.isna(start_time) or pd.isna(end_time) or pd.isna(count):
                continue
            
            # 处理时间格式：确保是字符串格式
            start_time = str(start_time)
            end_time = str(end_time)

            # 处理动作计数
            if isinstance(count, float) and not count.is_integer():
                count_list = [int(count), int(count) - 1, int(count) + 1]
            else:
                count_list = [int(count)]

            # 检查复杂动作标记
            complex_action = False
            # if isinstance(df.iloc[j, i+2], str) and any('\u4e00' <= char <= '\u9fff' for char in df.iloc[j, i+2]):
            #     complex_action = True

            # 添加到动作列表
            actions.append({
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
                "count": count_list,
                "fuzzy_action": isinstance(count, float) and not count.is_integer(),
                "complex_action": complex_action
            })

        # 添加到结果列表
        result.append({
            "v_id": v_id,
            "v_name": v_name,
            "url": url,
            "actions": actions
        })

    return json.dumps(result, indent=4)

# 使用示例
# file_path = 'Ground True.xlsx'
file_path = 'Ground True 1.xlsx'
json_output = parse_excel_to_json(file_path)
print(json_output)
with open('video_list.json', 'w', encoding='utf-8') as f:
    f.write(json_output)
