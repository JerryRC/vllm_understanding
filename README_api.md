# 视频帧分析API批量访问脚本

这个脚本用于自动遍历视频数据集，将每个动作的所有帧发送给多模态大语言模型(LLM) API进行分析，并保存返回结果。

## 主要功能

- **自动遍历**：自动扫描downloads目录下所有视频目录和frames文件夹
- **动作识别**：根据文件夹名称提取动作名称和标准次数（ground truth）
- **帧批量处理**：一次性发送每个动作的所有帧给API
- **支持多种模型**：
  - 通过OpenRouter访问:
    - OpenAI GPT-4o
    - Claude-3 Opus 
    - Qwen 2.5 VL 72B
  - 通过Google API访问:
    - Gemini Pro Vision

## 数据集结构

脚本期望的数据集结构如下：

```
downloads/
├── [视频哈希ID1]/
│   └── frames/
│       ├── [动作1]_frames/
│       │   ├── 000001.png
│       │   ├── 000002.png
│       │   └── ...
│       ├── [动作2]_frames/
│       └── ...
├── [视频哈希ID2]/
└── ...
```

## 安装依赖

```bash
pip install requests
```

## 设置API密钥

在使用脚本前，需要设置相应的API密钥环境变量：

```bash
# OpenRouter API密钥（用于访问OpenAI、Claude、Qwen等模型）
export OPENROUTER_API_KEY=your_openrouter_api_key_here

# Google API密钥（用于访问Gemini模型，目前未启用）
export GOOGLE_API_KEY=your_google_api_key_here
```

## 使用方法

运行脚本处理所有视频：

```bash
python frames_touch_api_batch.py
```

## 输出结果

脚本将结果保存在两个位置：

1. **每个frames目录下的frames-response.jsonl**：包含该视频中所有动作的分析结果
2. **主下载目录下的frames-all-videos-response.jsonl**：包含所有视频所有动作的分析结果汇总

输出的JSON格式如下：

```json
{
  "file": "0250_jack knife_[10]_frames",
  "movement": "jack knife",
  "ground_truth": "10",
  "generated_response": "模型的详细回复..."
}
```

## 工作流程

1. 遍历downloads目录下所有视频哈希目录
2. 找到每个视频中的frames目录
3. 遍历frames目录下的所有动作子目录
4. 提取动作名称和标准次数
5. 将所有帧编码为base64格式
6. 发送API请求（当前默认使用Qwen模型）
7. 保存分析结果到JSON文件

## 特别说明

1. 当前脚本默认使用Qwen模型，但代码中保留了调用其他模型的函数
2. 提示词设计为让模型识别和计数视频中的动作次数
3. 每个动作的所有帧将在一个API请求中发送
4. 请注意API的使用限制和费用，特别是当处理大量视频和帧时
