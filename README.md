# 视频处理与动作计数项目

## 项目概述

本项目用于从Excel表格中批量下载视频，然后根据指定的时间点对视频进行裁剪、慢放处理和帧提取，以便后续进行动作计数分析。

## 工作流程

1. **数据解析**：从Excel表格中提取视频URL和动作信息
2. **视频下载**：批量下载所有指定的视频
3. **视频裁剪**：根据动作时间信息裁剪视频片段
4. **视频慢放**：对裁剪后的视频进行慢放处理（3倍慢放）
5. **帧提取**：从原始视频中按照指定的 fps 提取帧，用于后续分析

## 环境要求

- Python 3.6+
- FFmpeg（用于视频处理）
- 依赖包（待补全）：
  - pandas
  - yt-dlp
  - json
  - subprocess

## 安装

```bash
# 克隆仓库
git clone [仓库URL]
cd [仓库目录]

# 安装依赖
pip install pandas yt-dlp

# 安装FFmpeg（Ubuntu/Debian）
sudo apt-get update
sudo apt-get install ffmpeg

# 安装FFmpeg（CentOS/RHEL）
sudo yum install ffmpeg
```

## 使用方法

### 1. 准备Excel文件

准备包含视频链接和动作信息的Excel文件（参考 `Ground True.xlsx`）。Excel文件格式应为：
- 每4列代表一个视频
- 第1行：表头（可为空）
- 第2行：视频URL
- 之后每4行：动作名称、开始时间、结束时间、计数

### 2. 解析Excel文件

```bash
python parse_xlsx.py
```

这将生成 `video_list.json` 文件，包含所有视频和动作信息。

### 3. 下载视频

```bash
python download.py
```

视频将下载到 `downloads/` 目录，每个视频有自己的子目录，以视频ID命名。

### 4. 处理视频

```bash
python clip.py
```

这个脚本会执行三个主要功能（可以在 `main()` 函数中启用或禁用）：

- `first_cut`: 裁剪视频，保存到 `downloads/{video_id}/clips/` 
- `second_slow`: 慢放处理，保存到 `downloads/{video_id}/slow/`
- `third_extract`: 提取帧，保存到 `downloads/{video_id}/frames/`

### 5. 视频处理效果检查

```bash
python check_videos.py
```

使用 `check_videos.py` 将会按照期望，检查所有慢放以及抽帧的文件夹内容是否满足，提供检测结果以及修复错误的脚本。后续只需要通过bash运行 `repair_videos.sh` 脚本即可重新进行视频处理（可能发生的错误通常是由于ffmpeg版本造成，需要进一步分析）。

### 6. 动作分析

根据裁切出的视频片段，可以使用第三方API或模型进行动作计数分析。项目包含了使用Gemini API进行分析的参考代码：
- `video_touch_api_single.py`：调用单视频分析
- `video_touch_api_batch.py`：批量分析

对于手动抽帧的api访问方案，提供了demo脚本 `gen_payload_and_curl.sh`，可生成对应的json载荷并通过curl进行一次api调用。该demo局限性在于：使用的是帧文件的本地路径而非base64编码的数据格式，可能会影响api访问的结果。

## 文件夹结构

```
downloads/
├── {视频ID1}/
│   ├── {视频名称}.mp4      # 原始视频
│   ├── metadata.json       # 动作信息元数据
│   ├── clips/              # 裁剪后的视频片段
│   ├── slow/               # 慢放处理后的视频
│   └── frames/             # 提取的视频帧
│       └── {视频片段名称}/   # 每个片段的帧文件夹
│           ├── 000001.png
│           ├── 000002.png
│           └── ...
└── {视频ID2}/
    └── ...
```

## 注意事项

1. 视频下载使用yt-dlp，支持多种视频平台
2. 视频裁剪时会自动前后各延长1秒，以确保不丢失动作
3. 帧提取采用4fps速率，可在 `clip.py` 中修改
4. 每个视频的元数据保存为 `metadata.json`，包含所有动作信息

## 故障排除

1. 如果下载失败，检查URL是否有效，或者尝试添加Cookie文件
2. 如果裁剪失败，检查FFmpeg是否正确安装
3. 慢放处理可能会占用较多CPU资源，请确保系统有足够资源
