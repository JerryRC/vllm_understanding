#!/bin/bash
# 请确保 OPENROUTER_API_KEY 环境变量已设置

# 定义存放帧图像的目录（注意目录中包含空格，所以用引号）
target_dir="/home/cjr/WorkSpace/VLLM-action-count/downloads/54cd3d4802f41a8f9195d165159f07cc/frames/0839_side stretch(R)_[7]"

# 获取目录下的所有文件，按自然数字顺序排序
files=($(ls -v "$target_dir"))

# 构造 JSON 中 messages.content 部分
# 第一个元素为文本消息
content_items="{
          \"type\": \"text\",
          \"text\": \"These are frames of a video about right side stretch. Please count the total number of movements performed by the creator in the video.\"
        }"

# 遍历所有文件，依次添加 image_url 消息项
for file in "${files[@]}"; do
    full_path="$target_dir/$file"
    content_items+=",{
          \"type\": \"image_url\",
          \"image_url\": {
            \"url\": \"$full_path\"
          }
        }"
done

# 组合成完整的 JSON payload
json_payload=$(cat <<EOF
{
  "model": "google/gemini-2.0-flash-exp:free",
  "messages": [
    {
      "role": "user",
      "content": [
        $content_items
      ]
    }
  ],
  "top_p": 1,
  "temperature": 0,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "repetition_penalty": 1,
  "top_k": 0
}
EOF
)

# 可选：将生成的 JSON 输出到文件以供调试
echo "$json_payload" > payload.json

# 使用 curl 发出请求
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -d "$json_payload" > response.json

echo ""
jq -r '.choices[0].message.content' /home/cjr/WorkSpace/VLLM-action-count/response.json
echo ""
