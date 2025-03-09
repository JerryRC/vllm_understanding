import os
import json
import subprocess
from datetime import datetime
import sys

def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'json', 
            video_path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(output)
        return float(data['format']['duration'])
    except Exception as e:
        print(f"无法获取视频时长: {e}")
        return 0

def count_frames(frames_dir):
    """计算指定目录下的帧数量"""
    if not os.path.exists(frames_dir):
        return 0
    return len([f for f in os.listdir(frames_dir) if f.endswith('.png')])

def check_videos(download_dir):
    """检查视频处理状态"""
    results = {
        "total_videos": 0,
        "missing_slow": [],
        "short_slow": [],  # 慢放视频异常短
        "missing_frames": [],
        "few_frames": []   # 帧数异常少
    }
    
    # 遍历所有视频目录
    for v_id in os.listdir(download_dir):
        video_dir = os.path.join(download_dir, v_id)
        if not os.path.isdir(video_dir):
            continue
            
        clips_dir = os.path.join(video_dir, "clips")
        slow_dir = os.path.join(video_dir, "slow")
        frames_dir = os.path.join(video_dir, "frames")
        
        if not os.path.exists(clips_dir):
            continue
            
        # 遍历所有裁剪片段
        for clip_file in os.listdir(clips_dir):
            if not clip_file.endswith('.mp4'):
                continue
                
            results["total_videos"] += 1
            clip_path = os.path.join(clips_dir, clip_file)
            slow_path = os.path.join(slow_dir, clip_file)
            frame_dir = os.path.join(frames_dir, os.path.splitext(clip_file)[0])
            
            # 检查原始时长
            clip_duration = get_video_duration(clip_path)
            
            # 检查慢放视频
            if not os.path.exists(slow_path):
                results["missing_slow"].append((v_id, clip_file))
            else:
                slow_duration = get_video_duration(slow_path)
                # 检查慢放是否正常（应该是原时长的3倍左右）
                if slow_duration < clip_duration * 2:  # 如果慢放视频时长小于原视频的2倍
                    results["short_slow"].append((v_id, clip_file, clip_duration, slow_duration))
            
            # 检查帧提取
            if not os.path.exists(frame_dir):
                results["missing_frames"].append((v_id, clip_file))
            else:
                frame_count = count_frames(frame_dir)
                # 检查帧数量（应该是视频时长 * fps）
                expected_frames = int(clip_duration * 4)  # 4fps
                if frame_count < 10:  # 如果帧数少于10
                    results["few_frames"].append((v_id, clip_file, frame_count, expected_frames))
    
    return results

def print_results(results):
    """格式化打印结果"""
    print(f"总视频片段数: {results['total_videos']}")
    
    print("\n缺少慢放视频:")
    for v_id, clip in results["missing_slow"]:
        print(f"  - {v_id}/{clip}")
    
    print("\n慢放视频异常短:")
    for v_id, clip, orig_dur, slow_dur in results["short_slow"]:
        print(f"  - {v_id}/{clip} (原始: {orig_dur:.1f}s, 慢放: {slow_dur:.1f}s, 比例: {slow_dur/orig_dur:.1f}x)")
    
    print("\n缺少帧提取:")
    for v_id, clip in results["missing_frames"]:
        print(f"  - {v_id}/{clip}")
    
    print("\n帧数异常少:")
    for v_id, clip, frames, expected in results["few_frames"]:
        print(f"  - {v_id}/{clip} (实际: {frames}, 预期: {expected})")

def generate_repair_script(results):
    """生成修复脚本"""
    script_path = "repair_videos.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# 自动生成的修复脚本 - {}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        # 修复慢放视频
        if results["missing_slow"] or results["short_slow"]:
            f.write("# 修复慢放视频\n")
            for v_id, clip in results["missing_slow"]:
                f.write(f"python3 -c \"import clip; clip.slow_down_video('downloads/{v_id}/clips/{clip}', 'downloads/{v_id}/slow/{clip}', 3)\"\n")
            
            for v_id, clip, _, _ in results["short_slow"]:
                f.write(f"python3 -c \"import clip; clip.slow_down_video('downloads/{v_id}/clips/{clip}', 'downloads/{v_id}/slow/{clip}', 3)\"\n")
        
        # 修复帧提取
        if results["missing_frames"] or results["few_frames"]:
            f.write("\n# 修复帧提取\n")
            for v_id, clip in results["missing_frames"]:
                f.write(f"python3 -c \"import clip; clip.extract_frames('downloads/{v_id}/clips/{clip}', 'downloads/{v_id}/frames', 4)\"\n")
            
            for v_id, clip, _, _ in results["few_frames"]:
                vid_name = os.path.splitext(clip)[0]
                f.write(f"rm -rf 'downloads/{v_id}/frames/{vid_name}'\n")
                f.write(f"python3 -c \"import clip; clip.extract_frames('downloads/{v_id}/clips/{clip}', 'downloads/{v_id}/frames', 4)\"\n")
    
    print(f"\n修复脚本已生成: {script_path}")
    print(f"使用 'bash {script_path}' 执行修复")
    os.chmod(script_path, 0o755)  # 添加执行权限

def main():
    download_dir = "downloads"
    if not os.path.exists(download_dir):
        print(f"错误: 目录 '{download_dir}' 不存在!")
        return
    
    print("正在检查视频处理状态...\n")
    results = check_videos(download_dir)
    print_results(results)
    
    # 如果有问题，生成修复脚本
    if any(len(results[key]) > 0 for key in ["missing_slow", "short_slow", "missing_frames", "few_frames"]):
        generate_repair_script(results)

if __name__ == "__main__":
    main() 