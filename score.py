# 评分脚本，用于计算生成的动作计数与ground truth的差异
# demo 版本，目前比较依赖规则提取
import json
import re
import numpy as np

def extract_count(response):
    # Replace all asterisks with spaces to avoid bold markdown format issues
    response = response.replace('*', ' ')
    
    # Replace numbers followed by punctuation with the number followed by a space
    response = re.sub(r'(\d)([.,!?])', r'\1 ', response)
    
    # Use regex to find the last standalone number
    match = re.findall(r'\s(\d+)\s', response.strip() + ' ')
    if match:
        return int(match[-1])
    
    return None

def process_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for ii, line in enumerate(infile):
            data = json.loads(line)
            ground_truth = data['ground_truth']
            generated_response = data['generated_response']
            generated_count = extract_count(generated_response)
            
            if generated_count is not None:
                outfile.write(f"Ground Truth: {ground_truth}, Generated Count: {generated_count}\n")
            else:
                print(ii+1)

def calculate_statistics(output_file, max_count=None):
    with open(output_file, 'r') as file:
        differences = []
        exact_matches = 0
        for line in file:
            parts = line.strip().split(', ')
            ground_truth_str = parts[0].split(': ')[1]
            generated_count = int(parts[1].split(': ')[1])
            
            ground_truth_values = list(map(int, ground_truth_str.split('-')))
            if max_count is not None and min(ground_truth_values) > max_count:
                continue
            min_diff = min(abs(generated_count - gt) for gt in ground_truth_values)
            differences.append(min_diff)
            
            if generated_count in ground_truth_values:
                exact_matches += 1
        
        max_diff = max(differences)
        min_diff = min(differences)
        mean_diff = np.mean(differences)
        
        print(f"Max Difference: {max_diff}")
        print(f"Min Difference: {min_diff}")
        print(f"Mean Difference: {mean_diff}")
        print(f"Exact Matches: {exact_matches}")


if __name__ == "__main__":
    input_file = '/home/cjr/WorkSpace/VLLM-action-count/downloads/all-videos-response.jsonl'
    output_file = 'output.txt'
    process_jsonl(input_file, output_file)
    calculate_statistics(output_file)
