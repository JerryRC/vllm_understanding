import os
import time
import re
import json
import multiprocessing
import numpy as np
import math
from pathlib import Path
from collections import defaultdict
import argparse
import base64
from io import BytesIO
from abc import ABC, abstractmethod
import logging

# Try importing necessary libraries, provide instructions if missing
try:
    from google import genai
    from google.genai import types as google_types
except ImportError:
    print("Google Generative AI library not found. Please install using: pip install google-generativeai")
    exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found. Please install using: pip install openai")
    exit(1)

# OpenCV no longer needed for Qwen
# try:
#     import cv2
# except ImportError:
#     print("OpenCV library not found. Please install using: pip install opencv-python")
#     exit(1)

# PIL no longer needed for Qwen
# try:
#     from PIL import Image
# except ImportError:
#     print("PIL (Pillow) library not found. Please install using: pip install Pillow")
#     exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置 API 密钥 and Client
# genai.configure(api_key=os.environ["GEMINI_API_KEY"]) # Configure is deprecated with Client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Remove old upload_to_gemini function
# def upload_to_gemini(path, mime_type=None):
#     """
#     上传指定文件到 Gemini，并返回上传后的文件对象。
#     """
#     file = genai.upload_file(path, mime_type=mime_type)
#     print(f"Uploaded file '{file.display_name}' as: {file.uri}")
#     return file

# Remove old wait_for_file_active function
# def wait_for_file_active(file):
#     """
#     等待上传的文件处理完成，直到状态变为 ACTIVE（失败则抛出异常）。
#     """
#     print("Waiting for file processing...", end="")
#     while True:
#         current_file = genai.get_file(file.name)
#         if current_file.state.name == "ACTIVE":
#             print(" done.")
#             return current_file
#         elif current_file.state.name == "FAILED":
#             raise Exception(f"File {file.name} failed to process")
#         else:
#             print(".", end="", flush=True)
#             time.sleep(10)

def extract_movement_info(filename):
    # Extract movement name and side (if any)
    movement_match = re.search(r'_(.*?)(?:\((L|R)\))?_', filename)
    if movement_match:
        movement = movement_match.group(1)
        side = movement_match.group(2)
        if side == 'L':
            movement = f'left {movement}'
        elif side == 'R':
            movement = f'right {movement}'
    else:
        movement = None

    # Extract ground truth numbers
    ground_truth_match = re.search(r'\[(\d+(-\d+)*)\]', filename)
    if ground_truth_match:
        ground_truth = ground_truth_match.group(1)
    else:
        ground_truth = None

    return movement, ground_truth

def extract_boxed_answer(text):
    """Extract the number from the \\boxed{} in the response"""
    if not text: return None
    match = re.search(r'\\boxed\{(\d+)\}', text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Found \\boxed{{}}, but content was not an integer: {match.group(1)}")
            return None
    logger.warning(f"Could not find \\boxed{{...}} in response.")
    return None

# --- Abstract Base Class for Processors ---

class VideoProcessor(ABC):
    def __init__(self, config):
        self.config = config
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        pass

    @abstractmethod
    def process_single_video(self, video_path: str, movement: str, prompt_template: str) -> dict:
        """Processes a single video and returns result data."""
        pass

    def _get_common_result_data(self, video_path: str, movement: str, ground_truth: str):
        """Helper to get common fields for the result dictionary."""
        gt_number = None
        if ground_truth:
            parts = ground_truth.split('-')
            try:
                gt_number = int(parts[0])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert ground truth '{ground_truth}' to a number for {video_path}")

        return {
             "file": os.path.basename(video_path),
             "video_path": video_path,
             "movement": movement,
             "ground_truth": ground_truth,
             "ground_truth_number": gt_number,
             "generated_response": None, # To be filled by specific processor
             "predicted_count": None,    # To be filled by specific processor
             "error": None               # To store potential errors
        }

# --- Gemini Processor Implementation ---

class GeminiProcessor(VideoProcessor):
    def _initialize_client(self):
        logger.info("Initializing Gemini Client...")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        return genai.Client(api_key=api_key)

    def process_single_video(self, video_path: str, movement: str, prompt_template: str) -> dict:
        _, ground_truth = extract_movement_info(os.path.basename(video_path))
        result = self._get_common_result_data(video_path, movement, ground_truth)
        model_name = self.config.get('model_name', "gemini-1.5-flash-latest") # Default to 1.5 flash
        result["model_name"] = model_name  # Add model name to result for tracking

        try:
            # 1. Upload file
            logger.info(f"[Gemini - {model_name}] Uploading file: {video_path}")
            uploaded_file = self.client.files.upload(file=video_path)
            logger.info(f"[Gemini - {model_name}] Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")

            # 2. Wait for processing
            logger.info(f"[Gemini - {model_name}] Waiting for file processing...")
            while uploaded_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(5) # Shorter sleep for Gemini
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name != "ACTIVE":
                error_msg = f"[Gemini - {model_name}] File processing failed or did not complete: {uploaded_file.state.name} - {uploaded_file.uri}"
                logger.error(error_msg)
                result["error"] = error_msg
                # Attempt to delete the failed file
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.info(f"[Gemini - {model_name}] Deleted non-ACTIVE file: {uploaded_file.name}")
                except Exception as delete_err:
                    logger.warning(f"[Gemini - {model_name}] Failed to delete non-ACTIVE file {uploaded_file.name}: {delete_err}")
                return result
            print(" Done.")
            logger.info(f"[Gemini - {model_name}] File processing complete.")

            # 3. Prepare Prompt & Config
            prompt_text = prompt_template.format(movement=movement)
            generation_config_dict = self.config.get('generation_config', {})
            generation_config = google_types.GenerateContentConfig(**generation_config_dict)

            # 4. Generate Content
            logger.info(f"[Gemini - {model_name}] Generating content for {result['file']}...")
            response = self.client.models.generate_content(
                model=model_name,
                contents=[uploaded_file, prompt_text],
                config=generation_config
            )

            # 5. Process Response
            generated_text = response.text
            logger.info(f"[Gemini - {model_name}] Response received for {result['file']}. Summary: {generated_text[:100]}...")
            result["generated_response"] = generated_text
            result["predicted_count"] = extract_boxed_answer(generated_text)

            # 6. Delete uploaded file (optional, but good practice)
            try:
                self.client.files.delete(name=uploaded_file.name)
                logger.info(f"[Gemini - {model_name}] Deleted processed file: {uploaded_file.name}")
            except Exception as delete_err:
                logger.warning(f"[Gemini - {model_name}] Failed to delete processed file {uploaded_file.name}: {delete_err}")

        except Exception as e:
            error_msg = f"[Gemini - {model_name}] Error processing {video_path}: {str(e)}"
            logger.exception(error_msg) # Log full traceback
            result["error"] = error_msg
            # Ensure file is deleted if it exists and failed mid-process
            if 'uploaded_file' in locals() and hasattr(uploaded_file, 'name'):
                 try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.info(f"[Gemini - {model_name}] Deleted file after error: {uploaded_file.name}")
                 except Exception as delete_err:
                     logger.warning(f"[Gemini - {model_name}] Failed to delete file {uploaded_file.name} after error: {delete_err}")

        return result

# --- Qwen Processor Implementation ---

class QwenProcessor(VideoProcessor):
    def _initialize_client(self):
        logger.info("Initializing Qwen OpenAI **Synchronous** Client...")
        api_key = self.config.get("qwen_api_key", "EMPTY")
        base_url = self.config.get("qwen_api_url")
        if not base_url:
            raise ValueError("Qwen API URL (--qwen_api_url) must be provided.")
        # Use the synchronous client
        return OpenAI(api_key=api_key, base_url=base_url)

    def process_single_video(self, video_path: str, movement: str, prompt_template: str) -> dict:
         _, ground_truth = extract_movement_info(os.path.basename(video_path))
         result = self._get_common_result_data(video_path, movement, ground_truth)
         
         # Ensure model_name is never None - FIXED BUG HERE
         model_name = self.config.get('model_name')
         if not model_name:
             model_name = "Qwen2.5-VL-7B-Instruct" # Default model if none specified
             logger.info(f"No model name provided, using default: {model_name}")
             
         result["model_name"] = model_name  # Add model name to result for tracking
         generation_config_dict = self.config.get('generation_config', {})

         try:
             # 1. Read and Encode Full Video
             logger.info(f"[Qwen - {model_name}] Reading and encoding video file: {video_path}")
             with open(video_path, "rb") as video_file:
                 video_bytes = video_file.read()
             base64_video = base64.b64encode(video_bytes).decode('utf-8')
             video_data_url = f"data:video/mp4;base64,{base64_video}"
             logger.info(f"[Qwen - {model_name}] Video encoded to data URL (length: {len(video_data_url)} chars).")

             # 2. Prepare Prompt & Messages (OpenAI format)
             prompt_text = prompt_template.format(movement=movement)
             messages = [
                 {"role": "user", "content": [
                     # Send the video data URL
                     {"type": "video_url", "video_url": {"url": video_data_url}},
                     # Add text prompt last
                     {"type": "text", "text": prompt_text}
                 ]}
             ]

             # 3. Generate Content (Synchronous Call)
             logger.info(f"[Qwen - {model_name}] Generating content for {result['file']}...")
             api_params = {
                 "temperature": generation_config_dict.get("temperature", 0),
                 "top_p": generation_config_dict.get("top_p", 1.0),
                 "max_tokens": generation_config_dict.get("max_output_tokens", 8192),
             }
             api_params = {k: v for k, v in api_params.items() if v is not None}

             # Direct synchronous call - double check model is not None
             logger.info(f"[Qwen] Making API call with model={model_name}")
             response = self.client.chat.completions.create(
                 model=model_name,  # This should now never be None
                 messages=messages,
                 **api_params
             )
             generated_text = response.choices[0].message.content

             # 4. Process Response
             logger.info(f"[Qwen - {model_name}] Response received for {result['file']}. Summary: {generated_text[:100]}...")
             result["generated_response"] = generated_text
             result["predicted_count"] = extract_boxed_answer(generated_text)

         except FileNotFoundError:
             error_msg = f"[Qwen - {model_name}] Video file not found: {video_path}"
             logger.error(error_msg)
             result["error"] = error_msg
         except Exception as e:
            error_msg = f"[Qwen - {model_name}] Error processing {video_path}: {str(e)}"
            logger.exception(error_msg)
            result["error"] = error_msg

         return result


# --- Multiprocessing Worker ---

def worker_function(args):
    """Top-level function for multiprocessing pool."""
    processor_config, service, video_path, prompt_template = args
    try:
        # Create processor instance in the worker process to avoid pickling issues
        if service == 'gemini':
            processor = GeminiProcessor(processor_config)
        elif service == 'qwen':
            processor = QwenProcessor(processor_config)
        else:
            raise ValueError(f"Unsupported service: {service}")
            
        # Extract movement here to pass to processor
        file = os.path.basename(video_path)
        movement, _ = extract_movement_info(file)
        if not movement:
            logger.warning(f"Could not extract movement from filename: {file}. Skipping.")
            return None
        return processor.process_single_video(video_path, movement, prompt_template)
    except Exception as e:
        logger.error(f"Unhandled exception in worker for {video_path}: {e}", exc_info=True)
        # Return a basic error structure if needed
        return {
             "file": os.path.basename(video_path),
             "video_path": video_path,
             "error": f"Worker exception: {str(e)}"
        }

# --- Metrics Calculation (Keep Existing) ---

def calculate_metrics(results):
    """Calculate MAE and RMSE from the results"""
    # Filter results where both ground truth and prediction are available
    valid_pairs = [(r["ground_truth_number"], r["predicted_count"])
                  for r in results
                  if r is not None and r.get("error") is None and # Check for None result and error field
                     r.get("ground_truth_number") is not None and r.get("predicted_count") is not None]

    if not valid_pairs:
        return {
            "mae": None,
            "rmse": None,
            "num_samples": 0
        }

    ground_truths, predictions = zip(*valid_pairs)

    # Calculate errors
    errors = [abs(gt - pred) for gt, pred in valid_pairs]
    squared_errors = [error**2 for error in errors]

    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

    return {
        "mae": mae,
        "rmse": rmse,
        "num_samples": len(valid_pairs)
    }

# --- Get model-specific folder name ---
def get_model_folder_name(model_name):
    """Convert model name to a safe folder name."""
    if not model_name:
        return "default_model"
    # Replace problematic characters with underscore
    safe_name = re.sub(r'[^\w\-\.]', '_', model_name)
    return safe_name

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Batch process videos for action counting using Gemini or Qwen.")
    parser.add_argument("--service", type=str, required=True, choices=['gemini', 'qwen'], help="LLM service to use.")
    parser.add_argument("--downloads_dir", type=str, default='downloads', help="Directory containing workout subdirectories.")
    parser.add_argument("--output_dir", type=str, default='analysis_results', help="Directory to save results.")
    parser.add_argument("--model_name", type=str, default=None, help="Model name to use (optional, defaults vary by service). Example: gemini-1.5-flash-latest or qwen-vl-plus")
    parser.add_argument("--qwen_api_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL for Qwen OpenAI-compatible API (required if service is qwen).")
    parser.add_argument("--qwen_api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY", "EMPTY"), help="API key for Qwen service.")
    parser.add_argument("--pool_size", type=int, default=None, help="Number of worker processes (defaults to cpu_count // 2).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p probability mass.")
    parser.add_argument("--max_output_tokens", type=int, default=8192, help="Maximum output tokens.")

    args = parser.parse_args()

    # --- Configuration ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model-specific subfolder
    model_folder_name = get_model_folder_name(args.model_name)
    model_output_dir = os.path.join(args.output_dir, model_folder_name)
    os.makedirs(model_output_dir, exist_ok=True)
    logger.info(f"Results will be saved to model-specific folder: {model_output_dir}")

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "response_mime_type": "text/plain", # Primarily for Gemini, ignored by Qwen processor
    }

    processor_config = {
        "model_name": args.model_name,
        "generation_config": generation_config,
        "qwen_api_url": args.qwen_api_url,
        "qwen_api_key": args.qwen_api_key,
    }

    # --- Prepare Video List and Arguments ---
    all_video_files = []
    workout_dirs = [os.path.join(args.downloads_dir, d) for d in os.listdir(args.downloads_dir)
                    if os.path.isdir(os.path.join(args.downloads_dir, d))]

    for workout_dir in workout_dirs:
        clips_path = os.path.join(workout_dir, "clips")
        if os.path.exists(clips_path):
            for f in os.listdir(clips_path):
                if f.endswith('.mp4'):
                    all_video_files.append(os.path.join(clips_path, f))

    if not all_video_files:
        logger.error(f"No .mp4 files found in any 'clips' subdirectories within {args.downloads_dir}. Exiting.")
        return

    logger.info(f"Found {len(all_video_files)} total video files to process.")

    prompt_template = "This is a video of a video blogger performing the {movement} movement. Please count how many {movement} movements the blogger has completed in total. Note that the blogger's movement frequency may not remain constant, so please count each movement carefully one by one. Analyze the video frame by frame and provide the reasoning behind your answer. Please reason step by step, and put your final answer within \\boxed{{}}"

    # Pass configuration rather than processor instance to avoid pickling errors
    worker_args_list = [(processor_config, args.service, video_path, prompt_template) for video_path in all_video_files]

    # --- Run Multiprocessing ---
    all_results = []
    default_pool_size = (multiprocessing.cpu_count() // 2) if multiprocessing.cpu_count() > 1 else 1
    pool_size = args.pool_size if args.pool_size is not None else default_pool_size
    pool_size = min(pool_size, len(all_video_files)) # Cannot have more workers than tasks
    logger.info(f"Using {pool_size} worker processes.")

    if pool_size > 0:
        with multiprocessing.Pool(processes=pool_size) as pool:
            results_iterator = pool.imap_unordered(worker_function, worker_args_list)
            # Process results as they complete
            for i, result in enumerate(results_iterator):
                if result: # Filter out None results from worker errors
                    all_results.append(result)
                logger.info(f"Progress: {i + 1}/{len(all_video_files)} videos processed.")
    else:
        logger.warning("Pool size is 0, skipping parallel processing.")
        # Optional: Add sequential execution fallback
        # for worker_args in worker_args_list:
        #     result = worker_function(worker_args)
        #     if result:
        #         all_results.append(result)

    logger.info(f"Processed {len(all_results)} videos successfully.")

    # --- Save Results to model-specific folder ---
    results_file = os.path.join(model_output_dir, 'all_results.jsonl')
    with open(results_file, 'w') as f:
        for result in all_results:
            # Ensure result is json serializable (convert numpy types if any)
            f.write(json.dumps(result, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x)) + '\n')
    logger.info(f"Results saved to {results_file}")

    # --- Calculate & Save Metrics ---
    metrics = calculate_metrics(all_results)
    metrics_file = os.path.join(model_output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    # Print metrics
    print("\nPerformance Metrics:")
    print(f"Service: {args.service}")
    print(f"Model: {args.model_name or ('Default Gemini' if args.service == 'gemini' else 'Default Qwen')}")
    print(f"Number of valid samples: {metrics['num_samples']}")
    if metrics['mae'] is not None:
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.4f}")
    else:
        print("No valid pairs of ground truth and predictions found to calculate metrics.")

    # Calculate & Save Per-Movement Metrics
    movements = defaultdict(list)
    for result in all_results:
         if result is not None and result.get("error") is None and result.get("movement") and result.get("ground_truth_number") is not None and result.get("predicted_count") is not None:
            movements[result["movement"]].append((result["ground_truth_number"], result["predicted_count"]))

    movement_metrics = {}
    for movement, pairs in movements.items():
        if pairs:
            ground_truths, predictions = zip(*pairs)
            errors = [abs(gt - pred) for gt, pred in pairs]
            squared_errors = [error**2 for error in errors]

            mae = sum(errors) / len(errors)
            rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

            movement_metrics[movement] = {
                "mae": mae,
                "rmse": rmse,
                "num_samples": len(pairs)
            }

    movement_metrics_file = os.path.join(model_output_dir, 'movement_metrics.json')
    if movement_metrics:
        with open(movement_metrics_file, 'w') as f:
            json.dump(movement_metrics, f, indent=2, sort_keys=True)
        logger.info(f"Per-movement metrics saved to {movement_metrics_file}")

        # Print per-movement metrics
        print("\nPer-Movement Metrics:")
        for movement in sorted(movement_metrics.keys()):
            metrics = movement_metrics[movement]
            print(f"\n  {movement}:")
            print(f"    Number of samples: {metrics['num_samples']}")
            print(f"    MAE: {metrics['mae']:.4f}")
            print(f"    RMSE: {metrics['rmse']:.4f}")
    else:
         logger.info("No valid per-movement metrics to calculate or save.")


if __name__ == "__main__":
    main()
