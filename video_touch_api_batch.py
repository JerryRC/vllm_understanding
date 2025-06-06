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

try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI library not found. Install with: pip install google-generativeai")
    # Decide if you want to exit or continue (maybe Qwen is still usable)
    # exit(1)

try:
    from dashscope import MultiModalConversation
except ImportError:
    print("Dashscope library not found. Install with: pip install dashscope")
    # exit(1)

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# 配置 API 密钥 and Client
# No need to configure the old way anymore
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
    movement_match = re.search(r'_(.*?)(?:\((L|R|l|r)\))_', filename)
    if movement_match:
        movement = movement_match.group(1)
        side = movement_match.group(2)
        if side and side.upper() == 'L':
            movement = f'left {movement}'
        elif side and side.upper() == 'R':
            movement = f'right {movement}'
    else:
        movement = None
        
    # 检查中括号模式 [L|R]
    bracket_match = re.search(r'_(.*?)\[(L|R|l|r)\]_', filename)
    if not movement and bracket_match:
        movement = bracket_match.group(1)
        side = bracket_match.group(2)
        if side and side.upper() == 'L':
            movement = f'left {movement}'
        elif side and side.upper() == 'R':
            movement = f'right {movement}'
    
    # 如果两种模式都没匹配到，再尝试匹配没有方向的动作名称
    if not movement:
        basic_match = re.search(r'_(.*?)_', filename)
        if basic_match:
            movement = basic_match.group(1)

    # Extract ground truth numbers
    ground_truth_match = re.search(r'\[(\d+(?:-\d+)*)\]', filename)
    if ground_truth_match:
        try:
            # Convert to integer if it's a single number, otherwise keep as string
            ground_truth = ground_truth_match.group(1)
            if '-' not in ground_truth:
                ground_truth = int(ground_truth)
        except ValueError:
            ground_truth = None
    else:
        ground_truth = None

    return movement, ground_truth

def extract_boxed_answer(text):
    """Extract the number from the \\boxed{} in the response"""
    if not text: return None
    match = re.search(r'\\boxed\{(\d+(?:\.\d+)?)\}', text) # Allow float/int
    if match:
        try:
            # Try converting to int first, then float if needed
            val_str = match.group(1)
            if '.' in val_str:
                return float(val_str)
            else:
                return int(val_str)
        except ValueError:
            logger.warning(f"Found \\boxed{{}}, but content was not a valid number: {match.group(1)}")
            return None
    logger.warning(f"Could not find \\boxed{{...}} in response: {text[:200]}...")
    return None

# --- Abstract Base Class for Processors ---

class VideoProcessor(ABC):
    def __init__(self, config):
        self.config = config
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        pass

    @abstractmethod
    def process_single_video(self, video_path: str, movement: str, prompt_template: str) -> dict:
        """Processes a single video and returns result data."""
        pass

    def _get_common_result_data(self, video_path: str, movement: str, ground_truth_number: int, fps: int = None) -> dict:
        """Creates the basic dictionary structure for a result."""
        filename = os.path.basename(video_path)
        return {
            "file": filename,
            "video_path": video_path,
            "movement": movement,
            "ground_truth_number": ground_truth_number,
            "fps": fps, # Add fps here
            "generated_response": None,
            "predicted_count": None,
            "error": None,
            "model_name": None # Will be filled later
        }

# --- Gemini Processor Implementation ---

class GeminiProcessor(VideoProcessor):
    def _initialize_client(self):
        # Configure the Gemini client using the new API
        try:
            self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
            if not os.getenv('GEMINI_API_KEY'):
                logger.warning("GEMINI_API_KEY environment variable not set. Gemini API calls may fail.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            # Consider raising the exception or handling it based on desired behavior

    def process_single_video(self, video_path: str, movement: str, prompt_template: str) -> dict:
        model_name = self.config.get('model_name', 'gemini-2.0-flash') # Updated default model
        generation_config = self.config.get('generation_config', {})
        _, ground_truth_number = extract_movement_info(os.path.basename(video_path))

        result = self._get_common_result_data(video_path, movement, ground_truth_number)
        result["model_name"] = model_name # Add model name specific to this processor call

        try:
            logger.info(f"[Gemini - {model_name}] Processing video: {result['file']} for movement: '{movement}'")

            # 1. Upload Video using new client API
            logger.debug(f"[Gemini - {model_name}] Uploading video: {result['file']}...")
            video_file = self.client.files.upload(file=video_path)
            logger.debug(f"[Gemini - {model_name}] Upload successful. Waiting for processing...")

            # 2. Poll for Video Processing Completion with new API
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = self.client.files.get(name=video_file.name)
                logger.debug(f"[Gemini - {model_name}] Video state: {video_file.state.name}")

            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed.")
            elif video_file.state.name != "ACTIVE":
                raise ValueError(f"Video is not active. State: {video_file.state.name}")

            logger.info(f"[Gemini - {model_name}] Video ready for model: {result['file']}")

            # 3. Prepare Prompt and Call Model with new API
            prompt = prompt_template.format(movement=movement)
            
            # Create the config object from the dictionary using proper types
            config = google_types.GenerateContentConfig(
                temperature=generation_config.get('temperature', 0.0),
                top_p=generation_config.get('top_p', 1.0),
                top_k=generation_config.get('top_k', 32),
                max_output_tokens=generation_config.get('max_output_tokens', 8192),
                response_mime_type=generation_config.get('response_mime_type', 'text/plain'),
            )
            
            logger.info(f"[Gemini - {model_name}] Sending prompt to model...")
            # Ensure we're using a valid model name - use a known working model if not specified
            actual_model = model_name if model_name else "gemini-2.0-flash"
            logger.info(f"[Gemini] Using model: {actual_model}")
            
            response = self.client.models.generate_content(
                model=actual_model,
                contents=[prompt, video_file],
                config=config
            )

            # 4. Process Response
            generated_text = response.text
            logger.info(f"[Gemini - {model_name}] Response received for {result['file']}. Summary: {generated_text[:100]}...")
            result["generated_response"] = generated_text
            result["predicted_count"] = extract_boxed_answer(generated_text)

            # Clean up uploaded file with new API
            try:
                self.client.files.delete(name=video_file.name)
                logger.debug(f"[Gemini - {model_name}] Deleted uploaded file: {video_file.name}")
            except Exception as delete_err:
                logger.warning(f"[Gemini - {model_name}] Failed to delete uploaded file {video_file.name}: {delete_err}")

        except FileNotFoundError:
            error_msg = f"[Gemini - {model_name}] Video file not found: {video_path}"
            logger.error(error_msg)
            result["error"] = error_msg
        except Exception as e:
            error_msg = f"[Gemini - {model_name}] Error processing {video_path}: {str(e)}"
            logger.exception(error_msg) # Log full traceback
            result["error"] = error_msg
            # Attempt to clean up if video_file was created
            if 'video_file' in locals() and hasattr(video_file, 'name'):
                try:
                    self.client.files.delete(name=video_file.name)
                    logger.debug(f"[Gemini - {model_name}] Deleted uploaded file on error: {video_file.name}")
                except Exception as delete_err:
                    logger.warning(f"[Gemini - {model_name}] Failed to delete file {video_file.name} on error: {delete_err}")

        return result

# --- Qwen Processor Implementation ---

class QwenProcessor(VideoProcessor):
    def _initialize_client(self):
        # Check for Dashscope API Key
        self.api_key = self.config.get('qwen_api_key')
        if not self.api_key or self.api_key == "EMPTY":
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
            if not self.api_key:
                logger.error("Qwen API Key not provided in config or DASHSCOPE_API_KEY env var.")
                # Decide whether to raise error or allow proceeding (will fail later)
                # raise ValueError("Missing Qwen API Key")
        
        # No explicit client object needed for dashscope library calls
        pass

    def process_single_video(self, video_path: str, movement: str, prompt_template: str) -> dict:
        model_name = self.config.get('model_name', 'qwen-vl-plus') # Default Qwen model
        fps = self.config.get('qwen_fps', 2) # Get FPS from config
        _, ground_truth_number = extract_movement_info(os.path.basename(video_path))
        
        # Pass FPS to common data function
        result = self._get_common_result_data(video_path, movement, ground_truth_number, fps=fps)
        result["model_name"] = model_name # Add model name specific to this processor call
        
        # Ensure API key is set (checked in init, but double-check)
        if not self.api_key:
            result["error"] = f"[Qwen - {model_name}] Missing API Key."
            logger.error(result["error"])
            return result
            
        try:
            logger.info(f"[Qwen - {model_name} - FPS {fps}] Processing video: {result['file']} for movement: '{movement}'")

            # 1. Get absolute path for file URI
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file does not exist: {video_path}")
            absolute_video_path = os.path.abspath(video_path)
            video_uri = f"file://{absolute_video_path}"
            logger.debug(f"[Qwen - {model_name}] Using video URI: {video_uri}")

            # 2. Prepare Prompt and Messages for Dashscope
            prompt = prompt_template.format(movement=movement)
            messages = [
                {'role': 'system', 'content': [{'text': 'You are a helpful assistant specialized in analyzing videos to count actions.'}]},
                {'role': 'user', 'content': [
                    {'video': video_uri, "fps": fps},
                    {'text': prompt}
                ]}
            ]

            # 3. Call Dashscope API
            logger.info(f"[Qwen - {model_name}] Sending request to Dashscope API...")
            start_time = time.time()
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model=model_name,
                messages=messages
                # Add other parameters like temperature if needed, check dashscope docs
            )
            end_time = time.time()
            logger.info(f"[Qwen - {model_name}] Call completed in {end_time - start_time:.2f} seconds. Status: {response.status_code}")

            # 4. Process Response
            if response.status_code == 200:
                generated_text = response["output"]["choices"][0]["message"]["content"][0]["text"]
                logger.info(f"[Qwen - {model_name}] Response received for {result['file']}. Summary: {generated_text[:100]}...")
                result["generated_response"] = generated_text
                result["predicted_count"] = extract_boxed_answer(generated_text)
            else:
                # Handle API errors gracefully
                error_code = response.get("code", "N/A")
                error_msg_api = response.get("message", "Unknown API Error")
                error_msg = f"[Qwen - {model_name}] Dashscope API Error (Status: {response.status_code}, Code: {error_code}): {error_msg_api}"
                logger.error(f"{error_msg} (Request ID: {response.get('request_id', 'N/A')})")
                result["error"] = error_msg

        except FileNotFoundError as fnf_err:
            error_msg = f"[Qwen - {model_name}] Video file not found during processing: {str(fnf_err)}"
            logger.error(error_msg)
            result["error"] = error_msg
        except Exception as e:
            error_msg = f"[Qwen - {model_name}] Unhandled error processing {video_path}: {str(e)}"
            logger.exception(error_msg) # Log full traceback
            result["error"] = error_msg

        return result


# --- Multiprocessing Worker ---

def check_existing_results(video_path, model_name, service, fps, results_file):
    """
    Check if a video has already been successfully processed with the correct FPS for Qwen.
    
    Args:
        video_path: Path to the video file
        model_name: Name of the model
        service: Name of the service ('gemini' or 'qwen')
        fps: FPS value used (relevant for Qwen)
        results_file: Path to the all_results.jsonl file
        
    Returns:
        dict: Existing result if found and valid, None otherwise
    """
    if not os.path.exists(results_file):
        return None
        
    try:
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    # Basic checks for matching video and model
                    if (result.get('video_path') != video_path or 
                            result.get('model_name') != model_name):
                        continue
                    
                    # Qwen specific FPS check
                    if service == 'qwen' and result.get('fps') != fps:
                        continue # Found result for same video/model but wrong FPS
                        
                    # Check if the result was successful
                    if (result.get('generated_response') is not None and
                        result.get('predicted_count') is not None and
                        result.get('error') is None): # Explicitly check for no error
                        
                        fps_log = f" (FPS: {result.get('fps', 'N/A')})" if service == 'qwen' else ""
                        logger.info(f"Found existing valid result for {os.path.basename(video_path)}{fps_log}")
                        return result
                        
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line in {results_file}: {line.strip()}")
                    continue
    except Exception as e:
        logger.error(f"Error reading existing results file {results_file}: {e}")
    
    return None

def worker_function(args):
    """Top-level function for multiprocessing pool."""
    processor_config, service, video_path, prompt_template, output_dir = args
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
        movement, ground_truth = extract_movement_info(file)
        if not movement:
            logger.warning(f"Could not extract movement from filename: {file}. Skipping.")
            return None

        model_name = processor_config.get('model_name', service)
        qwen_fps = processor_config.get('qwen_fps') if service == 'qwen' else None
        model_folder_name = get_model_folder_name(model_name, service, qwen_fps)
        results_file = os.path.join(output_dir, model_folder_name, "all_results.jsonl")
        
        # Check if we have already successfully processed this video (passing service and fps)
        existing_result = check_existing_results(video_path, model_name, service, qwen_fps, results_file)
        if existing_result:
            return existing_result

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
def get_model_folder_name(model_name, service=None, fps=None):
    """Convert model name to a safe folder name, optionally appending FPS for Qwen."""
    if not model_name:
        base_name = "default_model"
    else:
        # Replace problematic characters with underscore
        base_name = re.sub(r'[^\w\-\.]', '_', model_name)
    
    if service == 'qwen' and fps is not None:
        return f"{base_name}-fps{fps}"
    else:
        return base_name

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Batch process videos for action counting using Gemini or Qwen.")
    parser.add_argument("--service", type=str, required=True, choices=['gemini', 'qwen'], help="LLM service to use.")
    parser.add_argument("--downloads_dir", type=str, default='downloads_repcount_og', help="Directory containing workout subdirectories.")
    parser.add_argument("--output_dir", type=str, default='analysis_results_repcount', help="Directory to save results.")
    parser.add_argument("--model_name", type=str, default=None, help="Model name to use (optional, defaults vary by service). Example: gemini-1.5-flash-latest or qwen-vl-plus")
    parser.add_argument("--qwen_api_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL for Qwen OpenAI-compatible API (required if service is qwen).")
    parser.add_argument("--qwen_api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY", "EMPTY"), help="API key for Qwen service.")
    parser.add_argument("--qwen_fps", type=int, default=2, help="Frames per second for Qwen video processing.")
    parser.add_argument("--pool_size", type=int, default=5, help="Number of worker processes (defaults to cpu_count // 2).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p probability mass.")
    parser.add_argument("--max_output_tokens", type=int, default=8192, help="Maximum output tokens.")

    args = parser.parse_args()

    # --- Configuration ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Set default model name if not provided
    if args.service == 'gemini' and not args.model_name:
        args.model_name = 'gemini-2.0-flash'
    elif args.service == 'qwen' and not args.model_name:
        args.model_name = 'qwen-vl-plus'
        
    logger.info(f"Using model name: {args.model_name}")

    # Create model-specific subfolder
    model_folder_name = get_model_folder_name(args.model_name, args.service, args.qwen_fps)
    model_output_dir = os.path.join(args.output_dir, model_folder_name)
    os.makedirs(model_output_dir, exist_ok=True)
    logger.info(f"Results will be saved to model-specific folder: {model_output_dir}")

    # Create a generation config dictionary that works with both Gemini and Qwen processors
    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "top_k": 32,  # Add top_k for Gemini compatibility
        "response_mime_type": "text/plain", # Primarily for Gemini, ignored by Qwen processor
    }

    processor_config = {
        "model_name": args.model_name,
        "generation_config": generation_config,
        "qwen_api_url": args.qwen_api_url,
        "qwen_api_key": args.qwen_api_key,
        "qwen_fps": args.qwen_fps,
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
    worker_args_list = [(processor_config, args.service, video_path, prompt_template, args.output_dir) for video_path in all_video_files]

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
