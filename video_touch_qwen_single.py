import os
import time
import re
import base64
import logging
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import OpenAI client for Qwen API
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found. Please install using: pip install openai")
    exit(1)

# Qwen API endpoint
QWEN_API_URL = "https://f500-129-80-126-97.ngrok-free.app/v1"
QWEN_API_KEY = "EMPTY"  # Qwen doesn't require a real API key with ngrok setup

QWEN_API_KEY= os.getenv("DASHSCOPE_API_KEY")
QWEN_API_URL= "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Function to extract boxed answer from response
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

# Specify the video to test
video_path = "downloads/7a184635bb8578128a14c25954e0a873/clips/0412_circle + knee tuck_[8].mp4"
if not os.path.exists(video_path):
    logger.error(f"Video file not found: {video_path}")
    exit(1)

# Initialize OpenAI client with Qwen endpoint
logger.info(f"Initializing OpenAI client with Qwen API URL: {QWEN_API_URL}")
client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_URL)

# Read and encode video file
logger.info(f"Reading and encoding video file: {video_path}")
try:
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    base64_video = base64.b64encode(video_bytes).decode('utf-8')
    video_data_url = f"data:video/mp4;base64,{base64_video}"
except Exception as e:
    logger.error(f"Error reading video: {str(e)}")
    exit(1)

# Prepare prompt
movement = "side stretch"
prompt_text = f"This is a video of a video blogger performing the {movement} movement. Please count how many {movement} movements the blogger has completed in total. Note that the blogger's movement frequency may not remain constant, so please count each movement carefully one by one. Analyze the video frame by frame and provide the reasoning behind your answer. Please reason step by step, and put your final answer within \\boxed{{}}"

# Prepare messages in OpenAI format
messages = [
    {"role": "user", "content": [
        # Send the video data URL
        {"type": "video_url", "video_url": {"url": video_data_url}, 'fps': 20.0},
        # Add text prompt last
        {"type": "text", "text": prompt_text}
    ]}
]

# Generation parameters
model_name = "qwen2.5-vl-32b-instruct"  # Use the Qwen model name
generation_params = {
    "temperature": 0.0,
    "top_p": 0.95, 
    "max_tokens": 8192,
}

# Call the Qwen API
logger.info(f"Calling Qwen API with model: {model_name}")
try:
    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **generation_params
    )
    end_time = time.time()
    
    # Get the response text
    generated_text = response.choices[0].message.content
    
    # Extract the predicted count
    predicted_count = extract_boxed_answer(generated_text)
    
    # Print results
    print("\n" + "="*50)
    print(f"RESPONSE (took {end_time - start_time:.2f} seconds):")
    print("="*50)
    print(generated_text)
    print("\n" + "="*50)
    print(f"Predicted count: {predicted_count}")
    print("="*50)
    
except Exception as e:
    logger.error(f"Error calling Qwen API: {str(e)}", exc_info=True) 