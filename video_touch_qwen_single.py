import os
import time
import re
import logging
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Dashscope library
try:
    from dashscope import MultiModalConversation
except ImportError:
    print("Dashscope library not found. Please install using: pip install dashscope")
    exit(1)

# Use Dashscope API Key from environment variable
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    logger.error("DASHSCOPE_API_KEY environment variable not set.")
    exit(1)

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
local_video_path = "downloads/89e845984028a7ed30dfded20c0c00cb/clips/0341_lift curl(R)_[6].mp4"
if not os.path.exists(local_video_path):
    logger.error(f"Video file not found: {local_video_path}")
    exit(1)

# Get absolute path for the file URI
absolute_video_path = os.path.abspath(local_video_path)
video_uri = f"file://{absolute_video_path}"
logger.info(f"Using video file URI: {video_uri}")

# Prepare prompt
movement = "circle + knee tuck" # Updated based on the video path
prompt_text = f"This is a video of a video blogger performing the {movement} movement. Please count how many {movement} movements the blogger has completed in total. Note that the blogger's movement frequency may not remain constant, so please count each movement carefully one by one. Analyze the video frame by frame and provide the reasoning behind your answer. Please reason step by step, and put your final answer within \\boxed{{}}"

# Prepare messages in Dashscope format
messages = [
    {'role': 'system', 'content': [{'text': 'You are a helpful assistant specialized in analyzing videos to count actions.'}]},
    {'role': 'user', 'content': [
        # Send the video file URI
        {'video': video_uri, "fps": 2},
        # Add text prompt
        {'text': prompt_text}
    ]}
]

# Model name
model_name = "qwen2.5-vl-3b-instruct" # Use the specific Qwen model

# Call the Dashscope API
logger.info(f"Calling Dashscope API with model: {model_name}")
try:
    start_time = time.time()
    response = MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model=model_name,
        messages=messages
    )
    end_time = time.time()
    print(response)
    # Check response status
    if response.status_code == 200:
        # Get the response text
        generated_text = response["output"]["choices"][0]["message"]["content"][0]["text"]

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
    else:
        logger.error(f"Dashscope API call failed with status code {response.status_code}")
        logger.error(f"Request ID: {response.request_id}")
        logger.error(f"Error Code: {response.code}")
        logger.error(f"Error Message: {response.message}")

except Exception as e:
    logger.error(f"Error calling Dashscope API: {str(e)}", exc_info=True) 