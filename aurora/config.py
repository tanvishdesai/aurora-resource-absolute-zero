import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
# API Configuration - Key Rotation Logic
API_KEYS = []

# Load the single default key
default_key = os.getenv("GEMINI_API_KEY")
if default_key:
    API_KEYS.append(default_key)

# Load numbered keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
i = 1
while True:
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if not key:
        break
    API_KEYS.append(key)
    i += 1

if not API_KEYS:
    print("⚠️  Warning: No GEMINI_API_KEYs found in environment variables. Please set them in a .env file.")
else:
    print(f"✅ Loaded {len(API_KEYS)} API keys.")

import itertools
_key_cycle = itertools.cycle(API_KEYS)

def get_next_api_key():
    """Returns the next API key from the rotation."""
    if not API_KEYS:
        return None
    return next(_key_cycle)

# Initialize with the first available key
CURRENT_API_KEY = get_next_api_key()
if CURRENT_API_KEY:
    genai.configure(api_key=CURRENT_API_KEY)

def rotate_api_key():
    """Rotates to the next API key and re-configures genai."""
    global CURRENT_API_KEY
    new_key = get_next_api_key()
    if new_key:
        print(f"🔄 Switching API Key...")
        CURRENT_API_KEY = new_key
        genai.configure(api_key=CURRENT_API_KEY)
        return new_key
    return None

# Model Configuration
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025" # Or gemini-1.5-flash if preferred/available
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

# Simulator Configuration
DEFAULT_TIME_STEP = 1.0  # seconds

# Evolution Configuration
INITIAL_ELO = 1500
K_FACTOR = 32

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
