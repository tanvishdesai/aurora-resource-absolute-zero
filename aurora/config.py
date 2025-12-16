import os

# Model Configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"
# MODEL_PATH = "/kaggle/input/qwen-3/transformers/8b/1" # Legacy Qwen path

# Simulator Configuration
DEFAULT_TIME_STEP = 1.0  # seconds

# Evolution Configuration
INITIAL_ELO = 1500
K_FACTOR = 32

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
