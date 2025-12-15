
# Model Configuration
MODEL_PATH = "/kaggle/input/qwen2.5/transformers/14b-instruct/1" # Default Kaggle path

# Simulator Configuration
DEFAULT_TIME_STEP = 1.0  # seconds

# Evolution Configuration
INITIAL_ELO = 1500
K_FACTOR = 32

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
