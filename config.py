import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
MODELS_DIR = os.path.join(BASE_DIR, "models", "embeddings")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Default Settings
DEFAULT_SETTINGS = {
    "model_name": "Facenet512",
    "detector_backend": "opencv",
    "distance_metric": "cosine",
    "recognition_threshold": 0.45,
    "attendance_cooldown_seconds": 60,
    "process_every_n_frames": 3,
    "camera_index": 0,
    "min_samples_per_student": 3,
    "frame_width": 640,
    "frame_height": 480,
    "departments": [
        "Computer Science & Engineering",
        "Information Technology",
        "Electronics & Communication",
        "Electrical Engineering",
        "Mechanical Engineering",
        "Civil Engineering"
    ],
    "years": ["1st Year", "2nd Year", "3rd Year", "4th Year"],
    "sections": ["Section A", "Section B", "Section C", "Section D"]
}

def load_settings():
    """Load configuration settings from file or return defaults."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                user_settings = json.load(f)
                settings = DEFAULT_SETTINGS.copy()
                settings.update(user_settings)
                return settings
        except Exception as e:
            print(f"Error loading settings file: {e}")
    return DEFAULT_SETTINGS.copy()

def save_settings(new_settings):
    """Save user settings to JSON file."""
    try:
        current = load_settings()
        current.update(new_settings)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(current, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# Current Active Configuration
config = load_settings()
