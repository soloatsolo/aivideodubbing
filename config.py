import os
from pathlib import Path

def check_gpu_availability():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Base configuration
CONFIG = {
    'UPLOAD_FOLDER': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads'),
    'TEMP_FOLDER': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp'),
    'MAX_CONTENT_LENGTH': 500 * 1024 * 1024,  # 500MB max file size
    'ALLOWED_EXTENSIONS': {'mp4', 'avi', 'mov', 'mkv'},
    
    # Audio processing settings
    'AUDIO': {
        'SAMPLE_RATE': 44100,
        'CHANNELS': 2,
        'BACKGROUND_MUSIC_RATIO': 0.3,
        'MAX_AUDIO_DURATION': 3600,  # 1 hour in seconds
        'SPLEETER_CONFIG': {
            'model': '2stems',
            'pretrained_model': None,  # Use default pretrained model
            'multiprocess': True
        }
    },
    
    # Model settings
    'MODELS': {
        'WHISPER_MODEL': 'base',
        'TRANSLATION_MODELS': {
            'en-ar': 'Helsinki-NLP/opus-mt-en-ar',
            'fr-ar': 'Helsinki-NLP/opus-mt-fr-ar',
            'es-ar': 'Helsinki-NLP/opus-mt-es-ar',
            'de-ar': 'Helsinki-NLP/opus-mt-de-ar'
        },
        'TTS_MODEL': 'facebook/mms-tts-ara',
        'VOICE_CLONE_MODEL': 'facebook/fastspeech2-en-ljspeech'
    },
    
    # Processing options
    'PROCESSING': {
        'USE_GPU': check_gpu_availability(),
        'BATCH_SIZE': 16,
        'NUM_WORKERS': 4,
        'USE_VOICE_CLONING': True,  # Set to False to always use basic TTS
        'PRESERVE_MUSIC': True,  # Set to False to skip audio separation
        'ENABLE_LIP_SYNC': True,  # Enable lip sync by default
        'ENABLE_SUBTITLES': True  # Enable subtitles by default
    },
    
    # Subtitle settings
    'SUBTITLES': {
        'FONT': 'Arial',
        'FONT_SIZE': 24,
        'PRIMARY_COLOR': '&HFFFFFF&',  # White
        'OUTLINE_COLOR': '&H000000&',  # Black
        'OUTLINE_WIDTH': 2,
        'MIN_DURATION': 2000,  # Minimum subtitle duration in ms
        'MAX_DURATION': 5000,  # Maximum subtitle duration in ms
        'CHARS_PER_LINE': 42  # Maximum characters per line
    },
    
    # Language settings
    'LANGUAGES': {
        'en': 'الإنجليزية',
        'fr': 'الفرنسية',
        'es': 'الإسبانية',
        'de': 'الألمانية'
    }
}

# Create necessary directories
for folder in [CONFIG['UPLOAD_FOLDER'], CONFIG['TEMP_FOLDER']]:
    Path(folder).mkdir(parents=True, exist_ok=True)

def get_config(key, default=None):
    """Get a configuration value"""
    keys = key.split('.')
    value = CONFIG
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def update_config(key, value):
    """Update a configuration value"""
    keys = key.split('.')
    config = CONFIG
    for k in keys[:-1]:
        config = config.setdefault(k, {})
    config[keys[-1]] = value

# Initialize cleanup schedule for temp files
def schedule_cleanup():
    import threading
    import time
    import shutil
    
    def cleanup_old_files():
        while True:
            temp_dir = CONFIG['TEMP_FOLDER']
            if os.path.exists(temp_dir):
                current_time = time.time()
                for root, dirs, files in os.walk(temp_dir):
                    for name in files:
                        filepath = os.path.join(root, name)
                        # Remove files older than 1 hour
                        if current_time - os.path.getmtime(filepath) > 3600:
                            try:
                                os.remove(filepath)
                            except OSError:
                                continue
                    # Remove empty directories
                    for name in dirs:
                        try:
                            dirpath = os.path.join(root, name)
                            if not os.listdir(dirpath):
                                shutil.rmtree(dirpath)
                        except OSError:
                            continue
            time.sleep(3600)  # Run cleanup every hour
    
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()

# Start cleanup schedule
schedule_cleanup()