import os
import torch
import logging
import functools
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration, pipeline
import whisper
from config import get_config
from pathlib import Path

logger = logging.getLogger("model_manager")

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_paths = {}
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.translation_cache = {}
        self.transcription_cache = {}
        self.max_cache_size = 100  # Maximum number of cached items
        
    def download_model(self, model_name):
        """Download and cache a model"""
        try:
            logger.info(f"Downloading model: {model_name}")
            if 'whisper' in model_name.lower():
                model = whisper.load_model(get_config('MODELS')['WHISPER_MODEL'])
            else:
                model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name, cache_dir=self.cache_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                model = pipeline("translation", model=model, tokenizer=tokenizer)
            
            self.models[model_name] = model
            logger.info(f"Successfully downloaded model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            return None
    
    def get_model(self, model_name, force_download=False):
        """Get a model, downloading it if necessary"""
        if model_name not in self.models or force_download:
            self.models[model_name] = self.download_model(model_name)
        return self.models.get(model_name)
    
    def verify_models(self):
        """Verify all required models are available"""
        required_models = [
            get_config('MODELS')['WHISPER_MODEL'],
            get_config('MODELS')['TRANSLATION_MODEL'],
            get_config('MODELS')['TTS_MODEL']
        ]
        
        missing_models = []
        for model_name in required_models:
            if not self.get_model(model_name):
                missing_models.append(model_name)
        
        return len(missing_models) == 0, missing_models
    
    def clear_cache(self):
        """Clear the model cache"""
        try:
            for model in self.models.values():
                if hasattr(model, 'to'):
                    model.to('cpu')
            torch.cuda.empty_cache()
            self.models.clear()
            logger.info("Successfully cleared model cache")
        except Exception as e:
            logger.error(f"Error clearing model cache: {str(e)}")
    
    def use_gpu(self, model_name):
        """Move a model to GPU if available"""
        if torch.cuda.is_available() and get_config('PROCESSING')['USE_GPU']:
            try:
                model = self.models.get(model_name)
                if model and hasattr(model, 'to'):
                    model.to('cuda')
                    logger.info(f"Moved model {model_name} to GPU")
                    return True
            except Exception as e:
                logger.error(f"Error moving model {model_name} to GPU: {str(e)}")
        return False
    
    @functools.lru_cache(maxsize=32)
    def handle_translation_error(self, text, retries=3):
        """Handle translation errors with retries and fallback with caching"""
        cache_key = f"{text}"
        if cache_key in self.translation_cache:
            logger.info("Translation cache hit")
            return self.translation_cache[cache_key]

        for i in range(retries):
            try:
                model = self.get_model(get_config('MODELS')['TRANSLATION_MODEL'], force_download=(i > 0))
                result = model(text)
                if result and result[0].get('translation_text'):
                    translation = result[0]['translation_text']
                    # Add to cache
                    if len(self.translation_cache) >= self.max_cache_size:
                        # Remove oldest entry if cache is full
                        self.translation_cache.pop(next(iter(self.translation_cache)))
                    self.translation_cache[cache_key] = translation
                    return translation
            except Exception as e:
                logger.error(f"Translation attempt {i+1} failed: {str(e)}")
                if i == retries - 1:
                    raise RuntimeError(f"فشلت جميع محاولات الترجمة: {str(e)}")
        
        return None

    @functools.lru_cache(maxsize=32)
    def transcribe_audio(self, audio_path):
        """Transcribe audio with caching"""
        cache_key = f"{audio_path}"
        if cache_key in self.transcription_cache:
            logger.info("Transcription cache hit")
            return self.transcription_cache[cache_key]

        try:
            model = self.get_model(get_config('MODELS')['WHISPER_MODEL'])
            result = model.transcribe(audio_path)
            if result and result.get("text"):
                transcription = result["text"]
                # Add to cache
                if len(self.transcription_cache) >= self.max_cache_size:
                    # Remove oldest entry if cache is full
                    self.transcription_cache.pop(next(iter(self.transcription_cache)))
                self.transcription_cache[cache_key] = transcription
                return transcription
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise RuntimeError(f"فشل في تحويل الصوت إلى نص: {str(e)}")
        
        return None