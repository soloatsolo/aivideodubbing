import sys
import subprocess
import logging
import logging.handlers
import tempfile
import shutil
import os
from pathlib import Path
from flask import Flask, request, send_file, render_template_string, jsonify, Response, stream_with_context
from flask_cors import CORS
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from transformers import pipeline, VitsModel, AutoProcessor
import whisper
from gtts import gTTS
import moviepy.editor as mp
from datetime import datetime
import queue
import threading

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger("video_dubber")
logger.setLevel(logging.DEBUG)

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(log_dir, "video_dubber.log"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatters
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(levelname)s: %(message)s')

file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Flask application setup
app = Flask(__name__)
CORS(app)

class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

class TranslationError(Exception):
    """Custom exception for translation errors"""
    pass

# Global progress queue
progress_queues = {}

class ProgressTracker:
    def __init__(self, session_id):
        self.queue = queue.Queue()
        self.session_id = session_id
        progress_queues[session_id] = self.queue

    def update_progress(self, stage, progress, message):
        try:
            self.queue.put({
                'stage': stage,
                'progress': progress,
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")

    def cleanup(self):
        if self.session_id in progress_queues:
            del progress_queues[self.session_id]

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 44100
        
    def separate_audio(self, audio_path, progress_tracker=None):
        """Separate vocals from background music"""
        try:
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 0, 'بدء فصل الصوت')
            
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 30, 'تحليل المقطع الصوتي')
            
            # Perform source separation using librosa
            S_full, phase = librosa.magphase(librosa.stft(y))
            S_filter = librosa.decompose.nn_filter(S_full,
                                                 aggregate=np.median,
                                                 metric='cosine',
                                                 width=int(librosa.time_to_frames(2, sr=sr)))
            S_filter = np.minimum(S_full, S_filter)
            margin_v = 2
            power = 2
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 60, 'فصل الأصوات')
            
            mask_v = librosa.util.softmask(S_full - S_filter,
                                         margin_v * S_filter,
                                         power=power)
            S_foreground = mask_v * S_full
            S_background = (1 - mask_v) * S_full
            
            # Convert back to audio
            vocals = librosa.istft(S_foreground * phase)
            background = librosa.istft(S_background * phase)
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 100, 'اكتمل فصل الصوت')
            
            return vocals, background, sr
        except Exception as e:
            logger.error(f"Error in audio separation: {str(e)}")
            raise RuntimeError(f"فشل في فصل الصوت: {str(e)}")

class VoiceCloner:
    def __init__(self):
        self.processor = None
        self.model = None
        
    def lazy_load(self):
        if self.model is None:
            try:
                self.model = VitsModel.from_pretrained("facebook/mms-tts-ara")
                self.processor = AutoProcessor.from_pretrained("facebook/mms-tts-ara")
            except Exception as e:
                print(f"Error loading voice model: {str(e)}")
                # Fallback to basic TTS if model loading fails
                pass
    
    def clone_voice(self, text, reference_audio=None):
        try:
            self.lazy_load()
            if self.model and self.processor:
                inputs = self.processor(text=text, return_tensors="pt")
                with torch.no_grad():
                    output = self.model(**inputs).waveform
                return output.numpy().squeeze(), self.model.config.sampling_rate
            else:
                raise Exception("Model not available")
        except Exception as e:
            print(f"Voice cloning failed: {str(e)}")
            # Fallback to basic TTS
            return None, None

def check_system_requirements():
    """Check if all required system components are available"""
    try:
        # Check FFmpeg
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("FFmpeg is not installed or not working properly")
        
        # Check CUDA availability for PyTorch
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        # Check disk space
        temp_dir = tempfile.gettempdir()
        total, used, free = shutil.disk_usage(temp_dir)
        free_gb = free // (2**30)
        if free_gb < 5:  # Less than 5GB free
            raise RuntimeError(f"Insufficient disk space. Only {free_gb}GB available")
        
        logger.info("System requirements check passed")
        return True
    except Exception as e:
        logger.error(f"System requirements check failed: {str(e)}")
        raise RuntimeError(f"System requirements check failed: {str(e)}")

class VideoDubber:
    def __init__(self):
        self.model_size = "base"
        self.temp_dir = None
        self.audio_processor = None
        self.voice_cloner = None
        self.translator = None
        self.initialized = False
        
    def initialize(self):
        """Initialize all components and models"""
        if self.initialized:
            return
        
        try:
            logger.info("Initializing VideoDubber components...")
            
            # Check system requirements
            check_system_requirements()
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {self.temp_dir}")
            
            # Initialize components
            self.audio_processor = AudioProcessor()
            self.voice_cloner = VoiceCloner()
            
            # Initialize translator
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
            
            self.initialized = True
            logger.info("VideoDubber initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize VideoDubber: {str(e)}")
            self.cleanup()
            raise RuntimeError(f"فشل في تهيئة النظام: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def initialize_session(self):
        """Create a new temporary directory for this session"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up old temp directory: {str(e)}")
        
        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir

    def cleanup_session(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp directory: {str(e)}")

    def extract_audio(self, video_path, audio_path, progress_tracker=None):
        try:
            if progress_tracker:
                progress_tracker.update_progress('extraction', 0, 'بدء استخراج الصوت')
            
            video = mp.VideoFileClip(video_path)
            if video.audio is None:
                raise ValueError("الفيديو لا يحتوي على مسار صوتي")
            
            if progress_tracker:
                progress_tracker.update_progress('extraction', 50, 'جارٍ استخراج الصوت')
            
            video.audio.write_audiofile(audio_path)
            duration = video.duration
            video.close()
            
            if progress_tracker:
                progress_tracker.update_progress('extraction', 100, 'تم استخراج الصوت بنجاح')
            
            return duration
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise RuntimeError(f"فشل في استخراج الصوت من الفيديو: {str(e)}")

    def transcribe_audio(self, audio_path, progress_tracker=None):
        try:
            if progress_tracker:
                progress_tracker.update_progress('transcription', 0, 'بدء تحويل الصوت إلى نص')
            
            model = whisper.load_model(self.model_size)
            result = model.transcribe(audio_path)
            
            if progress_tracker:
                progress_tracker.update_progress('transcription', 50, 'جارٍ تحويل الصوت إلى نص')
            
            if not result.get("text"):
                raise ValueError("لم يتم العثور على نص في الصوت")
            
            if progress_tracker:
                progress_tracker.update_progress('transcription', 100, 'تم تحويل الصوت إلى نص بنجاح')
            
            return result["text"]
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            raise RuntimeError(f"فشل في تحويل الصوت إلى نص: {str(e)}")

    def translate_to_arabic(self, text, progress_tracker=None):
        try:
            if progress_tracker:
                progress_tracker.update_progress('translation', 0, 'بدء الترجمة إلى العربية')
            
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
            translated = translator(text)
            
            if progress_tracker:
                progress_tracker.update_progress('translation', 50, 'جارٍ الترجمة إلى العربية')
            
            if not translated or not translated[0].get("translation_text"):
                raise ValueError("فشل في ترجمة النص")
            
            if progress_tracker:
                progress_tracker.update_progress('translation', 100, 'تمت الترجمة إلى العربية بنجاح')
            
            return translated[0]["translation_text"]
        except Exception as e:
            logger.error(f"Error in translation: {str(e)}")
            raise RuntimeError(f"فشل في الترجمة: {str(e)}")

    def generate_arabic_audio(self, text, output_audio_path, original_duration, progress_tracker=None):
        try:
            if progress_tracker:
                progress_tracker.update_progress('audio_generation', 0, 'بدء إنشاء الصوت العربي')
            
            # Try voice cloning first
            audio_data, sample_rate = self.voice_cloner.clone_voice(text)
            
            if audio_data is not None:
                sf.write(output_audio_path, audio_data, sample_rate)
            else:
                # Fallback to traditional TTS
                tts = gTTS(text=text, lang='ar')
                tts.save(output_audio_path)
            
            # Load and adjust audio duration
            audio = mp.AudioFileClip(output_audio_path)
            if audio.duration < original_duration:
                audio = mp.concatenate_audioclips([audio, mp.AudioClip(lambda t: 0, duration=original_duration - audio.duration)])
            elif audio.duration > original_duration:
                audio = audio.subclip(0, original_duration)
            
            audio.write_audiofile(output_audio_path)
            audio.close()
            
            if progress_tracker:
                progress_tracker.update_progress('audio_generation', 100, 'تم إنشاء الصوت العربي بنجاح')
        except Exception as e:
            logger.error(f"Error in generating Arabic audio: {str(e)}")
            raise RuntimeError(f"فشل في إنشاء الصوت العربي: {str(e)}")

    def merge_audio_video(self, video_path, dubbed_audio_path, original_audio_path, output_path, progress_tracker=None, background_mix_ratio=0.3):
        try:
            if progress_tracker:
                progress_tracker.update_progress('merging', 0, 'بدء دمج الصوت مع الفيديو')
            
            # Separate background audio
            vocals, background, sr = self.audio_processor.separate_audio(original_audio_path, progress_tracker)
            
            # Save background audio temporarily
            background_path = os.path.join(self.temp_dir, 'background.wav')
            sf.write(background_path, background, sr)
            
            # Load all audio components
            dubbed_audio = mp.AudioFileClip(dubbed_audio_path)
            background_audio = mp.AudioFileClip(background_path)
            
            # Mix dubbed audio with background
            mixed_audio = mp.CompositeAudioClip([
                dubbed_audio.volumex(1.0),
                background_audio.volumex(background_mix_ratio)
            ])
            
            # Merge with video
            video = mp.VideoFileClip(video_path)
            final_video = video.set_audio(mixed_audio)
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            # Cleanup
            video.close()
            dubbed_audio.close()
            background_audio.close()
            final_video.close()
            os.remove(background_path)
            
            if progress_tracker:
                progress_tracker.update_progress('merging', 100, 'اكتمل دمج الصوت مع الفيديو')
        except Exception as e:
            logger.error(f"Error in merging audio and video: {str(e)}")
            raise RuntimeError(f"فشل في دمج الصوت مع الفيديو: {str(e)}")

# Initialize the dubber
dubber = VideoDubber()

@app.before_first_request
def initialize_application():
    """Initialize the application before the first request"""
    try:
        dubber.initialize()
    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        raise

@app.route('/')
def home():
    try:
        with open('AI Dubbing App with Simple UI.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
        return render_template_string(html_content)
    except Exception as e:
        logger.error(f"Error loading HTML template: {str(e)}")
        return jsonify({'error': 'فشل في تحميل واجهة المستخدم'}), 500

@app.route('/progress/<session_id>')
def get_progress(session_id):
    def generate():
        if session_id not in progress_queues:
            yield "data: {\"error\": \"Session not found\"}\n\n"
            return
        
        queue = progress_queues[session_id]
        while True:
            try:
                progress_data = queue.get(timeout=30)  # 30 second timeout
                yield f"data: {jsonify(progress_data).get_data(as_text=True)}\n\n"
            except queue.Empty:
                yield "data: {\"status\": \"complete\"}\n\n"
                break
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                break
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/dub', methods=['POST'])
def dub_video():
    if 'video' not in request.files:
        return jsonify({'error': 'لم يتم تقديم ملف فيديو'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400

    # Generate session ID
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + secure_filename(video_file.filename)
    progress_tracker = ProgressTracker(session_id)

    try:
        # Initialize new session
        dubber.initialize_session()
        
        # Check file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        file_ext = os.path.splitext(video_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise ValueError('نوع الملف غير مدعوم. الأنواع المدعومة هي: mp4, avi, mov, mkv')

        # Save uploaded video
        video_path = os.path.join(dubber.temp_dir, secure_filename(video_file.filename))
        video_file.save(video_path)

        # Process paths
        temp_audio_path = os.path.join(dubber.temp_dir, 'temp_audio.wav')
        dubbed_audio_path = os.path.join(dubber.temp_dir, 'dubbed_audio.mp3')
        output_video_path = os.path.join(dubber.temp_dir, 'output_dubbed_video.mp4')

        # Extract and process audio with progress tracking
        progress_tracker.update_progress('overall', 0, 'بدء المعالجة')
        
        original_duration = dubber.extract_audio(video_path, temp_audio_path, progress_tracker)
        progress_tracker.update_progress('overall', 20, 'تم استخراج الصوت')
        
        transcribed_text = dubber.transcribe_audio(temp_audio_path, progress_tracker)
        progress_tracker.update_progress('overall', 40, 'تم التعرف على النص')
        
        arabic_text = dubber.translate_to_arabic(transcribed_text, progress_tracker)
        progress_tracker.update_progress('overall', 60, 'تمت الترجمة')
        
        dubber.generate_arabic_audio(arabic_text, dubbed_audio_path, original_duration, progress_tracker)
        progress_tracker.update_progress('overall', 80, 'تم إنشاء الصوت العربي')
        
        dubber.merge_audio_video(video_path, dubbed_audio_path, temp_audio_path, output_video_path, progress_tracker)
        progress_tracker.update_progress('overall', 100, 'اكتمل الدبلاج')

        # Send the output video file
        response = send_file(output_video_path, as_attachment=True, download_name='dubbed_video.mp4')
        
        # Clean up
        progress_tracker.cleanup()
        dubber.cleanup_session()
        
        return response

    except Exception as e:
        logger.error(f"Error in video dubbing: {str(e)}")
        progress_tracker.cleanup()
        dubber.cleanup_session()
        return jsonify({'error': str(e), 'session_id': session_id}), 500

if __name__ == '__main__':
    try:
        # Read the HTML template
        html_path = Path('AI Dubbing App with Simple UI.html')
        if not html_path.exists():
            raise FileNotFoundError("HTML template file not found")
        
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        app.run(debug=True, port=5000)
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        print(f"فشل في تشغيل التطبيق: {str(e)}")