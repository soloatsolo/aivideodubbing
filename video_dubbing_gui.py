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
from werkzeug.utils import secure_filename
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import moviepy.editor as mp
from datetime import datetime
import queue
import threading
from config import get_config, CONFIG
from model_manager import ModelManager

# Initialize logging and application
logger = logging.getLogger("video_dubber")
logger.setLevel(logging.DEBUG)

# File handler with rotation
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(log_dir, "video_dubber.log"),
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatters
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(levelname)s: %(message)s')

file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = CONFIG['MAX_CONTENT_LENGTH']

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
        """Separate vocals from background music using librosa"""
        try:
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 0, 'بدء فصل الصوت')
            
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 30, 'تحليل المقطع الصوتي')
            
            # Compute the spectrogram
            S_full, phase = librosa.magphase(librosa.stft(y))
            
            # Do soft-masking
            S_filter = librosa.decompose.nn_filter(
                S_full,
                aggregate=np.median,
                metric='cosine',
                width=int(librosa.time_to_frames(2, sr=sr))
            )
            
            S_filter = np.minimum(S_full, S_filter)
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 60, 'فصل الأصوات')
            
            # Compute and apply the mask
            margin = 2
            power = 2
            mask = librosa.util.softmask(
                S_full - S_filter,
                margin * S_filter,
                power=power
            )
            
            # Get the foreground (vocals)
            S_foreground = mask * S_full
            # Get the background (music)
            S_background = (1 - mask) * S_full
            
            # Convert back to time domain
            vocals = librosa.istft(S_foreground * phase)
            background = librosa.istft(S_background * phase)
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 100, 'اكتمل فصل الصوت')
            
            return vocals, background, sr
            
        except Exception as e:
            logger.error(f"Error in audio separation: {str(e)}")
            raise RuntimeError(f"فشل في فصل الصوت: {str(e)}")

    def adjust_audio_durations(self, dubbed_audio, background_audio, target_duration, sr):
        """Adjust audio durations to match target length"""
        try:
            # Convert target duration from seconds to samples
            target_samples = int(target_duration * sr)
            
            # Adjust dubbed audio
            if len(dubbed_audio) < target_samples:
                # Pad with silence if too short
                dubbed_audio = np.pad(dubbed_audio, (0, target_samples - len(dubbed_audio)))
            elif len(dubbed_audio) > target_samples:
                # Trim if too long
                dubbed_audio = dubbed_audio[:target_samples]
            
            # Adjust background audio similarly
            if len(background_audio) < target_samples:
                # Loop the background if too short
                repeats = int(np.ceil(target_samples / len(background_audio)))
                background_audio = np.tile(background_audio, repeats)[:target_samples]
            elif len(background_audio) > target_samples:
                background_audio = background_audio[:target_samples]
            
            return dubbed_audio, background_audio
            
        except Exception as e:
            logger.error(f"Error adjusting audio durations: {str(e)}")
            raise RuntimeError(f"فشل في تعديل مدة المقاطع الصوتية: {str(e)}")

    def mix_audio(self, dubbed_audio, background_audio, sr, background_volume=0.3):
        """Mix dubbed audio with background at specified volume"""
        try:
            # Normalize both audio streams
            dubbed_audio = dubbed_audio / np.max(np.abs(dubbed_audio))
            background_audio = background_audio / np.max(np.abs(background_audio))
            
            # Apply volume adjustment to background
            background_audio = background_audio * background_volume
            
            # Mix the audio streams
            mixed_audio = dubbed_audio + background_audio
            
            # Prevent clipping
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
            
            return mixed_audio
            
        except Exception as e:
            logger.error(f"Error mixing audio: {str(e)}")
            raise RuntimeError(f"فشل في دمج المقاطع الصوتية: {str(e)}")

    def save_audio(self, audio_data, output_path, sr):
        """Save audio data to file"""
        try:
            sf.write(output_path, audio_data, sr)
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            raise RuntimeError(f"فشل في حفظ الملف الصوتي: {str(e)}")

class VideoDubber:
    def __init__(self):
        self.temp_dir = None
        self.model_manager = ModelManager()
        self.audio_processor = AudioProcessor()
        self.lip_sync_processor = None  # Lazy initialization
        self.subtitle_handler = None    # Lazy initialization
        self.initialized = False
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        try:
            result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError("FFmpeg غير مثبت أو لا يعمل بشكل صحيح")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg غير مثبت. يرجى تثبيت FFmpeg لمعالجة الفيديو.")

    def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return True
        
        try:
            logger.info("Initializing video dubber...")
            
            # Initialize lip sync processor if needed
            if self.lip_sync_processor is None:
                from lip_sync import LipSyncProcessor
                self.lip_sync_processor = LipSyncProcessor()
            
            # Initialize subtitle handler if needed
            if self.subtitle_handler is None:
                from subtitle_handler import SubtitleHandler
                self.subtitle_handler = SubtitleHandler()
            
            # Verify models are available
            models_ok, missing_models = self.model_manager.verify_models()
            if not models_ok:
                logger.error(f"Missing required models: {missing_models}")
                raise RuntimeError(f"النماذج المطلوبة غير متوفرة: {', '.join(missing_models)}")
            
            self.initialized = True
            logger.info("Video dubber initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"فشل في تهيئة النظام: {str(e)}")
            
    def process_video(self, video_path, progress_tracker=None):
        """Main video processing pipeline with lip sync and subtitles"""
        try:
            # Extract and process audio
            temp_audio_path = os.path.join(self.temp_dir, 'temp_audio.wav')
            dubbed_audio_path = os.path.join(self.temp_dir, 'dubbed_audio.mp3')
            subtitle_path = os.path.join(self.temp_dir, 'subtitles.srt')
            lip_sync_path = os.path.join(self.temp_dir, 'lip_sync.mp4')
            final_output_path = os.path.join(self.temp_dir, 'output_dubbed_video.mp4')

            # Extract audio and get duration
            original_duration = self.extract_audio(video_path, temp_audio_path, progress_tracker)
            progress_tracker.update_progress('overall', 20, 'تم استخراج الصوت')

            # Transcribe and translate
            transcribed_text = self.transcribe_audio(temp_audio_path, progress_tracker)
            progress_tracker.update_progress('overall', 35, 'تم التعرف على النص')

            arabic_text = self.translate_to_arabic(transcribed_text, progress_tracker)
            progress_tracker.update_progress('overall', 45, 'تمت الترجمة')

            # Generate Arabic audio
            self.generate_arabic_audio(arabic_text, dubbed_audio_path, original_duration, progress_tracker)
            progress_tracker.update_progress('overall', 60, 'تم إنشاء الصوت العربي')

            # Process lip sync
            lip_sync_output = self.lip_sync_processor.process_video(video_path, dubbed_audio_path, progress_tracker)
            progress_tracker.update_progress('overall', 75, 'تمت مزامنة الشفاه')

            # Generate and burn subtitles
            self.subtitle_handler.create_subtitle_file(arabic_text, original_duration, subtitle_path, progress_tracker)
            progress_tracker.update_progress('overall', 85, 'تم إنشاء الترجمات')

            # Merge audio and video with subtitles
            self.subtitle_handler.burn_subtitles(
                lip_sync_output,
                subtitle_path, 
                final_output_path,
                progress_tracker
            )
            progress_tracker.update_progress('overall', 100, 'اكتمل الدبلاج')

            return final_output_path

        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            raise RuntimeError(f"فشل في معالجة الفيديو: {str(e)}")

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

    # Get options from request
    enable_lip_sync = request.form.get('enableLipSync', 'false').lower() == 'true'
    enable_subtitles = request.form.get('enableSubtitles', 'false').lower() == 'true'
    background_volume = float(request.form.get('backgroundVolume', '30')) / 100.0  # Convert percentage to decimal

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
        subtitle_path = os.path.join(dubber.temp_dir, 'subtitles.srt')
        lip_sync_path = os.path.join(dubber.temp_dir, 'lip_sync.mp4')
        output_video_path = os.path.join(dubber.temp_dir, 'output_dubbed_video.mp4')

        # Extract and process audio with progress tracking
        progress_tracker.update_progress('overall', 0, 'بدء المعالجة')
        
        original_duration = dubber.extract_audio(video_path, temp_audio_path, progress_tracker)
        progress_tracker.update_progress('overall', 15, 'تم استخراج الصوت')
        
        transcribed_text = dubber.transcribe_audio(temp_audio_path, progress_tracker)
        progress_tracker.update_progress('overall', 30, 'تم التعرف على النص')
        
        arabic_text = dubber.translate_to_arabic(transcribed_text, progress_tracker)
        progress_tracker.update_progress('overall', 45, 'تمت الترجمة')
        
        dubber.generate_arabic_audio(arabic_text, dubbed_audio_path, original_duration, progress_tracker)
        progress_tracker.update_progress('overall', 60, 'تم إنشاء الصوت العربي')

        # Process with lip sync if enabled
        current_video = video_path
        if enable_lip_sync:
            progress_tracker.update_progress('overall', 70, 'جارٍ مزامنة الشفاه')
            lip_sync_output = dubber.lip_sync_processor.process_video(
                current_video, 
                dubbed_audio_path, 
                progress_tracker
            )
            current_video = lip_sync_output
            progress_tracker.update_progress('overall', 80, 'تمت مزامنة الشفاه')

        # Generate subtitles if enabled
        if enable_subtitles:
            progress_tracker.update_progress('overall', 85, 'جارٍ إنشاء الترجمات')
            dubber.subtitle_handler.create_subtitle_file(
                arabic_text, 
                original_duration, 
                subtitle_path, 
                progress_tracker
            )
            dubber.subtitle_handler.burn_subtitles(
                current_video,
                subtitle_path, 
                output_video_path,
                progress_tracker
            )
            current_video = output_video_path
            progress_tracker.update_progress('overall', 90, 'تم إضافة الترجمات')

        # Final audio-video merge
        progress_tracker.update_progress('overall', 95, 'جارٍ دمج الصوت النهائي')
        dubber.merge_audio_video(
            current_video, 
            dubbed_audio_path, 
            temp_audio_path, 
            output_video_path, 
            progress_tracker,
            background_mix_ratio=background_volume
        )
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