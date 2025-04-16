import os
import logging
import pysubs2
import numpy as np
from datetime import timedelta
import subprocess

logger = logging.getLogger("subtitle_handler")

class SubtitleHandler:
    def __init__(self):
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Verify FFmpeg is available with subtitle support"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError("FFmpeg غير مثبت أو لا يعمل بشكل صحيح")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg غير مثبت. يرجى تثبيت FFmpeg لمعالجة الترجمات")

    def create_subtitle_file(self, text, duration, output_path, progress_tracker=None):
        """Create SRT subtitle file from text"""
        try:
            if progress_tracker:
                progress_tracker.update_progress('subtitles', 0, 'بدء إنشاء ملف الترجمات')
            
            # Create a new subtitles file
            subs = pysubs2.SSAFile()
            
            # Split text into sentences (simple implementation)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Calculate approximate duration per sentence
            total_chars = sum(len(s) for s in sentences)
            chars_per_second = total_chars / duration
            
            current_time = 0
            for i, sentence in enumerate(sentences):
                # Calculate duration based on sentence length and reading speed
                sentence_duration = (len(sentence) / chars_per_second) * 1000  # Convert to milliseconds
                sentence_duration = min(max(sentence_duration, 2000), 5000)  # Between 2-5 seconds
                
                # Create subtitle event
                event = pysubs2.SSAEvent(
                    start=int(current_time),
                    end=int(current_time + sentence_duration),
                    text=sentence
                )
                subs.events.append(event)
                
                current_time += sentence_duration
                
                if progress_tracker:
                    progress = int((i + 1) / len(sentences) * 50)
                    progress_tracker.update_progress('subtitles', progress, 
                        f'معالجة الجملة {i + 1} من {len(sentences)}')
            
            # Save the subtitle file
            subs.save(output_path)
            
            if progress_tracker:
                progress_tracker.update_progress('subtitles', 100, 'تم إنشاء ملف الترجمات')
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating subtitles: {str(e)}")
            raise RuntimeError(f"فشل في إنشاء ملف الترجمات: {str(e)}")

    def burn_subtitles(self, video_path, subtitle_path, output_path, progress_tracker=None):
        """Burn subtitles into video using FFmpeg"""
        try:
            if progress_tracker:
                progress_tracker.update_progress('subtitles', 50, 'بدء دمج الترجمات مع الفيديو')
            
            # FFmpeg command to burn subtitles
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f"subtitles={subtitle_path}:force_style='Fontname=Arial,FontSize=24,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,Outline=2'",
                '-c:a', 'copy',
                output_path
            ]
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            duration_seconds = None
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                    
                if "Duration" in line and duration_seconds is None:
                    time_str = line.split("Duration: ")[1].split(",")[0]
                    h, m, s = time_str.split(":")
                    duration_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                
                if "time=" in line and duration_seconds:
                    time_str = line.split("time=")[1].split(" ")[0]
                    h, m, s = time_str.split(":")
                    current_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                    progress = int((current_seconds / duration_seconds) * 50) + 50
                    
                    if progress_tracker:
                        progress_tracker.update_progress('subtitles', progress, 
                            'جارٍ دمج الترجمات مع الفيديو')
            
            # Check if process completed successfully
            if process.wait() != 0:
                raise RuntimeError("فشل في دمج الترجمات مع الفيديو")
            
            if progress_tracker:
                progress_tracker.update_progress('subtitles', 100, 'تم دمج الترجمات بنجاح')
            
            return True
            
        except Exception as e:
            logger.error(f"Error burning subtitles: {str(e)}")
            raise RuntimeError(f"فشل في دمج الترجمات مع الفيديو: {str(e)}")

    def extract_subtitles(self, video_path, output_path=None):
        """Extract subtitles from video if they exist"""
        try:
            if output_path is None:
                output_path = os.path.splitext(video_path)[0] + '.srt'
            
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-map', '0:s:0',
                output_path
            ]
            
            result = subprocess.run(command, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                return output_path
            return None
            
        except Exception as e:
            logger.error(f"Error extracting subtitles: {str(e)}")
            return None
