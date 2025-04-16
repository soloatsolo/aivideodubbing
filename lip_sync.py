import os
import torch
import numpy as np
import face_alignment
import imageio
import cv2
import subprocess
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

class LipSyncProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fa = None
        self.wav2lip_model = None
        self.initialized = False
        
    def initialize(self):
        """Initialize face alignment and Wav2Lip models"""
        if not self.initialized:
            try:
                # Initialize face alignment with GPU support if available
                self.fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType._2D,
                    flip_input=False,
                    device=self.device
                )
                
                # Initialize Wav2Lip model
                if not self.wav2lip_model:
                    from wav2lip import Wav2Lip
                    self.wav2lip_model = Wav2Lip()
                    self.wav2lip_model.load_state_dict(torch.load(
                        'models/wav2lip_gan.pth',
                        map_location=self.device
                    ))
                    self.wav2lip_model.to(self.device)
                    self.wav2lip_model.eval()
                
                self.initialized = True
                logger.info("Lip sync processor initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing lip sync processor: {str(e)}")
                raise RuntimeError("فشل في تهيئة نظام مزامنة الشفاه")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process a single frame for lip syncing"""
        try:
            landmarks = self.fa.get_landmarks(frame)
            if landmarks is None or len(landmarks) == 0:
                return frame, None
                
            # Get lip landmarks (indices 48-68)
            lip_landmarks = landmarks[0][48:68]
            
            # Calculate lip bounding box
            x_min = int(np.min(lip_landmarks[:, 0])) - 10
            x_max = int(np.max(lip_landmarks[:, 0])) + 10
            y_min = int(np.min(lip_landmarks[:, 1])) - 10
            y_max = int(np.max(lip_landmarks[:, 1])) + 10
            
            # Extract lip region
            lip_region = frame[y_min:y_max, x_min:x_max]
            return frame, (lip_region, (x_min, y_min, x_max, y_max))
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, None
    
    def apply_wav2lip(self, video_frames: np.ndarray, audio_path: str, 
                     progress_tracker=None) -> np.ndarray:
        """Apply Wav2Lip model to synchronize lips with audio"""
        try:
            # Convert frames to tensor
            frames_tensor = torch.FloatTensor(video_frames).to(self.device)
            
            # Load and preprocess audio
            mel_chunks = self.wav2lip_model.load_audio(audio_path)
            
            # Process in batches
            batch_size = 32
            synced_frames = []
            
            for i in range(0, len(frames_tensor), batch_size):
                batch_frames = frames_tensor[i:i+batch_size]
                batch_mel = mel_chunks[i:i+batch_size]
                
                with torch.no_grad():
                    pred = self.wav2lip_model(batch_mel, batch_frames)
                    synced_frames.extend(pred.cpu().numpy())
                
                if progress_tracker:
                    progress = int((i + batch_size) / len(frames_tensor) * 100)
                    progress_tracker.update_progress('lip_sync', progress, 
                        f'مزامنة الشفاه: {progress}%')
            
            return np.array(synced_frames)
            
        except Exception as e:
            logger.error(f"Error applying Wav2Lip: {str(e)}")
            raise RuntimeError(f"فشل في مزامنة الشفاه: {str(e)}")
    
    def process_video(self, video_path: str, audio_path: str, 
                     progress_tracker=None) -> str:
        """Process video for lip sync using Wav2Lip"""
        try:
            if progress_tracker:
                progress_tracker.update_progress('lip_sync', 0, 'بدء مزامنة الشفاه')
            
            self.initialize()
            
            # Read video
            video = imageio.get_reader(video_path)
            fps = video.get_meta_data()['fps']
            frames = [frame for frame in video]
            video.close()
            
            if progress_tracker:
                progress_tracker.update_progress('lip_sync', 20, 'تحليل إطارات الفيديو')
            
            # Process frames with parallel processing
            processed_frames = []
            with ThreadPoolExecutor() as executor:
                processed_frames = list(executor.map(self.process_frame, frames))
            
            if progress_tracker:
                progress_tracker.update_progress('lip_sync', 40, 'تطبيق مزامنة الشفاه')
            
            # Apply Wav2Lip
            synced_frames = self.apply_wav2lip(
                np.array([f[0] for f in processed_frames if f[1] is not None]),
                audio_path,
                progress_tracker
            )
            
            # Create output path
            output_path = os.path.join(os.path.dirname(video_path), 'lip_synced_video.mp4')
            
            # Write output video
            writer = imageio.get_writer(output_path, fps=fps)
            for frame in synced_frames:
                writer.append_data(frame)
            writer.close()
            
            if progress_tracker:
                progress_tracker.update_progress('lip_sync', 100, 'اكتملت مزامنة الشفاه')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in lip sync processing: {str(e)}")
            raise RuntimeError(f"فشل في مزامنة الشفاه: {str(e)}")
            
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.wav2lip_model:
                del self.wav2lip_model
            if self.fa:
                del self.fa
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")
