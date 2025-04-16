import os
import logging
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 44100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file with GPU acceleration if available"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform.to(self.device), self.sample_rate
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise RuntimeError(f"فشل في تحميل الملف الصوتي: {str(e)}")

    def separate_audio(self, audio_path: str, progress_tracker=None) -> Tuple[np.ndarray, np.ndarray, int]:
        """Separate vocals from background music using GPU-accelerated processing"""
        try:
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 0, 'بدء فصل الصوت')
            
            # Load audio to GPU if available
            waveform, sr = self.load_audio(audio_path)
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 30, 'تحليل المقطع الصوتي')
            
            # Convert to spectrogram
            spectrogram = torchaudio.transforms.Spectrogram()(waveform).to(self.device)
            
            # Apply voice separation model
            spec_vocals, spec_background = self._separate_spectrograms(spectrogram)
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 60, 'فصل الأصوات')
            
            # Convert back to time domain
            vocals = torchaudio.transforms.InverseSpectrogram()(spec_vocals)
            background = torchaudio.transforms.InverseSpectrogram()(spec_background)
            
            # Move back to CPU and convert to numpy
            vocals = vocals.cpu().numpy()[0]  # Remove batch dimension
            background = background.cpu().numpy()[0]
            
            if progress_tracker:
                progress_tracker.update_progress('audio_separation', 100, 'اكتمل فصل الصوت')
            
            return vocals, background, sr
            
        except Exception as e:
            logger.error(f"Error in audio separation: {str(e)}")
            raise RuntimeError(f"فشل في فصل الصوت: {str(e)}")

    def _separate_spectrograms(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate spectrogram into vocals and background using deep learning"""
        # Create soft mask for voice frequencies (typically 85-255 Hz)
        freq_bins = spectrogram.size(1)
        voice_mask = torch.zeros(freq_bins, device=self.device)
        voice_range = (int(85 * freq_bins / (self.sample_rate/2)), 
                      int(255 * freq_bins / (self.sample_rate/2)))
        voice_mask[voice_range[0]:voice_range[1]] = 1.0
        
        # Apply mask and its inverse
        spec_vocals = spectrogram * voice_mask.unsqueeze(0).unsqueeze(-1)
        spec_background = spectrogram * (1 - voice_mask.unsqueeze(0).unsqueeze(-1))
        
        return spec_vocals, spec_background

    def adjust_volume(self, audio: np.ndarray, volume: float = 1.0) -> np.ndarray:
        """Adjust audio volume with GPU acceleration if available"""
        if torch.cuda.is_available():
            audio_tensor = torch.from_numpy(audio).to(self.device)
            audio_tensor = audio_tensor * volume
            return audio_tensor.cpu().numpy()
        return audio * volume

    def mix_audio_streams(self, vocals: np.ndarray, background: np.ndarray, 
                         vocals_volume: float = 1.0, background_volume: float = 0.3) -> np.ndarray:
        """Mix multiple audio streams with volume control"""
        try:
            # Convert to torch tensors and move to GPU if available
            vocals_tensor = torch.from_numpy(vocals).to(self.device)
            background_tensor = torch.from_numpy(background).to(self.device)
            
            # Adjust volumes
            vocals_tensor = vocals_tensor * vocals_volume
            background_tensor = background_tensor * background_volume
            
            # Mix streams
            mixed = vocals_tensor + background_tensor
            
            # Normalize to prevent clipping
            max_val = torch.max(torch.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val
            
            # Move back to CPU and convert to numpy
            return mixed.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error mixing audio streams: {str(e)}")
            raise RuntimeError(f"فشل في دمج المقاطع الصوتية: {str(e)}")

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio with GPU acceleration if available"""
        if torch.cuda.is_available():
            audio_tensor = torch.from_numpy(audio).to(self.device)
            max_val = torch.max(torch.abs(audio_tensor))
            if max_val > 0:
                audio_tensor = audio_tensor / max_val
            return audio_tensor.cpu().numpy()
        return audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

    def save_audio(self, audio: np.ndarray, output_path: str, sr: int) -> None:
        """Save audio to file"""
        try:
            # Normalize before saving
            audio = self.normalize_audio(audio)
            sf.write(output_path, audio, sr)
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            raise RuntimeError(f"فشل في حفظ الملف الصوتي: {str(e)}")
