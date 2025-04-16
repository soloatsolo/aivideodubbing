#!/usr/bin/env python3
import os
import sys
import subprocess
import pkg_resources

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ FFmpeg is not installed")
        return False

def install_ffmpeg():
    """Install FFmpeg"""
    print("Installing FFmpeg...")
    try:
        if sys.platform.startswith('linux'):
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True)
        elif sys.platform == 'darwin':
            subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
        else:
            print("Please install FFmpeg manually from https://ffmpeg.org/download.html")
            return False
        return True
    except subprocess.CalledProcessError:
        print("Failed to install FFmpeg")
        return False

def check_dependencies():
    """Check if all required Python packages are installed"""
    required_packages = [
        'flask==3.1.0',
        'flask-cors==5.0.1',
        'werkzeug==3.0.1',
        'moviepy==2.1.2',
        'openai-whisper==20240930',
        'transformers==4.51.3',
        'torch==2.6.0',
        'torchaudio==2.6.0',
        'librosa==0.10.1',
        'numpy>=1.25.0',
        'gTTS==2.5.4',
        'soundfile==0.13.1',
        'scipy>=1.15.2',
        'scikit-learn>=1.6.1',
        'spleeter==2.4.0',
        'face_alignment==1.3.5',
        'imageio==2.9.0',
        'pysubs2==1.4.2',
        'gradio==4.19.1',
        'opencv-python==4.9.0.80',
        'opencv-python-headless==4.9.0.80'
    ]
    
    missing = []
    for package in required_packages:
        name = package.split('==')[0].split('>=')[0]
        try:
            pkg_resources.require(package)
            print(f"✓ {name} is installed")
        except pkg_resources.DistributionNotFound:
            missing.append(package)
            print(f"✗ {name} is missing")
        except pkg_resources.VersionConflict:
            missing.append(package)
            print(f"✗ {name} version conflict")
    
    return missing

def setup_virtual_environment():
    """Set up a virtual environment if not already in one"""
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        print("Setting up virtual environment...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            if sys.platform == 'win32':
                activate_script = os.path.join('venv', 'Scripts', 'activate')
            else:
                activate_script = os.path.join('venv', 'bin', 'activate')
            
            if sys.platform == 'win32':
                os.system(f'call {activate_script}')
            else:
                os.system(f'source {activate_script}')
            
            return True
        except subprocess.CalledProcessError:
            print("Failed to create virtual environment")
            return False
    return True

def install_dependencies(missing_packages):
    """Install missing Python packages"""
    if missing_packages:
        print("\nInstalling missing packages...")
        try:
            # First install torch and torchaudio
            torch_packages = [p for p in missing_packages if p.startswith(('torch==', 'torchaudio=='))]
            other_packages = [p for p in missing_packages if not p.startswith(('torch==', 'torchaudio=='))]
            
            if torch_packages:
                print("Installing PyTorch packages...")
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + torch_packages, check=True)
            
            if other_packages:
                print("Installing other packages...")
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + other_packages, check=True)
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install some packages: {str(e)}")
            return False
    return True

def main():
    """Main function to check and set up the environment"""
    print("Checking system requirements...")
    
    # Check and setup virtual environment
    if not setup_virtual_environment():
        sys.exit(1)
    
    # Check FFmpeg
    if not check_ffmpeg():
        if not install_ffmpeg():
            sys.exit(1)
    
    # Check Python dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        if not install_dependencies(missing_packages):
            sys.exit(1)
    
    print("\nAll requirements satisfied! Starting the application...")
    
    # Create necessary directories
    for dir_name in ['temp', 'logs', 'model_cache']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Start the application
    try:
        subprocess.run([sys.executable, 'video_dubbing_gui.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()