from flask import Flask, request, send_file
from flask_cors import CORS
import whisper
import moviepy.editor as mp
from transformers import pipeline
from gtts import gTTS
from spleeter.separator import Separator
import os
import shutil

app = Flask(__name__)
CORS(app)

if not os.path.exists("temp"):
    os.makedirs("temp")

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    file_path = f"temp/{file.filename}"
    file.save(file_path)
    return {"fileName": file.filename}

@app.route('/process', methods=['POST'])
def process_audio():
    data = request.json
    file_name = data.get("fileName")
    video_path = f"temp/{file_name}"
    video = mp.VideoFileClip(video_path)
    audio_path = "temp/temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path, "temp")
    vocal_path = "temp/temp_audio/vocals.wav"
    
    model = whisper.load_model("base")
    result = model.transcribe(vocal_path)
    text = result["text"]
    
    translator = pipeline("translation", model="facebook/m2m100_418M")
    translated = translator(text, src_lang="en", tgt_lang="ar")[0]["translation_text"]
    
    dubbed_audio_path = "temp/dubbed_audio.mp3"
    tts = gTTS(translated, lang="ar")
    tts.save(dubbed_audio_path)
    
    return {"audioUrl": "dubbed_audio.mp3"}

@app.route('/sync', methods=['POST'])
def sync_audio():
    data = request.json
    audio_url = data.get("audioUrl")
    video_file = data.get("videoFile")
    video_path = f"temp/{video_file}"
    video = mp.VideoFileClip(video_path)
    audio_path = f"temp/{audio_url}"
    audio = mp.AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    output_path = "temp/output_video.mp4"
    final_video.write_videofile(output_path)
    return {"videoUrl": "output_video.mp4"}

@app.route('/download/<file_name>', methods=['GET'])
def download_file(file_name):
    file_path = f"temp/{file_name}"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return {"error": "الملف غير موجود"}, 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
