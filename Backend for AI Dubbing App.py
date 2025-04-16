from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import moviepy.editor as mp
from transformers import pipeline
from gtts import gTTS
from spleeter.separator import Separator
import os
import shutil

app = FastAPI()

# إعداد CORS للسماح بالطلبات من الواجهة الأمامية
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إنشاء مجلد مؤقت للملفات
if not os.path.exists("temp"):
    os.makedirs("temp")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"fileName": file.filename}

@app.post("/process")
async def process_audio(data: dict):
    file_name = data.get("fileName")
    target_language = data.get("targetLanguage", "ar")
    
    # استخراج الصوت من الفيديو
    video_path = f"temp/{file_name}"
    video = mp.VideoFileClip(video_path)
    audio_path = "temp/temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    
    # فصل الحوار عن الموسيقى باستخدام Spleeter
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path, "temp")
    vocal_path = "temp/temp_audio/vocals.wav"
    
    # تفريغ النص باستخدام Whisper
    model = whisper.load_model("base")
    result = model.transcribe(vocal_path)
    text = result["text"]
    
    # ترجمة النص إلى العربية
    translator = pipeline("translation", model="facebook/m2m100_418M")
    translated = translator(text, src_lang="en", tgt_lang="ar")[0]["translation_text"]
    
    # توليد الصوت العربي باستخدام gTTS (حل مؤقت)
    dubbed_audio_path = "temp/dubbed_audio.mp3"
    tts = gTTS(translated, lang="ar")
    tts.save(dubbed_audio_path)
    
    return {"audioUrl": "dubbed_audio.mp3"}

@app.post("/sync")
async def sync_audio(data: dict):
    audio_url = data.get("audioUrl")
    video_file = data.get("videoFile")
    
    # تحميل الفيديو الأصلي
    video_path = f"temp/{video_file}"
    video = mp.VideoFileClip(video_path)
    
    # تحميل الصوت المدبلج
    audio_path = f"temp/{audio_url}"
    audio = mp.AudioFileClip(audio_path)
    
    # دمج الصوت مع الفيديو
    final_video = video.set_audio(audio)
    output_path = "temp/output_video.mp4"
    final_video.write_videofile(output_path)
    
    return {"videoUrl": "output_video.mp4"}

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = f"temp/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=file_name)
    return {"error": "الملف غير موجود"}