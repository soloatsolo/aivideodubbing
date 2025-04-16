# AI Video Dubbing Application | تطبيق الدبلجة بالذكاء الاصطناعي

[English](#english) | [العربية](#arabic)

## English

### Overview
This application uses artificial intelligence to automatically dub videos from English to Arabic while preserving the original background music and sound effects. It provides a user-friendly web interface for uploading videos and monitoring the dubbing process.

### Features
- Automatic speech recognition and transcription
- Neural machine translation from English to Arabic
- Voice cloning and synthesis
- Background music preservation
- Real-time progress tracking
- Support for multiple video formats (MP4, AVI, MOV, MKV)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/soloatsolo/aivideodubbing.git
cd aivideodubbing
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg (required for video processing):
- On Ubuntu/Debian: `sudo apt-get install ffmpeg`
- On MacOS: `brew install ffmpeg`
- On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

### Usage
1. Start the application:
```bash
python video_dubbing_gui.py
```

2. Open your web browser and navigate to `http://localhost:5000`
3. Upload a video file
4. Monitor the dubbing progress
5. Download the dubbed video when complete

### Technical Requirements
- Python 3.8 or higher
- FFmpeg
- Modern web browser
- Sufficient disk space for video processing

### Need Help?
If you encounter any issues or need assistance, please:
1. Check the error messages in the application
2. Verify that all dependencies are installed correctly
3. Create an issue on GitHub with details about your problem

---

## Arabic {#arabic}

### نظرة عامة
يستخدم هذا التطبيق الذكاء الاصطناعي لدبلجة مقاطع الفيديو تلقائياً من اللغة الإنجليزية إلى العربية مع الحفاظ على الموسيقى الخلفية والمؤثرات الصوتية الأصلية. يوفر واجهة ويب سهلة الاستخدام لرفع مقاطع الفيديو ومراقبة عملية الدبلجة.

### المميزات
- التعرف التلقائي على الكلام ونسخه
- الترجمة الآلية العصبية من الإنجليزية إلى العربية
- استنساخ الصوت وتوليده
- الحفاظ على الموسيقى الخلفية
- تتبع التقدم في الوقت الفعلي
- دعم صيغ متعددة للفيديو (MP4, AVI, MOV, MKV)

### التثبيت
1. استنساخ المستودع:
```bash
git clone https://github.com/soloatsolo/aivideodubbing.git
cd aivideodubbing
```

2. إنشاء وتفعيل البيئة الافتراضية:
```bash
python -m venv venv
source venv/bin/activate  # لينكس/ماك
# أو
.\venv\Scripts\activate  # ويندوز
```

3. تثبيت الحزم المطلوبة:
```bash
pip install -r requirements.txt
```

4. تثبيت FFmpeg (مطلوب لمعالجة الفيديو):
- على أوبونتو/ديبيان: `sudo apt-get install ffmpeg`
- على ماك: `brew install ffmpeg`
- على ويندوز: تحميل من [موقع FFmpeg](https://ffmpeg.org/download.html)

### الاستخدام
1. تشغيل التطبيق:
```bash
python video_dubbing_gui.py
```

2. فتح المتصفح والذهاب إلى `http://localhost:5000`
3. رفع ملف الفيديو
4. مراقبة تقدم عملية الدبلجة
5. تحميل الفيديو المدبلج عند الانتهاء

### المتطلبات التقنية
- بايثون 3.8 أو أحدث
- FFmpeg
- متصفح ويب حديث
- مساحة كافية على القرص لمعالجة الفيديو

### تحتاج مساعدة؟
إذا واجهت أي مشاكل أو تحتاج إلى مساعدة:
1. تحقق من رسائل الخطأ في التطبيق
2. تأكد من تثبيت جميع المتطلبات بشكل صحيح
3. قم بإنشاء issue على GitHub مع تفاصيل المشكلة التي تواجهها