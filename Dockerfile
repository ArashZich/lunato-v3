# استفاده از تصویر پایه پایتون با نسخه 3.10
FROM python:3.10-slim

# تنظیم متغیرهای محیطی
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# تنظیم دایرکتوری کاری
WORKDIR /app

# نصب پکیج‌های مورد نیاز برای OpenCV و Mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-xcb1 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# کپی کردن فایل‌های نیازمندی و نصب آن‌ها
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# نصب کتابخانه mediapipe به صورت جداگانه
RUN pip install --no-cache-dir mediapipe==0.10.18

# کپی کردن کل پروژه به داخل کانتینر
COPY . /app/

# ایجاد دایرکتوری داده اگر وجود نداشته باشد
RUN mkdir -p /app/data

# اجرای برنامه با uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]