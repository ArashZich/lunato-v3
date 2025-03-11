from celery import Celery
from app.config import settings

# تنظیمات Celery
app = Celery(
    "eyeglass_recommendation",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# تنظیم صف‌ها و مسیریابی وظایف
app.conf.task_routes = {
    "app.services.tasks.detect_face": {"queue": "face_detection"},
    "app.services.tasks.analyze_face_shape": {"queue": "face_analysis"},
    "app.services.tasks.match_frames": {"queue": "frame_matching"},
}

# تنظیمات اضافی
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_time_limit=180,  # 3 دقیقه حداکثر زمان اجرای هر وظیفه
    timezone="UTC",
    enable_utc=True,
)

# ایمپورت تسک‌ها برای دسترسی
app.autodiscover_tasks(["app.services.tasks"])