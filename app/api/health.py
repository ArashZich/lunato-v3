from fastapi import APIRouter, Depends
from datetime import datetime
import logging

from app.models.responses import HealthResponse
from app.config import settings, get_settings
from app.db.connection import get_database

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# تعریف روتر
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(config: settings = Depends(get_settings)):
    """
    بررسی سلامت API.
    
    این API برای بررسی دسترسی‌پذیری و درستی کارکرد سیستم استفاده می‌شود.
    همچنین اطلاعات مربوط به نسخه و محیط اجرا را برمی‌گرداند.
    """
    try:
        # بررسی اتصال به دیتابیس
        db = get_database()
        await db.command("ping")
        db_status = "متصل"
    except Exception as e:
        logger.error(f"خطا در اتصال به دیتابیس: {str(e)}")
        db_status = "خطای اتصال"
    
    # بررسی اتصال به Celery
    try:
        from app.celery_app import app as celery_app
        celery_status = "سالم" if celery_app.control.ping() else "خطای اتصال"
    except Exception as e:
        logger.error(f"خطا در اتصال به Celery: {str(e)}")
        celery_status = "خطای اتصال"
    
    return HealthResponse(
        success=True,
        message="سیستم تشخیص چهره و پیشنهاد فریم عینک به درستی کار می‌کند",
        version="1.0.0",
        environment=config.ENVIRONMENT,
        timestamp=datetime.utcnow(),
        database_status=db_status,
        celery_status=celery_status
    )