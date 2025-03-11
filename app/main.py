from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import asyncio
import time
from typing import Callable

from app.config import settings, create_required_directories
from app.api.face_analysis import router as face_analysis_router
from app.api.health import router as health_router
from app.api.analytics import router as analytics_router
from app.middleware import client_info_middleware
from app.db.connection import connect_to_mongo, close_mongo_connection
from app.services.woocommerce import initialize_product_cache
from app.db.repository import create_database_indexes, check_and_update_request_analytics

# تنظیمات لاگینگ
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# اطمینان از وجود دایرکتوری‌های مورد نیاز
create_required_directories()

# بررسی وجود داشتن فایل داده‌های مرجع
if not os.path.exists(settings.FACE_SHAPE_DATA_PATH):
    logging.warning(f"فایل داده‌های مرجع شکل صورت در مسیر {settings.FACE_SHAPE_DATA_PATH} یافت نشد!")

# ایجاد نمونه برنامه FastAPI
app = FastAPI(
    title="سیستم تشخیص چهره و پیشنهاد فریم عینک",
    description="API برای تحلیل شکل صورت و پیشنهاد فریم عینک مناسب",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# افزودن middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در محیط تولید، دامنه‌های خاص را مشخص کنید
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# افزودن میدلور برای استخراج اطلاعات کاربر
app.middleware("http")(client_info_middleware)


# میدلور برای مدیریت استثناء‌ها
@app.middleware("http")
async def db_exception_handler(request: Request, call_next: Callable):
    try:
        return await call_next(request)
    except RuntimeError as e:
        if str(e).startswith("اتصال به MongoDB برقرار نشده است"):
            logger.error(f"خطا در اتصال به دیتابیس: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "سرویس دیتابیس در دسترس نیست. لطفاً بعداً دوباره امتحان کنید."}
            )
        raise
    except Exception as e:
        logger.error(f"خطای پیش‌بینی نشده: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "خطای داخلی سرور رخ داده است. لطفاً بعداً دوباره امتحان کنید."}
        )


# میدلور برای ثبت زمان پاسخگویی
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# افزودن روترها
app.include_router(face_analysis_router, prefix="/api/v1", tags=["تحلیل چهره"])
app.include_router(health_router, prefix="/api/v1", tags=["سلامت سیستم"])
app.include_router(analytics_router, prefix="/api/v1", tags=["آمار و تحلیل"])


@app.on_event("startup")
async def startup_event():
    """
    رویداد راه‌اندازی برنامه
    """
    logging.info("سیستم تشخیص چهره و پیشنهاد فریم عینک در حال راه‌اندازی...")
    
    # بررسی و ایجاد دایرکتوری داده در صورت نیاز
    os.makedirs(os.path.dirname(settings.FACE_SHAPE_DATA_PATH), exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # اتصال به MongoDB با چند بار تلاش
    max_retries = 5
    retry_delay = 5  # ثانیه
    
    for attempt in range(max_retries):
        try:
            await connect_to_mongo()
            logging.info("اتصال به MongoDB با موفقیت برقرار شد")
            
            # ایجاد ایندکس‌های دیتابیس
            await create_database_indexes()
            
            # بررسی و بروزرسانی داده‌های تحلیلی
            await check_and_update_request_analytics()
            
            # راه‌اندازی کش محصولات WooCommerce (در یک تسک جداگانه تا مانع راه‌اندازی سریع سیستم نشود)
            asyncio.create_task(initialize_product_cache())
            
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"خطا در اتصال به MongoDB (تلاش {attempt+1}/{max_retries}): {str(e)}")
                await asyncio.sleep(retry_delay)
            else:
                logging.error(f"خطا در اتصال به MongoDB پس از {max_retries} تلاش: {str(e)}")
                # در حالت آخر می‌توانیم خطا را بالا بفرستیم یا بدون دیتابیس ادامه دهیم
                logging.warning("برنامه بدون اتصال به دیتابیس ادامه می‌یابد. برخی عملکردها محدود خواهند بود.")


@app.on_event("shutdown")
async def shutdown_event():
    """
    رویداد خاموش شدن برنامه
    """
    logging.info("سیستم تشخیص چهره و پیشنهاد فریم عینک در حال خاموش شدن...")
    
    # بستن اتصال MongoDB
    await close_mongo_connection()


# مسیر ریشه
@app.get("/")
async def root():
    """مسیر ریشه برای بررسی دسترسی‌پذیری API"""
    return {
        "message": "سیستم تشخیص چهره و پیشنهاد فریم عینک",
        "version": "1.0.0",
        "status": "آنلاین",
        "docs": "/docs"
    }


# راه‌اندازی کارگزاری زمان‌بندی‌شده برای بروزرسانی آمارها
async def start_scheduled_tasks():
    """شروع کارهای زمان‌بندی‌شده"""
    
    async def update_analytics_data():
        """بروزرسانی دوره‌ای داده‌های تحلیلی"""
        while True:
            try:
                await check_and_update_request_analytics()
                # اجرای هر 12 ساعت یک‌بار
                await asyncio.sleep(12 * 60 * 60)
            except Exception as e:
                logger.error(f"خطا در بروزرسانی داده‌های تحلیلی: {str(e)}")
                await asyncio.sleep(60 * 60)  # در صورت خطا، یک ساعت صبر می‌کنیم
    
    # شروع تسک در پس‌زمینه
    asyncio.create_task(update_analytics_data())


# راه‌اندازی کارهای زمان‌بندی‌شده در شروع برنامه
@app.on_event("startup")
async def start_scheduler():
    await start_scheduled_tasks()


# راه‌اندازی برنامه با uvicorn در صورت اجرای مستقیم این فایل
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )