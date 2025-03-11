import time
import logging
import uuid
from fastapi import Request
from app.utils.client_info import extract_client_info
from app.db.repository import save_request_info

# تنظیمات لاگر
logger = logging.getLogger(__name__)


async def client_info_middleware(request: Request, call_next):
    """
    میدلور برای استخراج و ذخیره اطلاعات کاربر از هر درخواست.
    
    این میدلور اطلاعات مرورگر، دستگاه و مشخصات فنی کاربر را از هدرهای درخواست استخراج 
    و در صورت نیاز در دیتابیس ذخیره می‌کند.
    """
    # ایجاد شناسه منحصر به فرد برای درخواست
    request_id = str(uuid.uuid4())
    
    # ثبت زمان شروع
    start_time = time.time()
    
    # استخراج اطلاعات کاربر
    client_info = extract_client_info(request)
    
    # افزودن اطلاعات به استیت درخواست برای دسترسی در ادامه
    request.state.client_info = client_info
    request.state.request_id = request_id
    
    # ثبت درخواست در لاگ
    logger.info(
        f"درخواست جدید: {request_id} - "
        f"مسیر: {request.url.path} - "
        f"دستگاه: {client_info.device_type} - "
        f"مرورگر: {client_info.browser_name}"
    )
    
    # ادامه پردازش درخواست
    response = await call_next(request)
    
    # محاسبه زمان پاسخ‌دهی
    process_time = time.time() - start_time
    
    # ثبت زمان پاسخ‌دهی
    response.headers["X-Process-Time"] = str(process_time)
    
    # ذخیره اطلاعات درخواست در دیتابیس (اگر مسیر API است)
    if request.url.path.startswith("/api/"):
        try:
            await save_request_info(
                path=request.url.path,
                method=request.method,
                client_info=client_info.dict(),
                status_code=response.status_code,
                process_time=process_time,
                request_id=request_id
            )
        except Exception as e:
            logger.error(f"خطا در ذخیره اطلاعات درخواست: {e}")
    
    return response