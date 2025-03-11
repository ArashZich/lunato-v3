import cv2
import numpy as np
import logging
import base64
import os
from typing import Optional, Tuple
from fastapi import UploadFile
import aiofiles
import binascii

# تنظیمات لاگر
logger = logging.getLogger(__name__)


async def validate_image_file(file: UploadFile) -> bool:
    """
    بررسی اعتبار فایل تصویر.
    
    Args:
        file: فایل آپلود شده
        
    Returns:
        bool: نتیجه اعتبارسنجی
    """
    try:
        # بررسی نوع فایل
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        if file.content_type not in allowed_types:
            logger.warning(f"نوع فایل نامعتبر: {file.content_type}")
            return False
        
        # خواندن بخشی از فایل برای بررسی
        chunk = await file.read(1024)  # خواندن 1KB اول
        await file.seek(0)  # بازگشت به ابتدای فایل
        
        # بررسی سیگنچر فایل JPG/PNG
        jpeg_signature = bytes([0xFF, 0xD8, 0xFF])
        png_signature = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        
        if (chunk.startswith(jpeg_signature) or chunk.startswith(png_signature)):
            return True
        else:
            logger.warning("سیگنچر فایل تصویر نامعتبر است")
            return False
        
    except Exception as e:
        logger.error(f"خطا در اعتبارسنجی فایل تصویر: {str(e)}")
        return False


async def read_image_file(file: UploadFile) -> Optional[np.ndarray]:
    """
    خواندن فایل تصویر و تبدیل به آرایه NumPy.
    
    Args:
        file: فایل آپلود شده
        
    Returns:
        numpy.ndarray: تصویر به فرمت OpenCV یا None در صورت خطا
    """
    try:
        # خواندن محتوای فایل
        async with aiofiles.open("temp_image", "wb") as f:
            content = await file.read()
            await f.write(content)
        
        # بازگرداندن به ابتدای فایل
        await file.seek(0)
        
        # استفاده از OpenCV برای خواندن تصویر
        image = cv2.imread("temp_image")
        
        # حذف فایل موقت
        os.remove("temp_image")
        
        if image is None:
            logger.error("خطا در خواندن تصویر با OpenCV")
            return None
        
        return image
        
    except Exception as e:
        logger.error(f"خطا در خواندن فایل تصویر: {str(e)}")
        if os.path.exists("temp_image"):
            os.remove("temp_image")
        return None


def base64_to_opencv(base64_string: str) -> Optional[np.ndarray]:
    """
    تبدیل تصویر base64 به فرمت OpenCV.
    
    Args:
        base64_string: رشته base64 تصویر
        
    Returns:
        numpy.ndarray: تصویر به فرمت OpenCV یا None در صورت خطا
    """
    try:
        # حذف متای داده احتمالی از ابتدای رشته base64
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        # تبدیل به بایت
        image_bytes = base64.b64decode(base64_string)
        
        # تبدیل به آرایه NumPy
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # تبدیل به تصویر OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return image
        
    except (binascii.Error, ValueError, cv2.error) as e:
        logger.error(f"خطا در تبدیل base64 به تصویر: {str(e)}")
        return None


def opencv_to_base64(image: np.ndarray, image_format: str = ".jpg") -> Optional[str]:
    """
    تبدیل تصویر OpenCV به base64.
    
    Args:
        image: تصویر OpenCV
        image_format: فرمت تصویر خروجی (".jpg" یا ".png")
        
    Returns:
        str: رشته base64 تصویر یا None در صورت خطا
    """
    try:
        # بررسی فرمت تصویر
        if image_format.lower() not in [".jpg", ".jpeg", ".png"]:
            image_format = ".jpg"  # فرمت پیش‌فرض
        
        # تنظیم پارامترهای فشرده‌سازی
        if image_format.lower() in [".jpg", ".jpeg"]:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        else:  # PNG
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        
        # فشرده‌سازی تصویر
        success, encoded_image = cv2.imencode(image_format, image, encode_param)
        
        if not success:
            logger.error("خطا در فشرده‌سازی تصویر")
            return None
        
        # تبدیل به رشته base64
        base64_string = base64.b64encode(encoded_image).decode("utf-8")
        
        # افزودن متای داده به ابتدای رشته base64
        mime_type = "image/jpeg" if image_format.lower() in [".jpg", ".jpeg"] else "image/png"
        return f"data:{mime_type};base64,{base64_string}"
        
    except Exception as e:
        logger.error(f"خطا در تبدیل تصویر به base64: {str(e)}")
        return None


def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    تغییر اندازه تصویر با حفظ نسبت ابعاد.
    
    Args:
        image: تصویر OpenCV
        max_size: حداکثر اندازه (عرض یا ارتفاع)
        
    Returns:
        numpy.ndarray: تصویر با اندازه جدید
    """
    # دریافت ابعاد تصویر
    height, width = image.shape[:2]
    
    # اگر اندازه کمتر از حداکثر است، تغییری ایجاد نکن
    if height <= max_size and width <= max_size:
        return image
    
    # محاسبه نسبت ابعاد و مقیاس
    if height > width:
        ratio = max_size / height
    else:
        ratio = max_size / width
    
    # تغییر اندازه تصویر
    new_size = (int(width * ratio), int(height * ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    return resized_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    نرمال‌سازی تصویر برای پردازش بهتر.
    
    Args:
        image: تصویر OpenCV
        
    Returns:
        numpy.ndarray: تصویر نرمال‌سازی شده
    """
    # تغییر اندازه به ابعاد استاندارد
    image = resize_image(image)
    
    # تبدیل به سیاه و سفید برای پردازش بهتر
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # بهبود کنتراست
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # حذف نویز
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # تبدیل به BGR برای پردازش‌های بعدی
    normalized = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    return normalized


def crop_to_face(image: np.ndarray, face_coordinates: dict) -> np.ndarray:
    """
    برش تصویر به محدوده چهره.
    
    Args:
        image: تصویر OpenCV
        face_coordinates: مختصات چهره
        
    Returns:
        numpy.ndarray: تصویر برش خورده
    """
    # استخراج مختصات چهره
    x, y, w, h = face_coordinates["x"], face_coordinates["y"], face_coordinates["width"], face_coordinates["height"]
    
    # اضافه کردن حاشیه (20%)
    padding_x = int(w * 0.2)
    padding_y = int(h * 0.2)
    
    # مختصات جدید با حاشیه
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(image.shape[1], x + w + padding_x)
    y2 = min(image.shape[0], y + h + padding_y)
    
    # برش تصویر
    face_image = image[y1:y2, x1:x2]
    
    return face_image