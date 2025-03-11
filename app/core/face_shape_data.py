# app/core/face_shape_data.py
import json
import logging
import os
from typing import Dict, Any, List

from app.config import settings

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# دیکشنری اطلاعات شکل‌های چهره
_face_shape_info = None


def load_face_shape_data() -> Dict[str, Any]:
    """
    بارگیری اطلاعات مربوط به شکل‌های چهره و فریم‌های مناسب.
    
    Returns:
        dict: اطلاعات شکل‌های چهره و فریم‌های مناسب
    """
    global _face_shape_info
    
    if _face_shape_info is not None:
        return _face_shape_info
    
    try:
        # بررسی وجود فایل داده
        if not os.path.exists(settings.FACE_SHAPE_DATA_PATH):
            logger.warning(f"فایل داده‌های شکل چهره یافت نشد: {settings.FACE_SHAPE_DATA_PATH}")
            # استفاده از داده‌های پیش‌فرض
            return _get_default_face_shape_info()
        
        # خواندن فایل JSON
        with open(settings.FACE_SHAPE_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        _face_shape_info = data
        logger.info("اطلاعات شکل چهره با موفقیت بارگیری شد")
        return _face_shape_info
        
    except Exception as e:
        logger.error(f"خطا در بارگیری اطلاعات شکل چهره: {str(e)}")
        # استفاده از داده‌های پیش‌فرض
        return _get_default_face_shape_info()


def _get_default_face_shape_info() -> Dict[str, Any]:
    """
    ایجاد داده‌های پیش‌فرض شکل چهره.
    
    Returns:
        dict: داده‌های پیش‌فرض شکل چهره
    """
    return {
        "face_shapes": {
            "OVAL": {
                "name": "بیضی",
                "description": "صورت بیضی متعادل‌ترین شکل صورت است. پهنای گونه‌ها با پهنای پیشانی و فک متناسب است.",
                "recommendation": "اکثر فریم‌ها برای این نوع صورت مناسب هستند، اما فریم‌های مستطیلی و مربعی بهترین گزینه هستند."
            },
            "ROUND": {
                "name": "گرد",
                "description": "صورت گرد دارای عرض و طول تقریباً یکسان است و فاقد زوایای مشخص است.",
                "recommendation": "فریم‌های مستطیلی و مربعی که باعث ایجاد زاویه می‌شوند، مناسب هستند."
            },
            "SQUARE": {
                "name": "مربعی",
                "description": "صورت مربعی دارای فک زاویه‌دار و پهنای پیشانی و فک تقریباً یکسان است.",
                "recommendation": "فریم‌های گرد و بیضی که خطوط صورت را نرم‌تر می‌کنند، مناسب هستند."
            },
            "HEART": {
                "name": "قلبی",
                "description": "صورت قلبی دارای پیشانی پهن و فک باریک است.",
                "recommendation": "فریم‌های گرد و بیضی که در قسمت پایین پهن‌تر هستند، مناسب هستند."
            },
            "OBLONG": {
                "name": "کشیده",
                "description": "صورت کشیده دارای طول بیشتر نسبت به عرض است.",
                "recommendation": "فریم‌های گرد و مربعی با عمق بیشتر مناسب هستند."
            },
            "DIAMOND": {
                "name": "لوزی",
                "description": "صورت لوزی دارای گونه‌های برجسته و پیشانی و فک باریک است.",
                "recommendation": "فریم‌های گربه‌ای و بیضی که خط ابرو را برجسته می‌کنند، مناسب هستند."
            },
            "TRIANGLE": {
                "name": "مثلثی",
                "description": "صورت مثلثی دارای پیشانی باریک و فک پهن است.",
                "recommendation": "فریم‌های که در قسمت بالا پررنگ‌تر هستند، مناسب هستند."
            }
        },
        "frame_types": {
            "OVAL": ["مستطیلی", "مربعی", "هشت‌ضلعی", "گربه‌ای", "بیضی"],
            "ROUND": ["مستطیلی", "مربعی", "هشت‌ضلعی", "هاوایی"],
            "SQUARE": ["گرد", "بیضی", "گربه‌ای", "هاوایی"],
            "HEART": ["گرد", "بیضی", "هاوایی", "پایین‌بدون‌فریم"],
            "OBLONG": ["مربعی", "گرد", "گربه‌ای", "هاوایی"],
            "DIAMOND": ["گربه‌ای", "هاوایی", "بیضی", "بدون‌فریم"],
            "TRIANGLE": ["گربه‌ای", "مستطیلی", "هاوایی", "بالا‌پررنگ"]
        }
    }


def get_recommended_frame_types(face_shape: str) -> List[str]:
    """
    دریافت انواع فریم پیشنهادی براساس شکل چهره.
    
    Args:
        face_shape: شکل چهره
        
    Returns:
        list: انواع فریم پیشنهادی
    """
    # بارگیری اطلاعات شکل چهره
    face_shape_data = load_face_shape_data()
    
    # دریافت فریم‌های پیشنهادی
    frame_types = face_shape_data.get("frame_types", {}).get(face_shape, [])
    
    # اگر هیچ پیشنهادی نبود، یک لیست پیش‌فرض برگردان
    if not frame_types:
        logger.warning(f"هیچ نوع فریمی برای شکل چهره {face_shape} پیدا نشد. استفاده از مقادیر پیش‌فرض.")
        frame_types = ["مستطیلی", "گرد", "گربه‌ای", "هاوایی"]
    
    return frame_types