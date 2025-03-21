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
            logger.warning(
                f"فایل داده‌های شکل چهره یافت نشد: {settings.FACE_SHAPE_DATA_PATH}")
            # استفاده از داده‌های پیش‌فرض
            return _get_default_face_shape_info()

        # خواندن فایل JSON
        with open(settings.FACE_SHAPE_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        _face_shape_info = data

        # فیلتر کردن داده‌ها برای فقط 5 شکل چهره مورد نظر
        valid_shapes = {"HEART", "OBLONG", "OVAL", "ROUND", "SQUARE"}

        # فیلتر کردن شکل‌های چهره
        if "face_shapes" in _face_shape_info:
            filtered_shapes = {}
            for shape, info in _face_shape_info["face_shapes"].items():
                if shape in valid_shapes:
                    filtered_shapes[shape] = info
            _face_shape_info["face_shapes"] = filtered_shapes

        # فیلتر کردن فریم‌های پیشنهادی
        if "frame_types" in _face_shape_info:
            filtered_frames = {}
            for shape, frames in _face_shape_info["frame_types"].items():
                if shape in valid_shapes:
                    filtered_frames[shape] = frames
            _face_shape_info["frame_types"] = filtered_frames

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
            }
        },
        "frame_types": {
            "OVAL": ["مستطیلی", "مربعی", "هشت‌ضلعی", "گربه‌ای", "بیضی"],
            "ROUND": ["مستطیلی", "مربعی", "هشت‌ضلعی", "هاوایی"],
            "SQUARE": ["گرد", "بیضی", "گربه‌ای", "هاوایی"],
            "HEART": ["گرد", "بیضی", "هاوایی", "پایین‌بدون‌فریم"],
            "OBLONG": ["مربعی", "گرد", "گربه‌ای", "هاوایی"]
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
        logger.warning(
            f"هیچ نوع فریمی برای شکل چهره {face_shape} پیدا نشد. استفاده از مقادیر پیش‌فرض.")

        # پیش‌فرض‌های اختصاصی برای هر شکل چهره
        default_frames = {
            "OVAL": ["مستطیلی", "مربعی", "هشت‌ضلعی"],
            "ROUND": ["مستطیلی", "مربعی", "هاوایی"],
            "SQUARE": ["گرد", "بیضی", "گربه‌ای"],
            "HEART": ["گرد", "بیضی", "پایین‌بدون‌فریم"],
            "OBLONG": ["مربعی", "گرد", "گربه‌ای"]
        }

        frame_types = default_frames.get(
            face_shape, ["مستطیلی", "گرد", "گربه‌ای"])

    logger.info(f"فریم‌های پیشنهادی برای شکل چهره {face_shape}: {frame_types}")
    return frame_types
