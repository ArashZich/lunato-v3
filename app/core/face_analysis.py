import cv2
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple, List, Optional
import math
from datetime import datetime
import csv

from app.config import settings
from app.core.face_detection import detect_face_landmarks, visualize_landmarks
from app.services.classifier import predict_face_shape
# واردسازی از فایل جدید
from app.core.face_shape_data import load_face_shape_data, get_recommended_frame_types

# تنظیمات لاگر
logger = logging.getLogger(__name__)


def analyze_face_shape(image: np.ndarray, face_coordinates: Dict[str, int]) -> Dict[str, Any]:
    """تحلیل شکل چهره با استفاده از نسبت‌های هندسی دقیق‌تر"""
    try:
        logger.info("شروع تحلیل شکل چهره با روش هندسی...")
        # دریافت نقاط کلیدی چهره
        landmarks = detect_face_landmarks(image, face_coordinates)

        if landmarks is None:
            logger.error("امکان تشخیص نقاط کلیدی چهره وجود ندارد")
            return {
                "success": False,
                "message": "امکان تشخیص نقاط کلیدی چهره وجود ندارد"
            }

        # استخراج مقیاس‌های مهم چهره
        logger.info("محاسبه نسبت‌های هندسی چهره...")

        # عرض پیشانی - فاصله بین نقاط پیشانی
        forehead_width = np.linalg.norm(landmarks[0] - landmarks[2])
        logger.debug(f"عرض پیشانی: {forehead_width:.2f}")

        # عرض گونه‌ها - فاصله بین نقاط گونه
        cheekbone_width = np.linalg.norm(landmarks[6] - landmarks[8])
        logger.debug(f"عرض گونه‌ها: {cheekbone_width:.2f}")

        # عرض فک - فاصله بین نقاط فک
        jawline_width = np.linalg.norm(landmarks[3] - landmarks[5])
        logger.debug(f"عرض فک: {jawline_width:.2f}")

        # طول صورت - فاصله از چانه تا وسط پیشانی
        # استفاده از میانگین نقاط پیشانی برای تخمین بهتر نقطه بالای پیشانی
        forehead_mid = (landmarks[0] + landmarks[2]) / 2
        face_length = np.linalg.norm(landmarks[4] - forehead_mid)
        logger.debug(f"طول صورت: {face_length:.2f}")

        # اضافه کردن لاگ برای نقاط کلیدی
        logger.debug(f"نقاط کلیدی شناسایی شده: {landmarks}")

        # محاسبه نسبت‌های کلیدی با بررسی صحت مقادیر
        if face_length > 0 and cheekbone_width > 0 and jawline_width > 0:
            width_to_length_ratio = cheekbone_width / face_length

            # محدود کردن نسبت عرض به طول به مقادیر منطقی
            if width_to_length_ratio < 0.65:
                width_to_length_ratio = 0.65
                logger.warning(
                    f"نسبت عرض به طول ({width_to_length_ratio:.2f}) خیلی کوچک است - اصلاح شد")
            elif width_to_length_ratio > 1.2:
                width_to_length_ratio = 1.2
                logger.warning(
                    f"نسبت عرض به طول ({width_to_length_ratio:.2f}) خیلی بزرگ است - اصلاح شد")

            # بررسی و اصلاح نسبت گونه به فک
            if jawline_width < cheekbone_width * 0.3:  # فک خیلی باریک تشخیص داده شده
                logger.warning(
                    "تشخیص فک خیلی باریک - محدود کردن نسبت گونه به فک")
                # حداقل عرض فک را 45% عرض گونه در نظر می‌گیریم
                adjusted_jawline_width = max(
                    jawline_width, cheekbone_width * 0.45)
                cheekbone_to_jaw_ratio = cheekbone_width / adjusted_jawline_width
                logger.info(
                    f"نسبت گونه به فک اصلاح شده: {cheekbone_to_jaw_ratio:.2f} (قبلی: {cheekbone_width / jawline_width:.2f})")
            else:
                cheekbone_to_jaw_ratio = cheekbone_width / jawline_width

            # محدود کردن نسبت گونه به فک به مقادیر منطقی
            if cheekbone_to_jaw_ratio > 2.3:
                cheekbone_to_jaw_ratio = 2.3
                logger.warning(
                    f"نسبت گونه به فک ({cheekbone_to_jaw_ratio:.2f}) خیلی بزرگ است - اصلاح شد")
            elif cheekbone_to_jaw_ratio < 0.7:
                cheekbone_to_jaw_ratio = 0.7
                logger.warning(
                    f"نسبت گونه به فک ({cheekbone_to_jaw_ratio:.2f}) خیلی کوچک است - اصلاح شد")

            forehead_to_cheekbone_ratio = forehead_width / cheekbone_width

            # محدود کردن نسبت پیشانی به گونه به مقادیر منطقی
            if forehead_to_cheekbone_ratio > 1.2:
                forehead_to_cheekbone_ratio = 1.2
                logger.warning(
                    f"نسبت پیشانی به گونه ({forehead_to_cheekbone_ratio:.2f}) خیلی بزرگ است - اصلاح شد")
            elif forehead_to_cheekbone_ratio < 0.1:
                forehead_to_cheekbone_ratio = 0.1
                logger.warning(
                    f"نسبت پیشانی به گونه ({forehead_to_cheekbone_ratio:.2f}) خیلی کوچک است - اصلاح شد")

            # لاگ کردن مقادیر واقعی محاسبه شده
            logger.info(f"مقادیر واقعی محاسبه شده:")
            logger.info(f"عرض پیشانی: {forehead_width}")
            logger.info(f"عرض گونه‌ها: {cheekbone_width}")
            logger.info(f"عرض فک: {jawline_width}")
            logger.info(f"طول صورت: {face_length}")
            logger.info(f"نسبت عرض به طول: {width_to_length_ratio}")
            logger.info(f"نسبت گونه به فک: {cheekbone_to_jaw_ratio}")
            logger.info(f"نسبت پیشانی به گونه: {forehead_to_cheekbone_ratio}")
        else:
            # مقادیر پیش‌فرض در صورت عدم موفقیت محاسبات
            width_to_length_ratio = 0.8
            cheekbone_to_jaw_ratio = 1.2
            forehead_to_cheekbone_ratio = 0.7

        # محاسبه زاویه فک
        chin_to_jaw_left_vec = landmarks[3] - landmarks[4]
        chin_to_jaw_right_vec = landmarks[5] - landmarks[4]

        # محاسبه شباهت کسینوسی بین بردارها با دقت بیشتر
        if np.linalg.norm(chin_to_jaw_left_vec) > 0 and np.linalg.norm(chin_to_jaw_right_vec) > 0:
            dot_product = np.dot(chin_to_jaw_left_vec, chin_to_jaw_right_vec)
            norm_left = np.linalg.norm(chin_to_jaw_left_vec)
            norm_right = np.linalg.norm(chin_to_jaw_right_vec)

            cos_angle = dot_product / (norm_left * norm_right)
            # اطمینان از اینکه مقدار در محدوده معتبر برای arccos قرار دارد
            cos_angle = min(1.0, max(-1.0, cos_angle))
            jaw_angle = np.arccos(cos_angle) * 180 / np.pi  # تبدیل به درجه
        else:
            jaw_angle = 150.0  # مقدار پیش‌فرض

        # تعیین شکل چهره براساس نسبت‌های هندسی
        shape_metrics = {
            "width_to_length_ratio": float(width_to_length_ratio),
            "cheekbone_to_jaw_ratio": float(cheekbone_to_jaw_ratio),
            "forehead_to_cheekbone_ratio": float(forehead_to_cheekbone_ratio),
            "jaw_angle": float(jaw_angle)
        }

        logger.info(f"نسبت‌های محاسبه شده: {shape_metrics}")

        # تعیین شکل چهره
        face_shape = _determine_face_shape(shape_metrics)

        # ثبت لاگ با شکل چهره تشخیص داده شده
        log_face_metrics(shape_metrics, face_shape)

        # محاسبه میزان اطمینان
        confidence = _calculate_confidence(shape_metrics, face_shape)

        # دریافت توضیحات و توصیه‌ها
        face_shape_data = load_face_shape_data()
        face_shape_info = face_shape_data.get(
            "face_shapes", {}).get(face_shape, {})

        description = face_shape_info.get("description", "")
        recommendation = face_shape_info.get("recommendation", "")

        logger.info(
            f"تحلیل شکل چهره با روش هندسی انجام شد: {face_shape} با اطمینان {confidence:.1f}%")

        # ذخیره تصویر با نقاط کلیدی و شکل چهره برای عیب‌یابی
        try:
            visualized_img = visualize_landmarks(image, landmarks, face_shape)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                f"{debug_dir}/result_{face_shape}_{timestamp}.jpg", visualized_img)
            logger.info(
                f"تصویر نتیجه نهایی در فایل {debug_dir}/result_{face_shape}_{timestamp}.jpg ذخیره شد")
        except Exception as vis_error:
            logger.warning(f"خطا در ذخیره تصویر نتیجه: {str(vis_error)}")

        return {
            "success": True,
            "face_shape": face_shape,
            "confidence": confidence,
            "shape_metrics": shape_metrics,
            "description": description,
            "recommendation": recommendation
        }

    except Exception as e:
        logger.error(f"خطا در تحلیل شکل چهره: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تحلیل شکل چهره: {str(e)}"
        }


def _calculate_confidence(metrics: Dict[str, float], face_shape: str) -> float:
    """
    محاسبه میزان اطمینان تشخیص شکل چهره.

    Args:
        metrics: نسبت‌های هندسی
        face_shape: شکل چهره تشخیص داده شده

    Returns:
        float: میزان اطمینان (0-100)
    """
    # تعریف مقادیر ایده‌آل برای هر شکل چهره - تنها شامل 5 شکل مورد نظر
    ideal_metrics = {
        "OVAL": {
            "width_to_length_ratio": 0.8,
            "cheekbone_to_jaw_ratio": 1.3,
            "forehead_to_cheekbone_ratio": 0.7,
            "jaw_angle": 160
        },
        "ROUND": {
            "width_to_length_ratio": 0.95,
            "cheekbone_to_jaw_ratio": 1.1,
            "forehead_to_cheekbone_ratio": 0.85,
            "jaw_angle": 170
        },
        "SQUARE": {
            "width_to_length_ratio": 0.95,
            "cheekbone_to_jaw_ratio": 1.0,
            "forehead_to_cheekbone_ratio": 0.8,
            "jaw_angle": 140
        },
        "HEART": {
            "width_to_length_ratio": 0.8,
            "cheekbone_to_jaw_ratio": 1.5,
            "forehead_to_cheekbone_ratio": 1.0,
            "jaw_angle": 160
        },
        "OBLONG": {
            "width_to_length_ratio": 0.7,
            "cheekbone_to_jaw_ratio": 1.2,
            "forehead_to_cheekbone_ratio": 0.7,
            "jaw_angle": 160
        }
    }

    # دریافت مقادیر ایده‌آل برای شکل چهره تشخیص داده شده
    ideal = ideal_metrics.get(face_shape, ideal_metrics["OVAL"])

    # محاسبه میزان انحراف از مقادیر ایده‌آل
    deviations = [
        1 - min(abs(metrics["width_to_length_ratio"] -
                ideal["width_to_length_ratio"]) / ideal["width_to_length_ratio"], 1),
        1 - min(abs(metrics["cheekbone_to_jaw_ratio"] -
                ideal["cheekbone_to_jaw_ratio"]) / ideal["cheekbone_to_jaw_ratio"], 1),
        1 - min(abs(metrics["forehead_to_cheekbone_ratio"] -
                ideal["forehead_to_cheekbone_ratio"]) / ideal["forehead_to_cheekbone_ratio"], 1),
        1 - min(abs(metrics["jaw_angle"] -
                ideal["jaw_angle"]) / ideal["jaw_angle"], 1)
    ]

    # وزن‌دهی به انحرافات
    weights = [0.3, 0.35, 0.25, 0.1]  # افزایش وزن نسبت گونه به فک
    weighted_avg = sum(d * w for d, w in zip(deviations, weights))

    # تبدیل به درصد
    confidence = weighted_avg * 100

    # محدود کردن به بازه 60-95
    confidence = max(60, min(95, confidence))

    return round(confidence, 1)


def generate_full_analysis(image: np.ndarray, face_coordinates: Dict[str, int]) -> Dict[str, Any]:
    """
    تحلیل کامل شکل چهره با استفاده از ترکیب روش‌های هندسی و یادگیری ماشین.

    Args:
        image: تصویر OpenCV
        face_coordinates: مختصات چهره

    Returns:
        dict: نتیجه تحلیل شکل چهره
    """
    try:
        logger.info("شروع تحلیل کامل شکل چهره...")

        # 1. استفاده از روش هندسی (قابل اعتمادتر)
        geometric_result = analyze_face_shape(image, face_coordinates)
        geometric_success = geometric_result.get("success", False)

        # 2. تلاش برای استفاده از مدل ML (به عنوان کمکی)
        ml_success = False
        ml_face_shape = None
        ml_confidence = 0

        try:
            ml_face_shape, ml_confidence, shape_details = predict_face_shape(
                image, face_coordinates)
            ml_success = True
            logger.info(
                f"تشخیص شکل چهره با مدل ML: {ml_face_shape} با میزان اطمینان {ml_confidence:.1f}%")
        except Exception as model_error:
            logger.warning(f"خطا در استفاده از مدل ML: {str(model_error)}")

        # تصمیم‌گیری نهایی - اولویت با روش هندسی
        if geometric_success:
            # استفاده از روش هندسی که قابل اعتمادتر است
            result_face_shape = geometric_result.get("face_shape")
            result_confidence = geometric_result.get("confidence")

            # اگر مدل ML هم موفق بوده و شکل یکسانی تشخیص داده، اطمینان بیشتر
            if ml_success and ml_face_shape == result_face_shape:
                # میانگین وزن‌دار با اولویت بیشتر به روش هندسی
                result_confidence = (result_confidence *
                                     0.7) + (ml_confidence * 0.3)
                logger.info(
                    f"هر دو روش نتیجه یکسان دارند: {result_face_shape} با اطمینان ترکیبی {result_confidence:.1f}%")
            else:
                logger.info(
                    f"استفاده از نتایج تحلیل هندسی: {result_face_shape} با اطمینان {result_confidence:.1f}%")
        elif ml_success:
            # استفاده از نتایج مدل ML در صورتی که تحلیل هندسی موفق نبوده است
            result_face_shape = ml_face_shape
            result_confidence = ml_confidence
            logger.info(
                f"استفاده از نتایج مدل ML: {result_face_shape} با اطمینان {result_confidence:.1f}%")
        else:
            # هر دو روش شکست خورده‌اند
            return {
                "success": False,
                "message": "هر دو روش تحلیل (هندسی و ML) با شکست مواجه شدند"
            }

        # مطمئن شویم که شکل چهره یکی از 5 نوع مورد نظر باشد
        valid_shapes = {"HEART", "OBLONG", "OVAL", "ROUND", "SQUARE"}
        if result_face_shape not in valid_shapes:
            logger.warning(
                f"شکل چهره {result_face_shape} معتبر نیست. استفاده از OVAL به عنوان پیش‌فرض.")
            result_face_shape = "OVAL"
            result_confidence = max(60, result_confidence - 15)  # کاهش اطمینان

        # دریافت توضیحات و توصیه‌ها از فایل داده
        face_shape_data = load_face_shape_data()
        face_shape_info = face_shape_data.get(
            "face_shapes", {}).get(result_face_shape, {})

        description = face_shape_info.get("description", "")
        recommendation = face_shape_info.get("recommendation", "")

        # دریافت انواع فریم پیشنهادی
        recommended_frame_types = get_recommended_frame_types(
            result_face_shape)

        logger.info(
            f"تحلیل کامل شکل چهره انجام شد. نتیجه نهایی: {result_face_shape}")

        return {
            "success": True,
            "face_shape": result_face_shape,
            "confidence": result_confidence,
            "description": description,
            "recommendation": recommendation,
            "recommended_frame_types": recommended_frame_types
        }

    except Exception as e:
        logger.error(f"خطا در تحلیل کامل شکل چهره: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تحلیل شکل چهره: {str(e)}"
        }


def _determine_face_shape(metrics: Dict[str, float]) -> str:
    """
    تعیین شکل چهره براساس نسبت‌های هندسی با استفاده از قوانین دقیق‌تر.

    Args:
        metrics: مقادیر محاسبه شده نسبت‌های هندسی

    Returns:
        str: شکل چهره تشخیص داده شده
    """
    # استخراج مقادیر از metrics
    width_to_length = metrics["width_to_length_ratio"]
    cheekbone_to_jaw = metrics["cheekbone_to_jaw_ratio"]
    forehead_to_cheekbone = metrics["forehead_to_cheekbone_ratio"]
    jaw_angle = metrics["jaw_angle"]

    # لاگ کردن مقادیر محاسبه شده برای اشکال‌زدایی
    logger.info(f"نسبت‌های محاسبه شده برای تشخیص شکل چهره:")
    logger.info(f"نسبت عرض به طول: {width_to_length:.2f}")
    logger.info(f"نسبت گونه به فک: {cheekbone_to_jaw:.2f}")
    logger.info(f"نسبت پیشانی به گونه: {forehead_to_cheekbone:.2f}")
    logger.info(f"زاویه فک: {jaw_angle:.2f} درجه")

    # قوانین اصلاح شده برای تشخیص بهتر شکل چهره

    # صورت قلبی (HEART)
    # ویژگی اصلی: پیشانی پهن و فک باریک، نسبت گونه به فک بزرگ
    if cheekbone_to_jaw > 1.35 and (forehead_to_cheekbone > 0.75 or width_to_length < 0.85):
        logger.info("تشخیص شکل چهره: HEART (قلبی)")
        return "HEART"

    # صورت گرد (ROUND)
    # ویژگی اصلی: نسبت عرض به طول نزدیک به 1، زاویه فک بزرگ
    if width_to_length > 0.85 and jaw_angle > 150 and cheekbone_to_jaw < 1.3:
        logger.info("تشخیص شکل چهره: ROUND (گرد)")
        return "ROUND"

    # صورت مربعی (SQUARE)
    # ویژگی اصلی: عرض زیاد، نسبت گونه به فک نزدیک به 1، زاویه فک کم
    if width_to_length > 0.8 and cheekbone_to_jaw < 1.2 and jaw_angle < 145:
        logger.info("تشخیص شکل چهره: SQUARE (مربعی)")
        return "SQUARE"

    # صورت کشیده (OBLONG)
    # ویژگی اصلی: طول زیاد نسبت به عرض
    if width_to_length < 0.75 and cheekbone_to_jaw < 1.35:
        logger.info("تشخیص شکل چهره: OBLONG (کشیده)")
        return "OBLONG"

    # افزایش دقت برای تشخیص متعادل‌تر - تست دوباره با شرایط کمتر سختگیرانه

    # بررسی مجدد برای HEART
    if cheekbone_to_jaw > 1.3 and forehead_to_cheekbone > 0.7:
        logger.info("تشخیص شکل چهره (شرایط ثانویه): HEART (قلبی)")
        return "HEART"

    # بررسی مجدد برای ROUND
    if width_to_length > 0.82 and jaw_angle > 140:
        logger.info("تشخیص شکل چهره (شرایط ثانویه): ROUND (گرد)")
        return "ROUND"

    # بررسی مجدد برای SQUARE
    if width_to_length > 0.78 and cheekbone_to_jaw < 1.25 and jaw_angle < 150:
        logger.info("تشخیص شکل چهره (شرایط ثانویه): SQUARE (مربعی)")
        return "SQUARE"

    # بررسی مجدد برای OBLONG با شرایط کمتر سختگیرانه
    if width_to_length < 0.8 and cheekbone_to_jaw < 1.4:
        logger.info("تشخیص شکل چهره (شرایط ثانویه): OBLONG (کشیده)")
        return "OBLONG"

    # صورت بیضی (OVAL) - شکل متعادل
    if 0.75 <= width_to_length <= 0.85 and 1.1 <= cheekbone_to_jaw <= 1.4:
        logger.info("تشخیص شکل چهره: OVAL (بیضی)")
        return "OVAL"

    # پیش‌فرض به عنوان بیضی
    logger.info("تشخیص شکل چهره: OVAL (بیضی) - پیش‌فرض")
    return "OVAL"


def log_face_metrics(metrics: Dict[str, float], face_shape: str = None, metrics_log_file: str = "face_metrics_log.csv"):
    """ثبت نسبت‌های چهره در فایل لاگ برای تحلیل آماری"""
    import os
    import csv

    try:
        file_exists = os.path.exists(metrics_log_file)

        with open(metrics_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "face_shape"] + list(metrics.keys()))

            if not file_exists:
                writer.writeheader()

            row_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "face_shape": face_shape or "",
                **metrics
            }

            writer.writerow(row_data)

        logger.debug(
            f"نسبت‌های چهره با موفقیت در فایل {metrics_log_file} ثبت شد")
    except Exception as e:
        logger.warning(f"خطا در ثبت نسبت‌های چهره: {str(e)}")


def analyze_detection_statistics(metrics_log_file: str = "face_metrics_log.csv"):
    """
    تحلیل آماری داده‌های نسبت‌های چهره برای بهبود تشخیص
    """
    try:
        import pandas as pd

        # خواندن فایل لاگ
        metrics_df = pd.read_csv(metrics_log_file)

        # آمارهای توصیفی
        stats = metrics_df.describe()
        logger.info("آمارهای توصیفی نسبت‌های چهره:")
        logger.info(f"\n{stats}")

        # بررسی توزیع شکل‌های چهره
        if 'face_shape' in metrics_df.columns:
            shape_counts = metrics_df["face_shape"].value_counts()
            logger.info("\nتعداد هر شکل چهره:")
            logger.info(f"\n{shape_counts}")

            # میانگین نسبت‌ها به تفکیک شکل چهره
            logger.info("\nمیانگین نسبت‌ها به تفکیک شکل چهره:")
            shape_metrics = metrics_df.groupby("face_shape").mean()
            logger.info(f"\n{shape_metrics}")

        return True

    except Exception as e:
        logger.error(f"خطا در تحلیل آماری: {str(e)}")
        return False


def improve_face_shape_detection():
    """
    اجرای فرآیند بهبود سیستم تشخیص شکل چهره با استفاده از داده‌های آماری
    """
    try:
        # تحلیل آماری داده‌های موجود
        analyze_detection_statistics()

        # ایجاد دایرکتوری برای تصاویر عیب‌یابی
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)

        logger.info("فرآیند بهبود سیستم تشخیص شکل چهره با موفقیت اجرا شد")
        return True

    except Exception as e:
        logger.error(f"خطا در بهبود سیستم تشخیص شکل چهره: {str(e)}")
        return False
