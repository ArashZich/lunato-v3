import os
import numpy as np
import logging
import pickle
from typing import Dict, List, Tuple, Any, Optional
import cv2
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.config import settings
from app.core.face_detection import detect_face_landmarks

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# متغیرهای سراسری برای مدل و اسکیلر
model = None
scaler = None


def load_model():
    """
    بارگیری مدل scikit-learn از فایل.

    Returns:
        bool: نتیجه بارگیری مدل
    """
    global model, scaler

    if model is not None and scaler is not None:
        return True

    try:
        # بررسی وجود فایل مدل
        if not os.path.exists(settings.FACE_SHAPE_MODEL_PATH):
            logger.warning(
                f"فایل مدل در مسیر {settings.FACE_SHAPE_MODEL_PATH} یافت نشد")
            return False

        # بارگیری مدل
        model = joblib.load(settings.FACE_SHAPE_MODEL_PATH)

        # بارگیری اسکیلر
        scaler_path = os.path.splitext(settings.FACE_SHAPE_MODEL_PATH)[
            0] + "_scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("اسکیلر با موفقیت بارگیری شد")
        else:
            # ساخت اسکیلر پیش‌فرض
            scaler = StandardScaler()
            logger.warning(
                f"فایل اسکیلر در مسیر {scaler_path} یافت نشد، از اسکیلر پیش‌فرض استفاده می‌شود")

        logger.info("مدل با موفقیت بارگیری شد")
        return True

    except Exception as e:
        logger.error(f"خطا در بارگیری مدل: {str(e)}")
        model = None
        scaler = None
        return False


def extract_features_for_classification(image: np.ndarray, face_coordinates: Dict[str, int]) -> np.ndarray:
    """
    استخراج ویژگی‌های چهره برای طبقه‌بندی.

    Args:
        image: تصویر OpenCV
        face_coordinates: مختصات چهره

    Returns:
        numpy.ndarray: بردار ویژگی‌ها
    """
    # دریافت نقاط کلیدی چهره
    landmarks = detect_face_landmarks(image, face_coordinates)

    if landmarks is None:
        raise ValueError("نقاط کلیدی چهره قابل تشخیص نیست")

    # تبدیل به آرایه numpy
    landmarks_np = np.array(landmarks)

    # محاسبه مقیاس‌های مهم چهره
    # 1. عرض پیشانی (نقاط 0 و 2)
    forehead_width = np.linalg.norm(landmarks_np[0] - landmarks_np[2])

    # 2. عرض گونه‌ها (نقاط 6 و 8)
    cheekbone_width = np.linalg.norm(landmarks_np[6] - landmarks_np[8])

    # 3. عرض فک (نقاط 3 و 5)
    jawline_width = np.linalg.norm(landmarks_np[3] - landmarks_np[5])

    # 4. طول صورت (میانگین نقاط 0، 1، 2 تا نقطه 4)
    face_top = np.mean(landmarks_np[0:3], axis=0)
    face_bottom = landmarks_np[4]  # چانه
    face_length = np.linalg.norm(face_top - face_bottom)

    # 5. زاویه فک (بین وکتورهای 3-4 و 4-5)
    chin_to_left_jaw = landmarks_np[3] - landmarks_np[4]
    chin_to_right_jaw = landmarks_np[5] - landmarks_np[4]

    # محاسبه کسینوس زاویه بین دو بردار
    if np.linalg.norm(chin_to_left_jaw) > 0 and np.linalg.norm(chin_to_right_jaw) > 0:
        cos_angle = np.dot(chin_to_left_jaw, chin_to_right_jaw) / (
            np.linalg.norm(chin_to_left_jaw) *
            np.linalg.norm(chin_to_right_jaw)
        )
        # اطمینان از محدوده مجاز کسینوس
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        jaw_angle = np.arccos(cos_angle) * 180 / np.pi  # تبدیل به درجه
    else:
        jaw_angle = 150.0  # مقدار پیش‌فرض

    # محاسبه نسبت‌های کلیدی
    width_to_length_ratio = cheekbone_width / face_length if face_length > 0 else 0
    cheekbone_to_jaw_ratio = cheekbone_width / \
        jawline_width if jawline_width > 0 else 0
    forehead_to_cheekbone_ratio = forehead_width / \
        cheekbone_width if cheekbone_width > 0 else 0

    # ویژگی‌های اضافی مطابق با آموزش مدل
    face_shape_ratio = width_to_length_ratio
    face_taper_ratio = jawline_width / forehead_width if forehead_width > 0 else 0

    # لاگ کردن ویژگی‌های استخراج شده
    logger.debug(f"ویژگی‌های استخراج شده:")
    logger.debug(f"1. نسبت عرض به طول: {width_to_length_ratio:.3f}")
    logger.debug(f"2. نسبت گونه به فک: {cheekbone_to_jaw_ratio:.3f}")
    logger.debug(f"3. نسبت پیشانی به گونه: {forehead_to_cheekbone_ratio:.3f}")
    logger.debug(f"4. زاویه فک: {jaw_angle:.3f}")
    logger.debug(f"5. نسبت شکل صورت: {face_shape_ratio:.3f}")
    logger.debug(f"6. نسبت باریک‌شدگی: {face_taper_ratio:.3f}")

    # ساخت بردار ویژگی‌ها برای طبقه‌بندی - دقیقاً مطابق با ویژگی‌های آموزش مدل
    features = np.array([
        width_to_length_ratio,
        cheekbone_to_jaw_ratio,
        forehead_to_cheekbone_ratio,
        jaw_angle,
        face_shape_ratio,
        face_taper_ratio
    ]).reshape(1, -1)

    return features


def predict_face_shape(image: np.ndarray, face_coordinates: Dict[str, int]) -> Tuple[str, float, Dict[str, float]]:
    """
    پیش‌بینی شکل چهره با استفاده از مدل scikit-learn.

    Args:
        image: تصویر OpenCV
        face_coordinates: مختصات چهره

    Returns:
        tuple: (شکل_چهره، اطمینان، جزئیات_شکل)
    """
    # بارگیری مدل اگر هنوز بارگیری نشده است
    if not load_model():
        raise ValueError("بارگیری مدل ناموفق بود")

    # استخراج ویژگی‌ها
    features = extract_features_for_classification(image, face_coordinates)

    # مقیاس‌دهی ویژگی‌ها
    if scaler is not None:
        scaled_features = scaler.transform(features)
    else:
        scaled_features = features

    # پیش‌بینی شکل چهره
    face_shape_proba = model.predict_proba(scaled_features)[0]
    face_shape_idx = np.argmax(face_shape_proba)
    face_shape = model.classes_[face_shape_idx]
    confidence = face_shape_proba[face_shape_idx] * 100

    # محدود کردن به ۵ شکل چهره مورد نظر
    valid_shapes = {"HEART", "OBLONG", "OVAL", "ROUND", "SQUARE"}

    # اگر شکل تشخیص داده شده در لیست معتبر نباشد، نزدیک‌ترین شکل را انتخاب کنیم
    if face_shape not in valid_shapes:
        logger.warning(
            f"شکل چهره {face_shape} در لیست شکل‌های معتبر نیست. استفاده از شکل جایگزین.")

        # پیدا کردن نزدیک‌ترین شکل معتبر
        valid_shape_indices = [i for i, shape in enumerate(
            model.classes_) if shape in valid_shapes]
        if valid_shape_indices:
            valid_probs = [face_shape_proba[i] for i in valid_shape_indices]
            max_idx = valid_shape_indices[np.argmax(valid_probs)]
            face_shape = model.classes_[max_idx]
            confidence = face_shape_proba[max_idx] * 100
        else:
            # اگر هیچ شکل معتبری نباشد (که بعید است)، از پیش‌فرض OVAL استفاده می‌کنیم
            face_shape = "OVAL"
            confidence = 70.0

    # ساخت دیکشنری امتیازات برای هر شکل چهره معتبر
    shape_details = {}
    for idx, shape in enumerate(model.classes_):
        if shape in valid_shapes:
            shape_details[shape] = float(face_shape_proba[idx] * 100)

    logger.info(
        f"شکل چهره تشخیص داده شده: {face_shape} با میزان اطمینان {confidence:.2f}%")

    return face_shape, confidence, shape_details


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None
) -> Tuple[SVC, StandardScaler]:
    """
    آموزش مدل طبقه‌بندی برای تشخیص شکل چهره.

    Args:
        X: داده‌های ویژگی
        y: برچسب‌های شکل چهره
        save_path: مسیر ذخیره مدل (اختیاری)

    Returns:
        tuple: (مدل_آموزش_دیده، اسکیلر)
    """
    # مقیاس‌دهی داده‌ها
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ایجاد و آموزش مدل
    svm_model = SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=42)
    svm_model.fit(X_scaled, y)

    # ذخیره مدل اگر مسیر ذخیره مشخص شده باشد
    if save_path is not None:
        # ایجاد دایرکتوری در صورت عدم وجود
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # ذخیره مدل
        joblib.dump(svm_model, save_path)

        # ذخیره اسکیلر
        scaler_path = os.path.splitext(save_path)[0] + "_scaler.pkl"
        joblib.dump(scaler, scaler_path)

        logger.info(f"مدل و اسکیلر با موفقیت در مسیر {save_path} ذخیره شدند")

    return svm_model, scaler
