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
            logger.warning(f"فایل مدل در مسیر {settings.FACE_SHAPE_MODEL_PATH} یافت نشد")
            return False
        
        # بارگیری مدل
        model = joblib.load(settings.FACE_SHAPE_MODEL_PATH)
        
        # بارگیری اسکیلر اگر وجود داشته باشد
        scaler_path = os.path.splitext(settings.FACE_SHAPE_MODEL_PATH)[0] + "_scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            # ساخت اسکیلر پیش‌فرض
            scaler = StandardScaler()
        
        logger.info("مدل و اسکیلر با موفقیت بارگیری شدند")
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
    # عرض پیشانی
    forehead_width = np.linalg.norm(landmarks[0] - landmarks[8])
    
    # عرض گونه‌ها
    cheekbone_width = np.linalg.norm(landmarks[2] - landmarks[6])
    
    # عرض فک
    jawline_width = np.linalg.norm(landmarks[3] - landmarks[5])
    
    # طول صورت
    chin_point = landmarks[4]  # مرکز چانه
    forehead_point = (landmarks[0] + landmarks[8]) // 2  # وسط پیشانی
    face_length = np.linalg.norm(chin_point - forehead_point)
    
    # محاسبه نسبت‌های کلیدی
    width_to_length_ratio = cheekbone_width / face_length if face_length > 0 else 0
    cheekbone_to_jaw_ratio = cheekbone_width / jawline_width if jawline_width > 0 else 0
    forehead_to_cheekbone_ratio = forehead_width / cheekbone_width if cheekbone_width > 0 else 0
    
    # محاسبه زاویه فک
    chin_to_jaw_left_vec = landmarks[3] - landmarks[4]
    chin_to_jaw_right_vec = landmarks[5] - landmarks[4]
    
    # محاسبه شباهت کسینوسی بین بردارها
    if np.linalg.norm(chin_to_jaw_left_vec) > 0 and np.linalg.norm(chin_to_jaw_right_vec) > 0:
        cos_angle = np.dot(chin_to_jaw_left_vec, chin_to_jaw_right_vec) / (
            np.linalg.norm(chin_to_jaw_left_vec) * np.linalg.norm(chin_to_jaw_right_vec)
        )
        # اطمینان از اینکه مقدار در محدوده معتبر برای arccos قرار دارد
        cos_angle = min(1.0, max(-1.0, cos_angle))
        jaw_angle = np.arccos(cos_angle) * 180 / np.pi  # تبدیل به درجه
    else:
        jaw_angle = 0.0
    
    # ساخت بردار ویژگی‌ها برای طبقه‌بندی
    features = np.array([
        width_to_length_ratio,
        cheekbone_to_jaw_ratio,
        forehead_to_cheekbone_ratio,
        jaw_angle
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
    
    # ساخت دیکشنری امتیازات برای هر شکل چهره
    shape_details = {}
    for idx, shape in enumerate(model.classes_):
        shape_details[shape] = float(face_shape_proba[idx] * 100)
    
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
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
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