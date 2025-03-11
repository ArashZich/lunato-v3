import os
import cv2
import numpy as np
import joblib
import logging
import mediapipe as mp
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# تنظیم لاگر
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# مسیرهای مدل و خروجی
MODEL_PATH = 'data/face_shape_model.pkl'
SCALER_PATH = 'data/face_shape_model_scaler.pkl'
OUTPUT_DIR = 'test_results'

# نگاشت دسته‌ها به نام‌های فارسی برای نمایش
SHAPE_NAMES = {
    'HEART': 'قلبی',
    'OBLONG': 'کشیده',
    'OVAL': 'بیضی',
    'ROUND': 'گرد',
    'SQUARE': 'مربعی'
}

# نگاشت دسته‌ها به رنگ‌ها برای نمایش
SHAPE_COLORS = {
    'HEART': (255, 0, 0),    # قرمز
    'OBLONG': (0, 255, 0),   # سبز
    'OVAL': (0, 0, 255),     # آبی
    'ROUND': (255, 255, 0),  # زرد
    'SQUARE': (255, 0, 255)  # بنفش
}


class FaceShapeTester:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """مقداردهی اولیه کلاس تست مدل تشخیص شکل چهره"""
        # راه‌اندازی MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # نقاط کلیدی مهم برای تشخیص شکل چهره
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                          400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.FACE_TOP = [10, 338, 297, 332, 284,
                         251, 389, 356, 454, 323, 361, 288]
        self.FACE_BOTTOM = [397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                            150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.LEFT_CHEEK = [123, 50, 36, 137, 205, 206, 177, 147]
        self.RIGHT_CHEEK = [352, 345, 372, 383, 425, 427, 425, 366]
        self.JAW_LINE = [152, 176, 150, 136, 172, 58, 132, 93, 234]

        # بارگیری مدل و اسکیلر
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(
                f"مدل و اسکیلر با موفقیت از {model_path} و {scaler_path} بارگیری شدند")
        except Exception as e:
            logger.error(f"خطا در بارگیری مدل یا اسکیلر: {str(e)}")
            raise

    def extract_features_from_image(self, image_path=None, image=None):
        """استخراج ویژگی‌های چهره از تصویر با استفاده از MediaPipe"""
        try:
            # خواندن تصویر
            if image_path:
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"نمی‌توان تصویر را بخوانیم: {image_path}")
                    return None, None
            elif image is not None:
                img = image.copy()
            else:
                logger.error("هیچ تصویری ارائه نشده است!")
                return None, None

            # تغییر اندازه تصویر برای بهبود کارایی
            height, width = img.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                img = cv2.resize(
                    img, (int(width * scale), int(height * scale)))

            # تبدیل به RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # تشخیص نقاط کلیدی چهره
            with self.mp_face_mesh.FaceMesh(static_image_mode=True,
                                            max_num_faces=1,
                                            refine_landmarks=True,
                                            min_detection_confidence=0.5) as face_mesh:
                results = face_mesh.process(img_rgb)

                if not results.multi_face_landmarks:
                    logger.warning("هیچ چهره‌ای در تصویر یافت نشد")
                    return None, None

                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = img.shape

                # تبدیل نقاط کلیدی به مختصات پیکسل
                landmarks_px = []
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks_px.append((x, y))

                # استخراج نقاط مهم
                face_landmarks = {}
                for idx, landmark in enumerate(landmarks_px):
                    face_landmarks[idx] = landmark

                # محاسبه نسبت‌های مهم
                # 1. عرض پیشانی
                forehead_width = self._calculate_distance(
                    face_landmarks[10], face_landmarks[338])

                # 2. عرض گونه‌ها
                left_cheek = np.mean([face_landmarks[i]
                                     for i in self.LEFT_CHEEK], axis=0)
                right_cheek = np.mean([face_landmarks[i]
                                      for i in self.RIGHT_CHEEK], axis=0)
                cheekbone_width = self._calculate_distance(
                    left_cheek, right_cheek)

                # 3. عرض فک
                jaw_width = self._calculate_distance(
                    face_landmarks[58], face_landmarks[288])

                # 4. طول صورت
                face_top = np.mean([face_landmarks[i]
                                   for i in self.FACE_TOP], axis=0)
                face_bottom = face_landmarks[152]  # چانه
                face_length = self._calculate_distance(face_top, face_bottom)

                # 5. زاویه فک
                chin_to_left_jaw = np.array(
                    face_landmarks[152]) - np.array(face_landmarks[58])
                chin_to_right_jaw = np.array(
                    face_landmarks[152]) - np.array(face_landmarks[288])

                # محاسبه کسینوس زاویه بین دو بردار
                cos_angle = np.dot(chin_to_left_jaw, chin_to_right_jaw) / (
                    np.linalg.norm(chin_to_left_jaw) *
                    np.linalg.norm(chin_to_right_jaw)
                )
                jaw_angle = np.arccos(
                    np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

                # محاسبه نسبت‌های کلیدی
                width_to_length_ratio = cheekbone_width / face_length if face_length > 0 else 0
                cheekbone_to_jaw_ratio = cheekbone_width / jaw_width if jaw_width > 0 else 0
                forehead_to_cheekbone_ratio = forehead_width / \
                    cheekbone_width if cheekbone_width > 0 else 0

                # ویژگی‌های اضافی
                face_shape_ratio = width_to_length_ratio
                face_taper_ratio = jaw_width / forehead_width if forehead_width > 0 else 0

                # ساخت بردار ویژگی
                features = [
                    width_to_length_ratio,
                    cheekbone_to_jaw_ratio,
                    forehead_to_cheekbone_ratio,
                    jaw_angle,
                    face_shape_ratio,
                    face_taper_ratio
                ]

                # نمایش نقاط کلیدی روی
