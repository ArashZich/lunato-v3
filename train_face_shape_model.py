import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import mediapipe as mp
import logging
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# تنظیم لاگر
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# مسیرهای دیتاست
DATASET_ROOT = '.'  # مسیر ریشه پروژه
TRAINING_DIR = os.path.join(DATASET_ROOT, 'training_set')
TESTING_DIR = os.path.join(DATASET_ROOT, 'testing_set')
OUTPUT_DIR = os.path.join(DATASET_ROOT, 'data')

# اطمینان از وجود پوشه خروجی
os.makedirs(OUTPUT_DIR, exist_ok=True)

# دسته‌های شکل صورت
FACE_SHAPES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# نگاشت دسته‌ها به شناسه‌های استفاده شده در پروژه
SHAPE_MAPPING = {
    'Heart': 'HEART',
    'Oblong': 'OBLONG',
    'Oval': 'OVAL',
    'Round': 'ROUND',
    'Square': 'SQUARE'
}


class FaceShapeTrainer:
    def __init__(self):
        """مقداردهی اولیه کلاس آموزش مدل تشخیص شکل چهره"""
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
        self.JAWL_TO_CHIN = [58, 172, 136, 150, 176, 152, 400]
        self.JAWL_TO_CHIN_RIGHT = [332, 284, 251, 389, 356, 454, 323]

    def extract_features_from_image(self, image_path):
        """استخراج ویژگی‌های چهره از تصویر با استفاده از MediaPipe"""
        try:
            # خواندن تصویر
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"نمی‌توان تصویر را بخوانیم: {image_path}")
                return None

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
                    logger.warning(
                        f"هیچ چهره‌ای در تصویر یافت نشد: {image_path}")
                    return None

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

                return features

        except Exception as e:
            logger.error(
                f"خطا در استخراج ویژگی از تصویر {image_path}: {str(e)}")
            return None

    def _calculate_distance(self, p1, p2):
        """محاسبه فاصله اقلیدسی بین دو نقطه"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def prepare_dataset(self):
        """آماده‌سازی دیتاست برای آموزش"""
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # خواندن و استخراج ویژگی‌ها از تصاویر آموزشی
        logger.info("استخراج ویژگی‌ها از تصاویر آموزشی...")
        for shape in FACE_SHAPES:
            shape_dir = os.path.join(TRAINING_DIR, shape)

            if not os.path.exists(shape_dir):
                logger.warning(f"پوشه {shape_dir} یافت نشد!")
                continue

            image_files = [f for f in os.listdir(
                shape_dir) if f.endswith('.jpg')]
            logger.info(
                f"تعداد {len(image_files)} تصویر برای شکل {shape} یافت شد")

            for image_file in tqdm(image_files, desc=f"پردازش {shape}"):
                image_path = os.path.join(shape_dir, image_file)
                features = self.extract_features_from_image(image_path)

                if features:
                    X_train.append(features)
                    y_train.append(SHAPE_MAPPING[shape])

        # خواندن و استخراج ویژگی‌ها از تصاویر تست
        logger.info("استخراج ویژگی‌ها از تصاویر تست...")
        for shape in FACE_SHAPES:
            shape_dir = os.path.join(TESTING_DIR, shape)

            if not os.path.exists(shape_dir):
                logger.warning(f"پوشه {shape_dir} یافت نشد!")
                continue

            image_files = [f for f in os.listdir(
                shape_dir) if f.endswith('.jpg')]
            logger.info(
                f"تعداد {len(image_files)} تصویر تست برای شکل {shape} یافت شد")

            for image_file in tqdm(image_files, desc=f"پردازش تست {shape}"):
                image_path = os.path.join(shape_dir, image_file)
                features = self.extract_features_from_image(image_path)

                if features:
                    X_test.append(features)
                    y_test.append(SHAPE_MAPPING[shape])

        logger.info(f"تعداد کل داده‌های آموزش: {len(X_train)}")
        logger.info(f"تعداد کل داده‌های تست: {len(X_test)}")

        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    def train_model(self, X_train, y_train, X_test, y_test):
        """آموزش مدل SVM برای تشخیص شکل چهره"""
        logger.info("شروع آموزش مدل...")

        # مقیاس‌دهی ویژگی‌ها
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # آموزش مدل SVM
        svm_model = SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=42)

        # اعتبارسنجی متقابل
        cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
        logger.info(f"نتایج اعتبارسنجی متقابل: {cv_scores}")
        logger.info(f"میانگین دقت اعتبارسنجی متقابل: {np.mean(cv_scores):.4f}")

        # آموزش نهایی
        svm_model.fit(X_train_scaled, y_train)

        # ارزیابی روی مجموعه تست
        test_accuracy = svm_model.score(X_test_scaled, y_test)
        logger.info(f"دقت روی مجموعه تست: {test_accuracy:.4f}")

        # گزارش‌های دقیق‌تر
        y_pred = svm_model.predict(X_test_scaled)

        logger.info("\nگزارش طبقه‌بندی:")
        print(classification_report(y_test, y_pred))

        # نمایش ماتریس درهم‌ریختگی
        cm = confusion_matrix(y_test, y_pred)

        # ذخیره مدل و اسکیلر
        joblib.dump(svm_model, os.path.join(
            OUTPUT_DIR, 'face_shape_model.pkl'))
        joblib.dump(scaler, os.path.join(
            OUTPUT_DIR, 'face_shape_model_scaler.pkl'))

        logger.info(f"مدل و اسکیلر با موفقیت در مسیر {OUTPUT_DIR} ذخیره شدند")

        return svm_model, scaler, test_accuracy, cm

    def visualize_results(self, X_test, y_test, model, scaler, confusion_matrix):
        """نمایش نتایج و ویژوالایز مدل"""
        # نمایش ماتریس درهم‌ریختگی
        plt.figure(figsize=(10, 8))
        classes = list(SHAPE_MAPPING.values())
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('ماتریس درهم‌ریختگی')
        plt.ylabel('برچسب واقعی')
        plt.xlabel('برچسب پیش‌بینی شده')
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        plt.close()

        # بررسی اهمیت ویژگی‌ها (برای SVM)
        if hasattr(model, 'coef_'):
            plt.figure(figsize=(10, 6))
            feature_names = [
                'نسبت عرض به طول',
                'نسبت گونه به فک',
                'نسبت پیشانی به گونه',
                'زاویه فک',
                'نسبت شکل صورت',
                'نسبت باریک‌شدگی'
            ]
            plt.barh(feature_names, np.abs(model.coef_[0]))
            plt.title('اهمیت ویژگی‌ها')
            plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
            plt.close()

        # تحلیل توزیع ویژگی‌ها به تفکیک شکل چهره
        X_test_df = pd.DataFrame(X_test, columns=[
            'width_to_length_ratio',
            'cheekbone_to_jaw_ratio',
            'forehead_to_cheekbone_ratio',
            'jaw_angle',
            'face_shape_ratio',
            'face_taper_ratio'
        ])
        X_test_df['face_shape'] = y_test

        for feature in X_test_df.columns[:-1]:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='face_shape', y=feature, data=X_test_df)
            plt.title(f'توزیع {feature} به تفکیک شکل چهره')
            plt.savefig(os.path.join(
                OUTPUT_DIR, f'{feature}_distribution.png'))
            plt.close()


def main():
    """تابع اصلی برای اجرای فرآیند آموزش مدل"""
    logger.info("شروع آموزش مدل تشخیص شکل چهره...")

    # بررسی وجود پوشه‌های دیتاست
    if not os.path.exists(TRAINING_DIR) or not os.path.exists(TESTING_DIR):
        logger.error(
            "پوشه‌های دیتاست یافت نشدند. لطفاً اطمینان حاصل کنید که دیتاست را دانلود کرده‌اید.")
        logger.info(f"مسیرهای مورد انتظار: {TRAINING_DIR}, {TESTING_DIR}")
        return

    # ایجاد نمونه از کلاس آموزش
    trainer = FaceShapeTrainer()

    # آماده‌سازی دیتاست
    X_train, y_train, X_test, y_test = trainer.prepare_dataset()

    if len(X_train) == 0 or len(X_test) == 0:
        logger.error(
            "استخراج ویژگی‌ها ناموفق بود. هیچ داده‌ای برای آموزش یافت نشد.")
        return

    # آموزش مدل
    model, scaler, accuracy, confusion_mat = trainer.train_model(
        X_train, y_train, X_test, y_test)

    # نمایش نتایج
    trainer.visualize_results(X_test, y_test, model, scaler, confusion_mat)

    logger.info("فرآیند آموزش مدل با موفقیت به پایان رسید!")
    logger.info(
        f"مدل نهایی در مسیر {os.path.join(OUTPUT_DIR, 'face_shape_model.pkl')} ذخیره شد")
    logger.info(
        f"اسکیلر در مسیر {os.path.join(OUTPUT_DIR, 'face_shape_model_scaler.pkl')} ذخیره شد")


if __name__ == "__main__":
    main()
