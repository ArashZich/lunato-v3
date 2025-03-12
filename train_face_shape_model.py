import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

    def augment_image(self, image):
        """تولید تصاویر augmented با تغییرات مختلف"""
        augmented_images = []
        rows, cols = image.shape[:2]

        # 1. چرخش جزئی
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(image.copy(), M, (cols, rows))
            augmented_images.append(rotated)

        # 2. تغییر مقیاس
        try:
            scaled = cv2.resize(image.copy(), None, fx=1.1,
                                fy=1.1, interpolation=cv2.INTER_LINEAR)
            h, w = scaled.shape[:2]
            if w > cols and h > rows:  # اطمینان از اینکه تصویر بزرگتر شده است
                startx = w//2-(cols//2)
                starty = h//2-(rows//2)
                if startx >= 0 and starty >= 0 and startx+cols <= w and starty+rows <= h:
                    scaled = scaled[starty:starty+rows, startx:startx+cols]
                    augmented_images.append(scaled)
        except Exception as e:
            logger.warning(f"خطا در تغییر مقیاس تصویر: {str(e)}")

        # 3. تغییر کنتراست و روشنایی
        bright = cv2.convertScaleAbs(image.copy(), alpha=1.1, beta=10)
        dark = cv2.convertScaleAbs(image.copy(), alpha=0.9, beta=-10)
        augmented_images.append(bright)
        augmented_images.append(dark)

        # 4. افقی کردن تصویر
        flipped = cv2.flip(image.copy(), 1)
        augmented_images.append(flipped)

        return augmented_images

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

            # بهبود کنتراست تصویر
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                # تبدیل به لیست برای اطمینان از امکان تغییر
                lab_planes = list(cv2.split(lab))
                lab_planes[0] = clahe.apply(lab_planes[0])
                lab = cv2.merge(lab_planes)
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # کاهش نویز
                img = cv2.GaussianBlur(img, (3, 3), 0)
            except Exception as e:
                logger.warning(f"خطا در پیش‌پردازش تصویر: {str(e)}")
                # ادامه با تصویر اصلی در صورت بروز خطا

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

                # بررسی وجود نقاط کلیدی ضروری
                required_points = [10, 338, 58, 288, 152, 8, 33, 263]
                for point in required_points:
                    if point not in face_landmarks:
                        logger.warning(f"نقطه کلیدی {point} در تصویر یافت نشد")
                        return None

                # محاسبه نسبت‌های مهم
                # 1. عرض پیشانی
                forehead_width = self._calculate_distance(
                    face_landmarks[10], face_landmarks[338])

                # 2. عرض گونه‌ها - با بررسی وجود نقاط
                left_cheek_points = []
                for i in self.LEFT_CHEEK:
                    if i in face_landmarks:
                        left_cheek_points.append(face_landmarks[i])

                right_cheek_points = []
                for i in self.RIGHT_CHEEK:
                    if i in face_landmarks:
                        right_cheek_points.append(face_landmarks[i])

                if not left_cheek_points or not right_cheek_points:
                    logger.warning(
                        "نقاط کافی برای محاسبه عرض گونه‌ها وجود ندارد")
                    return None

                left_cheek = np.mean(left_cheek_points, axis=0)
                right_cheek = np.mean(right_cheek_points, axis=0)
                cheekbone_width = self._calculate_distance(
                    left_cheek, right_cheek)

                # 3. عرض فک
                jaw_width = self._calculate_distance(
                    face_landmarks[58], face_landmarks[288])

                # 4. طول صورت
                face_top_points = []
                for i in self.FACE_TOP:
                    if i in face_landmarks:
                        face_top_points.append(face_landmarks[i])

                if not face_top_points:
                    logger.warning("نقاط کافی برای محاسبه طول صورت وجود ندارد")
                    return None

                face_top = np.mean(face_top_points, axis=0)
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
                # محدود کردن به بازه معتبر arccos
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                jaw_angle = np.arccos(cos_angle) * 180 / np.pi

                # بررسی مقادیر صفر یا منفی برای جلوگیری از خطای تقسیم بر صفر
                if face_length <= 0 or cheekbone_width <= 0 or jaw_width <= 0 or forehead_width <= 0:
                    logger.warning(
                        "مقادیر صفر یا منفی برای محاسبات نسبت وجود دارد")
                    return None

                # محاسبه نسبت‌های کلیدی
                width_to_length_ratio = cheekbone_width / face_length
                cheekbone_to_jaw_ratio = cheekbone_width / jaw_width
                forehead_to_cheekbone_ratio = forehead_width / cheekbone_width

                # ویژگی‌های اضافی - اصلی
                face_shape_ratio = width_to_length_ratio
                face_taper_ratio = jaw_width / forehead_width

                # ویژگی‌های اضافی - جدید
                # نسبت طول به عرض صورت
                face_height_to_width_ratio = face_length / cheekbone_width
                # نسبت فک به پیشانی
                jaw_to_forehead_ratio = jaw_width / forehead_width
                # برجستگی چانه - با بررسی وجود نقطه 8
                if 8 in face_landmarks:
                    chin_prominence = self._calculate_distance(
                        face_landmarks[152], face_landmarks[8]) / face_length
                else:
                    chin_prominence = 0.5  # مقدار پیش‌فرض در صورت عدم وجود نقطه
                # فاصله چشم‌ها به عرض صورت
                eye_distance = self._calculate_distance(
                    face_landmarks[33], face_landmarks[263]) / cheekbone_width
                # نسبت عرض پایین صورت به بالای صورت
                bottom_to_top_width_ratio = jaw_width / forehead_width
                # زاویه نرمالایز شده
                normalized_jaw_angle = jaw_angle / 180.0

                # ساخت بردار ویژگی
                features = [
                    width_to_length_ratio,
                    cheekbone_to_jaw_ratio,
                    forehead_to_cheekbone_ratio,
                    normalized_jaw_angle,
                    face_shape_ratio,
                    face_taper_ratio,
                    face_height_to_width_ratio,  # ویژگی جدید
                    jaw_to_forehead_ratio,       # ویژگی جدید
                    chin_prominence,             # ویژگی جدید
                    eye_distance,                # ویژگی جدید
                    bottom_to_top_width_ratio    # ویژگی جدید
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
                shape_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
            logger.info(
                f"تعداد {len(image_files)} تصویر برای شکل {shape} یافت شد")

            # پردازش تصاویر آموزشی اصلی
            for image_file in tqdm(image_files, desc=f"پردازش {shape}"):
                image_path = os.path.join(shape_dir, image_file)
                features = self.extract_features_from_image(image_path)
                if features:
                    X_train.append(features)
                    y_train.append(SHAPE_MAPPING[shape])

            # افزایش داده برای کلاس‌هایی که دقت پایین دارند
            if shape in ['Heart', 'Oval', 'Round']:
                augmented_dir = os.path.join(shape_dir, 'augmented')
                os.makedirs(augmented_dir, exist_ok=True)

                # خالی کردن پوشه قبلی
                for old_file in os.listdir(augmented_dir):
                    file_path = os.path.join(augmented_dir, old_file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)

                # انتخاب تعدادی تصویر برای افزایش داده
                aug_candidates = image_files[:min(20, len(image_files))]
                for image_file in tqdm(aug_candidates, desc=f"افزایش داده برای {shape}"):
                    try:
                        image_path = os.path.join(shape_dir, image_file)
                        img = cv2.imread(image_path)
                        if img is None:
                            continue

                        augmented_images = self.augment_image(img)
                        for i, aug_img in enumerate(augmented_images):
                            aug_filename = f"aug_{i}_{image_file}"
                            aug_path = os.path.join(
                                augmented_dir, aug_filename)
                            cv2.imwrite(aug_path, aug_img)
                    except Exception as e:
                        logger.warning(
                            f"خطا در افزایش داده برای {image_file}: {str(e)}")
                        continue

                # اضافه کردن تصاویر افزایش یافته به لیست
                aug_files = [f for f in os.listdir(augmented_dir) if f.endswith(
                    '.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
                logger.info(
                    f"تعداد {len(aug_files)} تصویر افزایش داده شده برای {shape}")

                # پردازش تصاویر افزایش یافته
                for aug_file in tqdm(aug_files, desc=f"پردازش افزایش داده‌های {shape}"):
                    aug_path = os.path.join(augmented_dir, aug_file)
                    features = self.extract_features_from_image(aug_path)
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
                shape_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
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

        # آموزش مدل SVM با جستجوی پارامترهای بهینه
        param_grid = {
            'C': [1, 5, 10, 50],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf'],
            'class_weight': ['balanced', None]
        }

        # جستجوی پارامترهای بهینه
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            verbose=1
        )

        # آموزش مدل با GridSearch
        logger.info("جستجوی پارامترهای بهینه...")
        grid_search.fit(X_train_scaled, y_train)

        # استفاده از بهترین مدل
        svm_model = grid_search.best_estimator_
        logger.info(f"بهترین پارامترها: {grid_search.best_params_}")

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
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        plt.close()

        # بررسی اهمیت ویژگی‌ها (برای SVM)
        if hasattr(model, 'coef_'):
            plt.figure(figsize=(10, 6))
            feature_names = [
                'Width/Length Ratio',
                'Cheekbone/Jaw Ratio',
                'Forehead/Cheekbone Ratio',
                'Normalized Jaw Angle',
                'Face Shape Ratio',
                'Face Taper Ratio',
                'Height/Width Ratio',
                'Jaw/Forehead Ratio',
                'Chin Prominence',
                'Eye Distance',
                'Bottom/Top Width Ratio'
            ]
            plt.barh(feature_names, np.abs(model.coef_[0]))
            plt.title('Feature Importance')
            plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
            plt.close()

        # تحلیل توزیع ویژگی‌ها به تفکیک شکل چهره
        X_test_df = pd.DataFrame(X_test, columns=[
            'width_to_length_ratio',
            'cheekbone_to_jaw_ratio',
            'forehead_to_cheekbone_ratio',
            'normalized_jaw_angle',
            'face_shape_ratio',
            'face_taper_ratio',
            'face_height_to_width_ratio',
            'jaw_to_forehead_ratio',
            'chin_prominence',
            'eye_distance',
            'bottom_to_top_width_ratio'
        ])
        X_test_df['face_shape'] = y_test

        # نمایش توزیع برای ویژگی‌های مهم
        important_features = [
            'width_to_length_ratio',
            'cheekbone_to_jaw_ratio',
            'forehead_to_cheekbone_ratio',
            'normalized_jaw_angle',
            'face_height_to_width_ratio',
            'chin_prominence'
        ]

        for feature in important_features:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='face_shape', y=feature, data=X_test_df)
            plt.title(f'Distribution of {feature} by Face Shape')
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
