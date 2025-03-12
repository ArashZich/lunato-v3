import os
import cv2
import numpy as np
import joblib
import logging
import mediapipe as mp
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

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
    'SQUARE': 'مربعی',
    'DIAMOND': 'لوزی',
    'TRIANGLE': 'مثلثی'
}

# نگاشت دسته‌ها به رنگ‌ها برای نمایش
SHAPE_COLORS = {
    'HEART': (255, 0, 0),    # قرمز
    'OBLONG': (0, 255, 0),   # سبز
    'OVAL': (0, 0, 255),     # آبی
    'ROUND': (255, 255, 0),  # زرد
    'SQUARE': (255, 0, 255),  # بنفش
    'DIAMOND': (0, 255, 255),  # فیروزه‌ای
    'TRIANGLE': (128, 128, 128)  # خاکستری
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
                f"Model and scaler successfully loaded from {model_path} and {scaler_path}")
        except Exception as e:
            logger.error(f"Error loading model or scaler: {str(e)}")
            raise

    def _calculate_distance(self, p1, p2):
        """محاسبه فاصله اقلیدسی بین دو نقطه"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

            # بهبود کنتراست تصویر
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                # تبدیل به لیست تا بتوانیم ویرایش کنیم
                lab_planes = list(cv2.split(lab))  # اصلاح این خط
                lab_planes[0] = clahe.apply(lab_planes[0])
                lab = cv2.merge(lab_planes)
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # کاهش نویز
                img = cv2.GaussianBlur(img, (3, 3), 0)
            except Exception as e:
                logger.warning(f"خطا در پیش‌پردازش تصویر: {str(e)}")

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

                # عرض گونه‌ها - با بررسی وجود نقاط
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
                    return None, None

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
                    return None, None
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
                jaw_angle = np.arccos(
                    np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

                # محاسبه نسبت‌های کلیدی
                width_to_length_ratio = cheekbone_width / face_length if face_length > 0 else 0
                cheekbone_to_jaw_ratio = cheekbone_width / jaw_width if jaw_width > 0 else 0
                forehead_to_cheekbone_ratio = forehead_width / \
                    cheekbone_width if cheekbone_width > 0 else 0

                # ویژگی‌های اضافی - اصلی
                face_shape_ratio = width_to_length_ratio
                face_taper_ratio = jaw_width / forehead_width if forehead_width > 0 else 0

                # ویژگی‌های اضافی - جدید
                # نسبت طول به عرض صورت
                face_height_to_width_ratio = face_length / \
                    cheekbone_width if cheekbone_width > 0 else 0
                # نسبت فک به پیشانی
                jaw_to_forehead_ratio = jaw_width / forehead_width if forehead_width > 0 else 0
                # برجستگی چانه
                chin_prominence = self._calculate_distance(
                    face_landmarks[152], face_landmarks[8]) / face_length if face_length > 0 else 0
                # فاصله چشم‌ها به عرض صورت
                eye_distance = self._calculate_distance(
                    face_landmarks[33], face_landmarks[263]) / cheekbone_width if cheekbone_width > 0 else 0
                # نسبت عرض پایین صورت به بالای صورت
                bottom_to_top_width_ratio = jaw_width / \
                    forehead_width if forehead_width > 0 else 0
                # زاویه نرمالایز شده
                normalized_jaw_angle = jaw_angle / 180.0

                # ساخت بردار ویژگی
                features = np.array([
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
                ]).reshape(1, -1)

                # نمایش نقاط کلیدی روی تصویر
                img_with_landmarks = self.visualize_landmarks(
                    img, face_landmarks)

                # برگرداندن ویژگی‌ها و تصویر با نقاط کلیدی
                return features, img_with_landmarks

        except Exception as e:
            logger.error(f"خطا در استخراج ویژگی‌ها: {str(e)}")
            return None, None

    def visualize_landmarks(self, image, landmarks):
        """نمایش نقاط کلیدی روی تصویر"""
        img_copy = image.copy()

        # نمایش نقاط اصلی بیضی صورت
        for idx in self.FACE_OVAL:
            cv2.circle(img_copy, landmarks[idx], 2, (0, 255, 0), -1)

        # نمایش نقاط چشم‌ها و دهان
        eye_indices = [33, 133, 362, 263]  # نقاط چشم‌ها
        mouth_indices = [0, 17, 61, 291]   # نقاط دهان

        for idx in eye_indices:
            cv2.circle(img_copy, landmarks[idx], 3, (255, 0, 0), -1)

        for idx in mouth_indices:
            cv2.circle(img_copy, landmarks[idx], 3, (0, 0, 255), -1)

        return img_copy

    def predict_face_shape(self, features):
        """پیش‌بینی شکل چهره با استفاده از بردار ویژگی"""
        if features is None:
            return None, None

        # مقیاس‌دهی ویژگی‌ها
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        # پیش‌بینی شکل چهره
        predicted_shape = self.model.predict(features_scaled)[0]

        # پیش‌بینی احتمالات
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        max_proba_idx = np.argmax(prediction_proba)
        confidence = prediction_proba[max_proba_idx] * 100

        return predicted_shape, confidence

    def test_single_image(self, image_path, output_dir=OUTPUT_DIR):
        """تست یک تصویر و نمایش نتیجه"""
        # استخراج ویژگی‌ها
        features, img_with_landmarks = self.extract_features_from_image(
            image_path)

        if features is None:
            logger.error(
                f"نمی‌توان ویژگی‌ها را از تصویر استخراج کرد: {image_path}")
            return

        # پیش‌بینی شکل چهره
        predicted_shape, confidence = self.predict_face_shape(features)

        if predicted_shape is None:
            logger.error(f"خطا در پیش‌بینی شکل چهره: {image_path}")
            return

        # ایجاد پوشه خروجی
        os.makedirs(output_dir, exist_ok=True)

        # نمایش نتیجه
        result_img = cv2.imread(image_path)

        # افزودن متن پیش‌بینی
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Shape: {predicted_shape} ({SHAPE_NAMES.get(predicted_shape, '')})"
        text_confidence = f"Confidence: {confidence:.2f}%"

        cv2.putText(result_img, text, (10, 30), font, 1,
                    SHAPE_COLORS.get(predicted_shape, (255, 255, 255)), 2)
        cv2.putText(result_img, text_confidence,
                    (10, 70), font, 1, (0, 255, 0), 2)

        # افزودن متن ویژگی‌ها
        feature_names = [
            "Width/Length", "Cheekbone/Jaw", "Forehead/Cheekbone",
            "Normalized Jaw Angle", "Face Shape Ratio", "Face Taper",
            "Height/Width Ratio", "Jaw/Forehead", "Chin Prominence",
            "Eye Distance", "Bottom/Top Width Ratio"
        ]

        y_pos = 110
        # فقط 6 ویژگی اول را نمایش بده
        for i, feature in enumerate(features[0][:6]):
            feature_text = f"{feature_names[i]}: {feature:.2f}"
            cv2.putText(result_img, feature_text,
                        (10, y_pos), font, 0.5, (0, 0, 0), 1)
            y_pos += 30

        # ذخیره تصویر نتیجه
        filename = os.path.basename(image_path)
        output_path = os.path.join(
            output_dir, f"result_{predicted_shape}_{filename}")
        cv2.imwrite(output_path, result_img)

        # ذخیره تصویر با نقاط کلیدی
        if img_with_landmarks is not None:
            landmarks_path = os.path.join(
                output_dir, f"landmarks_{predicted_shape}_{filename}")
            cv2.imwrite(landmarks_path, img_with_landmarks)

        logger.info(
            f"نتیجه تست {filename}: شکل چهره {predicted_shape} ({SHAPE_NAMES.get(predicted_shape, '')}) با اطمینان {confidence:.2f}%")
        logger.info(
            f"تصاویر نتیجه در {output_path} و {landmarks_path} ذخیره شدند")

        return predicted_shape, confidence

    def test_batch(self, test_dir, output_dir=OUTPUT_DIR, num_samples=10):
        """تست مجموعه‌ای از تصاویر از هر دسته"""
        os.makedirs(output_dir, exist_ok=True)

        results = {}
        confusion_matrix = {}

        # تست هر دسته
        for shape_dir in os.listdir(test_dir):
            shape_path = os.path.join(test_dir, shape_dir)

            # تبدیل نام پوشه به کلاس استاندارد
            true_shape = shape_dir.upper()

            if not os.path.isdir(shape_path):
                continue

            if true_shape not in confusion_matrix:
                confusion_matrix[true_shape] = {}

            # لیست فایل‌های تصویر
            image_files = [f for f in os.listdir(
                shape_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # انتخاب تعدادی نمونه تصادفی
            if len(image_files) > num_samples:
                selected_files = random.sample(image_files, num_samples)
            else:
                selected_files = image_files

            shape_results = []

            # تست هر تصویر
            logger.info(
                f"Testing {len(selected_files)} images of type {true_shape}...")

            for image_file in tqdm(selected_files, desc=f"Testing {true_shape}"):
                image_path = os.path.join(shape_path, image_file)

                # استخراج ویژگی‌ها
                features, _ = self.extract_features_from_image(image_path)

                if features is None:
                    continue

                # پیش‌بینی شکل چهره
                predicted_shape, confidence = self.predict_face_shape(features)

                if predicted_shape is None:
                    continue

                # افزودن به نتایج
                shape_results.append({
                    "file": image_file,
                    "predicted": predicted_shape,
                    "confidence": confidence,
                    "features": features.tolist()
                })

                # افزودن به ماتریس درهم‌ریختگی
                if predicted_shape not in confusion_matrix[true_shape]:
                    confusion_matrix[true_shape][predicted_shape] = 0
                confusion_matrix[true_shape][predicted_shape] += 1

                # ذخیره تصویر نتیجه برای چند نمونه
                if len(shape_results) <= 3:  # فقط 3 تصویر نمونه را ذخیره می‌کنیم
                    self.test_single_image(image_path, output_dir)

            # محاسبه دقت برای این دسته
            if len(shape_results) > 0:
                correct = sum(
                    1 for r in shape_results if r["predicted"] == true_shape)
                accuracy = correct / len(shape_results) * 100

                results[true_shape] = {
                    "accuracy": accuracy,
                    "samples": len(shape_results),
                    "results": shape_results
                }

                logger.info(
                    f"Accuracy for {true_shape}: {accuracy:.2f}% ({correct}/{len(shape_results)})")
            else:
                logger.warning(f"No results found for {true_shape}")

        # ذخیره نتایج
        self.save_results(results, confusion_matrix, output_dir)

        return results, confusion_matrix

    def save_results(self, results, confusion_matrix, output_dir):
        """ذخیره نتایج و نمایش گرافیکی"""
        import json
        import pandas as pd
        import seaborn as sns

        # ذخیره نتایج به صورت JSON
        with open(os.path.join(output_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        # نمایش نتایج در کنسول
        logger.info("\n===== نتایج نهایی =====")
        for shape, result in results.items():
            logger.info(
                f"{shape} ({SHAPE_NAMES.get(shape, '')}): {result['accuracy']:.2f}% دقت")

        # ایجاد ماتریس درهم‌ریختگی
        all_shapes = sorted(list(set(list(confusion_matrix.keys()) +
                                     [shape for shapes in confusion_matrix.values() for shape in shapes.keys()])))

        cm_data = []
        for true_shape in all_shapes:
            row = []
            for pred_shape in all_shapes:
                if true_shape in confusion_matrix and pred_shape in confusion_matrix[true_shape]:
                    row.append(confusion_matrix[true_shape][pred_shape])
                else:
                    row.append(0)
            cm_data.append(row)

        cm_df = pd.DataFrame(cm_data, index=all_shapes, columns=all_shapes)

        # نمایش ماتریس درهم‌ریختگی
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Face Shape Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

        # نمایش دقت برای هر دسته
        accuracies = [results[shape]['accuracy']
                      for shape in all_shapes if shape in results]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(all_shapes)),
                       [results.get(shape, {'accuracy': 0})['accuracy']
                        for shape in all_shapes])

        # تبدیل رنگ‌های BGR به RGB برای matplotlib
        for i, shape in enumerate(all_shapes):
            if shape in SHAPE_COLORS:
                bgr_color = SHAPE_COLORS[shape]
                # تبدیل از BGR به RGB و نرمال‌سازی به محدوده 0-1
                rgb_color = (bgr_color[2]/255,
                             bgr_color[1]/255, bgr_color[0]/255)
                bars[i].set_color(rgb_color)

        plt.xticks(range(len(all_shapes)), [
                   f"{shape}\n({SHAPE_NAMES.get(shape, '')})" for shape in all_shapes], rotation=45)
        plt.ylabel('Accuracy (%)')
        plt.title('Detection Accuracy for Each Face Shape')
        plt.ylim(0, 100)

        # افزودن مقدار دقت روی نمودار
        for i, acc in enumerate([results.get(shape, {'accuracy': 0})['accuracy'] for shape in all_shapes]):
            plt.text(i, acc + 2, f"{acc:.1f}%", ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_shape.png'))

        logger.info(f"نتایج و نمودارها در {output_dir} ذخیره شدند")


def main():
    # پردازش آرگومان‌های خط فرمان
    parser = argparse.ArgumentParser(
        description='Face Shape Classification Model Test')
    parser.add_argument('--test_dir', type=str, default='testing_set',
                        help='Path to test images directory (default: testing_set)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Path to output directory (default: test_results)')
    parser.add_argument('--single_image', type=str, default=None,
                        help='Path to a single test image')
    parser.add_argument('--model_path', type=str, default='data/face_shape_model.pkl',
                        help='Path to model file (default: data/face_shape_model.pkl)')
    parser.add_argument('--scaler_path', type=str, default='data/face_shape_model_scaler.pkl',
                        help='Path to scaler file (default: data/face_shape_model_scaler.pkl)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples per class (default: 10)')

    args = parser.parse_args()

    # راه‌اندازی تستر
    try:
        tester = FaceShapeTester(args.model_path, args.scaler_path)

        # تست یک تصویر خاص
        if args.single_image:
            if os.path.exists(args.single_image):
                tester.test_single_image(args.single_image, args.output_dir)
            else:
                logger.error(f"Test image not found: {args.single_image}")

        # تست دسته‌ای
        else:
            if os.path.exists(args.test_dir):
                results, confusion_matrix = tester.test_batch(
                    args.test_dir, args.output_dir, args.num_samples)
            else:
                logger.error(f"Test directory not found: {args.test_dir}")

    except Exception as e:
        logger.error(f"Error in running test: {str(e)}")


if __name__ == "__main__":
    main()
