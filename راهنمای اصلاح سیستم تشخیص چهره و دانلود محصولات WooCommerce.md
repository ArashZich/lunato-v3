# راهنمای اصلاح سیستم تشخیص چهره و دانلود محصولات WooCommerce

در این راهنما، تغییرات مورد نیاز برای حل دو مشکل اصلی پروژه شرح داده شده است:
1. بهبود لاگ‌های مربوط به دانلود محصولات WooCommerce
2. اصلاح سیستم تشخیص شکل چهره

## 1. بهبود لاگ‌های دانلود محصولات WooCommerce

### فایل: `app/services/woocommerce.py`

#### تغییرات در تابع `initialize_product_cache`
- افزودن لاگ هنگام عدم وجود کش در دیتابیس: 
```python
logger.info("کش محصولات در دیتابیس یافت نشد. در حال دانلود محصولات از WooCommerce...")
```
- افزودن لاگ هنگام منقضی شدن کش:
```python
logger.info("کش محصولات در دیتابیس منقضی شده است. نیاز به بروزرسانی دارد.")
```

#### تغییرات در تابع `refresh_product_cache`
- افزودن لاگ در شروع دانلود:
```python
logger.info("شروع فرآیند دانلود و بروزرسانی کش محصولات از WooCommerce API")
```
- افزودن لاگ پس از اتمام دانلود:
```python
logger.info(f"دانلود محصولات از WooCommerce API با موفقیت انجام شد. تعداد محصولات: {len(products)}")
```
- افزودن لاگ در شروع ذخیره در دیتابیس:
```python
logger.info("در حال ذخیره محصولات دانلود شده در دیتابیس...")
```

#### تغییرات در تابع `fetch_all_woocommerce_products`
- افزودن لاگ در شروع دانلود:
```python
logger.info("شروع دانلود محصولات از WooCommerce API...")
```
- افزودن لاگ برای هر صفحه از محصولات:
```python
logger.info(f"در حال دانلود صفحه {page} از محصولات...")
logger.info(f"دانلود صفحه {page} با {len(products)} محصول انجام شد")
```
- افزودن لاگ در پایان دانلود:
```python
logger.info(f"دانلود و پیش‌پردازش {len(all_products)} محصول از WooCommerce API کامل شد. تعداد فریم عینک: {eyeglass_count}")
```
- افزودن لاگ در صورت نیاز به تلاش مجدد:
```python
logger.info(f"تلاش مجدد برای دانلود صفحه {page} پس از 2 ثانیه...")
```

## 2. اصلاح سیستم تشخیص شکل چهره

### فایل: `app/core/face_analysis.py`

#### تغییرات در تابع `_determine_face_shape`
- افزودن لاگ برای نمایش نسبت‌های محاسبه شده:
```python
logger.info(f"نسبت‌های محاسبه شده برای تشخیص شکل چهره:")
logger.info(f"نسبت عرض به طول: {width_to_length:.2f}")
logger.info(f"نسبت گونه به فک: {cheekbone_to_jaw:.2f}")
logger.info(f"نسبت پیشانی به گونه: {forehead_to_cheekbone:.2f}")
logger.info(f"زاویه فک: {jaw_angle:.2f} درجه")
```
- بهبود شرط‌های تشخیص شکل چهره:
```python
# صورت گرد (ROUND)
# شرط اصلی: نسبت عرض به طول نزدیک به 1، زاویه فک بزرگ، نسبت گونه به فک نزدیک به 1
if width_to_length >= 0.85 and width_to_length <= 1.1 and jaw_angle >= 160 and 0.9 <= cheekbone_to_jaw <= 1.1:
    logger.info("تشخیص شکل چهره: گرد (ROUND)")
    return "ROUND"

# صورت مربعی (SQUARE)
# شرط اصلی: نسبت عرض به طول نزدیک به 1، زاویه فک کوچک‌تر، نسبت گونه به فک نزدیک به 1
if width_to_length >= 0.85 and width_to_length <= 1.1 and jaw_angle <= 150 and 0.9 <= cheekbone_to_jaw <= 1.1:
    logger.info("تشخیص شکل چهره: مربعی (SQUARE)")
    return "SQUARE"
```
- استفاده از لاگ برای ثبت نوع تشخیص:
```python
logger.info("تشخیص شکل چهره: بیضی (OVAL) - پیش‌فرض")
return "OVAL"
```

#### تغییرات در تابع `analyze_face_shape`
- افزودن لاگ بیشتر برای مراحل تحلیل:
```python
logger.info("شروع تحلیل شکل چهره با روش هندسی...")
logger.info("محاسبه نسبت‌های هندسی چهره...")
logger.info(f"نسبت‌های محاسبه شده: {shape_metrics}")
logger.info(f"تحلیل شکل چهره با روش هندسی انجام شد: {face_shape} با اطمینان {confidence:.1f}%")
```

#### تغییرات در تابع `generate_full_analysis`
- اولویت‌دهی به روش هندسی برای بهبود تشخیص:
```python
# تصمیم‌گیری نهایی - اولویت با روش هندسی
if geometric_result.get("success", False):
    result_face_shape = geometric_result.get("face_shape")
    result_confidence = geometric_result.get("confidence")
    logger.info(f"استفاده از نتایج تحلیل هندسی: {result_face_shape} با اطمینان {result_confidence:.1f}%")
elif ml_success:
    # استفاده از نتایج مدل ML در صورتی که تحلیل هندسی موفق نبوده است
    result_face_shape = face_shape
    result_confidence = confidence
    logger.info(f"استفاده از نتایج مدل ML: {result_face_shape} با اطمینان {result_confidence:.1f}%")
```

### فایل: `app/core/face_detection.py`

#### تغییرات در تابع `_generate_default_landmarks`
- بهبود توزیع نقاط کلیدی:
```python
# محاسبه نقاط با مقادیر نسبی بهتر برای تشخیص دقیق‌تر شکل چهره
forehead_y = y + h * 0.15  # خط پیشانی (15% از بالای چهره)
middle_y = y + h * 0.5     # خط میانی صورت
jawline_y = y + h * 0.8    # خط فک
chin_y = y + h * 0.95      # خط چانه

landmarks = np.array([
    [x + w * 0.2, forehead_y],         # 0: گوشه بالا چپ پیشانی
    [x + w * 0.5, y + h * 0.05],       # 1: وسط پیشانی (کمی بالاتر)
    [x + w * 0.8, forehead_y],         # 2: گوشه بالا راست پیشانی
    [x + w * 0.15, jawline_y],         # 3: گوشه چپ فک
    [x + w * 0.5, chin_y],             # 4: چانه
    [x + w * 0.85, jawline_y],         # 5: گوشه راست فک
    [x + w * 0.3, middle_y],           # 6: گونه چپ
    [x + w * 0.5, middle_y],           # 7: وسط صورت
    [x + w * 0.7, middle_y]            # 8: گونه راست
])
```

#### تغییرات در تابع `detect_face_landmarks`
- بهبود پیش‌پردازش تصویر برای تشخیص بهتر:
```python
# بهبود کنتراست تصویر برای تشخیص بهتر نقاط کلیدی
try:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
except Exception as e:
    logger.warning(f"خطا در بهبود کنتراست تصویر: {str(e)}")
    enhanced_gray = gray
```
- انتخاب هوشمند نقاط کلیدی از مدل‌های مختلف:
```python
# انتخاب نقاط کلیدی مورد نیاز برای تشخیص شکل چهره
if len(landmarks) >= 68:  # نقاط کلیدی استاندارد dlib
    selected_landmarks = np.array([
        landmarks[16],   # گوشه راست پیشانی
        landmarks[27],   # وسط پیشانی
        landmarks[0],    # گوشه چپ پیشانی
        landmarks[4],    # گوشه چپ فک
        landmarks[8],    # چانه
        landmarks[12],   # گوشه راست فک
        landmarks[1],    # گونه چپ
        landmarks[30],   # وسط صورت (بینی)
        landmarks[15]    # گونه راست
    ])
    return selected_landmarks
```
- افزودن لاگ‌های بیشتر برای اشکال‌زدایی:
```python
logger.info(f"تشخیص {len(landmarks)} نقطه کلیدی با استفاده از dlib")
```

#### تغییرات در تابع `get_face_image`
- اضافه کردن بررسی‌های بیشتر برای اطمینان از تشخیص صحیح چهره:
```python
# بررسی کیفیت و وضوح تصویر
height, width = image.shape[:2]
logger.info(f"ابعاد تصویر: {width}x{height}")

if width < 100 or height < 100:
    logger.warning(f"تصویر با ابعاد {width}x{height} کیفیت مناسبی برای تشخیص چهره ندارد")
    return False, {
        "success": False, 
        "message": "تصویر با کیفیت مناسبی برای تشخیص چهره ندارد (ابعاد خیلی کوچک)"
    }, None
```
- بررسی نسبت ابعاد چهره برای تشخیص بهتر:
```python
# بررسی نسبت ابعاد چهره برای اطمینان از تشخیص درست
aspect_ratio = w / h if h > 0 else 0
logger.info(f"نسبت ابعاد چهره تشخیص داده شده: {aspect_ratio:.2f}")

if aspect_ratio < 0.5 or aspect_ratio > 2.0:
    logger.warning(f"نسبت ابعاد چهره ({aspect_ratio:.2f}) خارج از محدوده منطقی است")
    return False, {
        "success": False, 
        "message": f"چهره تشخیص داده شده نسبت ابعاد غیرطبیعی دارد ({aspect_ratio:.2f})"
    }, None
```
- بررسی اندازه چهره نسبت به کل تصویر:
```python
# بررسی اندازه چهره در مقایسه با کل تصویر
face_area = w * h
image_area = width * height
face_percentage = (face_area / image_area) * 100
logger.info(f"درصد اشغال تصویر توسط چهره: {face_percentage:.2f}%")

if face_percentage < 5:
    logger.warning(f"چهره تشخیص داده شده خیلی کوچک است (فقط {face_percentage:.2f}% از تصویر)")
    return False, {
        "success": False, 
        "message": f"چهره تشخیص داده شده خیلی کوچک است"
    }, None
```

## 3. نکات اجرایی و پیشنهادات تکمیلی

### 1. اجرای تدریجی تغییرات

برای حل مشکلات، بهتر است تغییرات را به صورت تدریجی اعمال کنید:
1. ابتدا تغییرات مربوط به لاگ‌های WooCommerce را اعمال کنید
2. سپس تغییرات مربوط به تشخیص شکل چهره را پیاده‌سازی کنید
3. هر بخش را جداگانه تست کنید تا از درستی عملکرد آن مطمئن شوید

### 2. تست با نمونه‌های متنوع

برای اطمینان از درستی تشخیص شکل چهره، بهتر است:
- چهره‌هایی با شکل‌های مختلف (گرد، بیضی، مربعی و...) را تست کنید
- زوایای مختلف چهره را آزمایش کنید
- تصاویر با کیفیت و شرایط نوری متفاوت را بررسی کنید

### 3. پیشنهادات تکمیلی

- **افزودن پنل مدیریت**: ایجاد یک بخش برای دریافت گزارش عملکرد سیستم و مشاهده آماری که چه درصدی از تصاویر به هر شکل چهره تشخیص داده می‌شوند
- **بهبود پیش‌پردازش تصویر**: افزودن مراحل پیش‌پردازش بیشتر مانند بهبود کنتراست و کاهش نویز
- **مکانیزم بازخورد کاربر**: اضافه کردن امکان دریافت بازخورد از کاربر در مورد صحت تشخیص شکل چهره
- **به‌روزرسانی مدل ML**: با جمع‌آوری داده‌های بیشتر، مدل ML را بهبود دهید