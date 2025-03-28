# مستندات سیستم تشخیص چهره و پیشنهاد فریم عینک

## معرفی

این سیستم یک API هوشمند برای تحلیل تصویر چهره کاربران و پیشنهاد فریم عینک متناسب با شکل چهره آن‌ها است. سیستم با استفاده از الگوریتم‌های بینایی ماشین و یادگیری ماشین، شکل چهره را تشخیص داده و فریم‌های مناسب را از فروشگاه WooCommerce استخراج می‌کند.

## مراحل کاری پروژه

### 1. مراحل پردازش اصلی:

#### 1.1. دریافت درخواست و تصویر کاربر
- کاربر تصویر چهره خود را به API ارسال می‌کند
- سیستم می‌تواند به دو روش (همزمان یا غیرهمزمان) پردازش را انجام دهد
- اطلاعات مرورگر و دستگاه کاربر نیز برای تحلیل‌های آماری ذخیره می‌شود

#### 1.2. تشخیص چهره
- استفاده از OpenCV و مدل Haar Cascade برای تشخیص چهره در تصویر
- استخراج مختصات چهره (x, y, عرض، ارتفاع)
- اگر چندین چهره تشخیص داده شود، بزرگترین چهره انتخاب می‌شود
- برای دقت بیشتر، حاشیه‌ای به اندازه ۲۰٪ به اطراف چهره اضافه می‌شود

#### 1.3. تحلیل شکل چهره
- رویکرد ترکیبی شامل:
  * تلاش برای استفاده از مدل یادگیری ماشین (scikit-learn)
  * پشتیبانی با روش هندسی در صورت عدم موفقیت مدل ML
- محاسبه نسبت‌های مهم چهره:
  * نسبت عرض به ارتفاع چهره
  * نسبت عرض گونه‌ها به عرض فک
  * نسبت عرض پیشانی به عرض گونه‌ها
  * زاویه فک
- تعیین شکل چهره (بیضی، گرد، مربعی، قلبی، لوزی، مثلثی، کشیده)
- محاسبه میزان اطمینان تشخیص (درصد)

#### 1.4. پیشنهاد فریم مناسب
- تعیین انواع فریم مناسب براساس شکل چهره (از فایل داده face_shape_frames.json)
- دریافت محصولات از WooCommerce API
- فیلتر کردن فریم‌های عینک از میان همه محصولات
- اعمال فیلترهای قیمت در صورت درخواست کاربر
- محاسبه امتیاز تطابق هر فریم با شکل چهره
- مرتب‌سازی و محدود کردن نتایج براساس امتیاز تطابق و تعداد درخواستی

#### 1.5. ذخیره نتایج و پاسخ به کاربر
- ذخیره نتایج تحلیل و پیشنهادات در MongoDB
- ارسال پاسخ نهایی به کاربر شامل:
  * شکل چهره تشخیص داده شده
  * میزان اطمینان
  * توضیحات شکل چهره
  * توصیه‌های مناسب
  * فریم‌های پیشنهادی با امتیاز تطابق، قیمت و تصاویر

### 2. انواع روش‌های پردازش

#### 2.1. پردازش همزمان (Synchronous)
- کاربر منتظر می‌ماند تا تمام مراحل انجام و نتیجه ارسال شود
- تمام مراحل در یک درخواست HTTP انجام می‌شود
- مناسب برای کاربردهای با حجم کم و نیاز به پاسخ سریع

#### 2.2. پردازش غیرهمزمان (Asynchronous با Celery)
- کاربر یک شناسه وظیفه (task_id) دریافت می‌کند
- پردازش در پس‌زمینه توسط کارگران Celery انجام می‌شود
- سه نوع کارگر متفاوت:
  * `worker_face_detection`: تشخیص چهره
  * `worker_face_analysis`: تحلیل شکل چهره
  * `worker_frame_matching`: پیشنهاد فریم
- کاربر با استفاده از task_id می‌تواند وضعیت پردازش را بررسی کند
- مناسب برای پردازش حجم بالا و کاربران همزمان زیاد

### 3. نقاط ورودی API

#### 3.1. آنالیز چهره و پیشنهاد فریم
- **آدرس**: `/api/v1/analyze`
- **متد**: POST
- **نوع درخواست**: multipart/form-data
- **پارامترها**:
  * `file`: تصویر چهره
  * `include_frames`: آیا فریم‌های پیشنهادی در پاسخ گنجانده شود (boolean)
  * `min_price`: حداقل قیمت (اختیاری)
  * `max_price`: حداکثر قیمت (اختیاری)
  * `limit`: حداکثر تعداد فریم‌های پیشنهادی
  * `async_process`: آیا پردازش به صورت غیرهمزمان انجام شود (boolean)

#### 3.2. بررسی وضعیت پردازش غیرهمزمان
- **آدرس**: `/api/v1/analyze/{task_id}`
- **متد**: GET
- **پارامترها**:
  * `task_id`: شناسه وظیفه

#### 3.3. پیشنهاد فریم مستقیم
- **آدرس**: `/api/v1/face-shapes/{face_shape}/frames`
- **متد**: GET
- **پارامترها**:
  * `face_shape`: شکل چهره (OVAL, ROUND, SQUARE, HEART, DIAMOND, TRIANGLE, OBLONG)
  * `min_price`: حداقل قیمت (اختیاری)
  * `max_price`: حداکثر قیمت (اختیاری)
  * `limit`: حداکثر تعداد فریم‌های پیشنهادی

#### 3.4. گزارش‌های تحلیلی
- **آدرس**: `/api/v1/analytics/summary`
- **متد**: GET
- **خروجی**: خلاصه اطلاعات تحلیلی سیستم

#### 3.5. بررسی سلامت سیستم
- **آدرس**: `/api/v1/health`
- **متد**: GET
- **خروجی**: وضعیت سلامت سیستم، اتصال به دیتابیس و Celery

### 4. ذخیره‌سازی داده

#### 4.1. MongoDB
- **کالکشن‌ها**:
  * `requests`: ذخیره اطلاعات درخواست‌ها
  * `analysis_results`: ذخیره نتایج تحلیل چهره
  * `recommendations`: ذخیره پیشنهادات فریم
- **ایندکس‌های مهم**:
  * شناسه درخواست
  * شناسه کاربر
  * شکل چهره
  * زمان ایجاد رکورد

### 5. تکنولوژی‌های استفاده شده

- **Backend**: FastAPI (Python)
- **تشخیص چهره**: OpenCV (Haar Cascade)
- **تحلیل چهره**: scikit-learn + روش‌های هندسی
- **پردازش غیرهمزمان**: Celery + Redis
- **ذخیره‌سازی**: MongoDB
- **دریافت محصولات**: WooCommerce API
- **کانتینرسازی**: Docker و docker-compose

### 6. ساختار پروژه

```
eyeglass-recommendation/
├── app/                    # کد اصلی پروژه
│   ├── api/                # APIهای FastAPI
│   ├── core/               # عملیات اصلی (تشخیص و تحلیل چهره)
│   ├── db/                 # ارتباط با پایگاه داده
│   ├── models/             # مدل‌های داده
│   ├── services/           # سرویس‌های خارجی
│   └── utils/              # توابع کمکی
├── data/                   # داده‌های مورد نیاز مانند مدل‌ها
├── .env                    # فایل تنظیمات محیطی
├── docker-compose.yml      # پیکربندی Docker
├── Dockerfile              # تنظیمات ساخت تصویر Docker
└── requirements.txt        # وابستگی‌های پایتون
```

### 7. نصب و راه‌اندازی

#### 7.1. با Docker (روش پیشنهادی)

1. کلون کردن مخزن:
```bash
git clone https://github.com/yourusername/eyeglass-recommendation.git
cd eyeglass-recommendation
```

2. تنظیم فایل `.env`:
```bash
cp .env.example .env
# ویرایش فایل .env با تنظیمات مناسب
```

3. اجرا با docker-compose:
```bash
docker-compose up -d
```

4. بررسی لاگ‌ها:
```bash
docker-compose logs -f
```

#### 7.2. نصب دستی

1. کلون کردن مخزن:
```bash
git clone https://github.com/yourusername/eyeglass-recommendation.git
cd eyeglass-recommendation
```

2. ایجاد محیط مجازی:
```bash
python -m venv venv
source venv/bin/activate  # برای لینوکس/مک
venv\Scripts\activate  # برای ویندوز
```

3. نصب وابستگی‌ها:
```bash
pip install -r requirements.txt
```

4. تنظیم فایل `.env`:
```bash
cp .env.example .env
# ویرایش فایل .env با تنظیمات مناسب
```

5. اجرای API:
```bash
uvicorn app.main:app --reload
```

6. اجرای کارگران Celery (در ترمینال‌های جداگانه):
```bash
celery -A app.celery_app worker --loglevel=info --concurrency=1 --queues=face_detection
celery -A app.celery_app worker --loglevel=info --concurrency=1 --queues=face_analysis
celery -A app.celery_app worker --loglevel=info --concurrency=1 --queues=frame_matching
```

### 8. نکات مهم برای توسعه‌دهندگان

#### 8.1. مدل scikit-learn
- برای استفاده از مدل scikit-learn، باید فایل `data/face_shape_model.pkl` وجود داشته باشد
- این مدل با استفاده از SVM آموزش داده شده است
- در صورت عدم وجود مدل، سیستم به طور خودکار از روش هندسی استفاده می‌کند

#### 8.2. فایل داده شکل چهره
- فایل `data/face_shape_frames.json` شامل:
  * توضیحات شکل‌های چهره
  * توصیه‌های مناسب برای هر شکل
  * انواع فریم پیشنهادی برای هر شکل
  * نگاشت‌های کلمات کلیدی برای تشخیص نوع فریم

#### 8.3. کش محصولات WooCommerce
- برای کاهش تعداد درخواست‌ها به WooCommerce API، محصولات در حافظه کش می‌شوند
- مدت زمان اعتبار کش: 1 ساعت

#### 8.4. فایل‌های محیط (.env)
- کلیدهای WooCommerce API در فایل محیط ذخیره می‌شوند
- رمز عبور دیتابیس MongoDB نیز در این فایل قرار دارد
- در محیط توسعه می‌توان از `.env.example` استفاده کرد
- در محیط تولید، این مقادیر باید به صورت امن تنظیم شوند

## نتیجه‌گیری

سیستم تشخیص چهره و پیشنهاد فریم عینک یک راه‌حل کامل برای فروشگاه‌های آنلاین عینک است که به مشتریان کمک می‌کند فریم‌های مناسب با شکل چهره خود را انتخاب کنند. با استفاده از ترکیب روش‌های هندسی و یادگیری ماشین، دقت تشخیص بالایی ارائه می‌شود و با پردازش غیرهمزمان، سیستم قادر به مدیریت حجم بالای درخواست‌ها است.