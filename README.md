# سیستم تشخیص چهره و پیشنهاد فریم عینک

سیستم هوشمند تشخیص شکل چهره و پیشنهاد فریم عینک مناسب با استفاده از FastAPI و ML.

## معرفی

این پروژه یک API هوشمند برای تحلیل تصویر چهره کاربران و پیشنهاد فریم عینک متناسب با شکل چهره آن‌ها ارائه می‌دهد. این سیستم با استفاده از الگوریتم‌های بینایی ماشین و یادگیری ماشین، شکل چهره را تشخیص داده و فریم‌های مناسب را با توجه به ویژگی‌های چهره از فروشگاه WooCommerce استخراج می‌کند.

## قابلیت‌ها

- تشخیص چهره در تصویر با استفاده از OpenCV
- تحلیل شکل چهره (بیضی، گرد، مربعی، قلبی، لوزی، مثلثی، کشیده)
- پیشنهاد انواع فریم عینک مناسب با شکل چهره
- پیشنهاد محصولات واقعی از فروشگاه WooCommerce
- پردازش همزمان یا غیرهمزمان با استفاده از Celery
- ذخیره‌سازی نتایج تحلیل و پیشنهادها در MongoDB
- جمع‌آوری اطلاعات تحلیلی از درخواست‌ها و پیشنهادها

## پیش‌نیازها

- Python 3.8+
- MongoDB
- Redis
- WooCommerce (فروشگاه آنلاین با API فعال)

## نصب و راه‌اندازی

### نصب با Docker (پیشنهادی)

1. کلون کردن مخزن:
```bash
git clone https://github.com/yourusername/eyeglass-recommendation.git
cd eyeglass-recommendation
```

2. تنظیم فایل `.env` (یا استفاده از `.env.example`):
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

### نصب دستی

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

## ساختار پروژه

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
├── .env.example            # نمونه فایل تنظیمات
├── docker-compose.yml      # پیکربندی Docker
├── Dockerfile              # تنظیمات ساخت تصویر Docker
└── requirements.txt        # وابستگی‌های پایتون
```

## استفاده از API

### تحلیل چهره و پیشنهاد فریم

**درخواست:**
```http
POST /api/v1/analyze
Content-Type: multipart/form-data

file: <تصویر چهره>
include_frames: true
min_price: 1000000
max_price: 5000000
limit: 10
async_process: false
```

**پاسخ:**
```json
{
  "success": true,
  "message": "تحلیل چهره با موفقیت انجام شد",
  "face_shape": "OVAL",
  "confidence": 85.5,
  "description": "صورت بیضی متعادل‌ترین شکل صورت است. پهنای گونه‌ها با پهنای پیشانی و فک متناسب است.",
  "recommendation": "اکثر فریم‌ها برای این نوع صورت مناسب هستند، اما فریم‌های مستطیلی و مربعی بهترین گزینه هستند.",
  "recommended_frame_types": ["مستطیلی", "مربعی", "هشت‌ضلعی"],
  "recommended_frames": [
    {
      "id": 123,
      "name": "فریم طبی مدل مستطیلی کلاسیک",
      "permalink": "https://lunato.shop/product/classic-rectangular",
      "price": "2500000",
      "frame_type": "مستطیلی",
      "images": ["https://lunato.shop/wp-content/uploads/frame1.jpg"],
      "match_score": 92.5
    }
  ]
}
```

## مستندات API

برای مشاهده مستندات کامل API، پس از راه‌اندازی به آدرس زیر مراجعه کنید:

```
http://localhost:8000/docs
```

## مجوز

این پروژه تحت مجوز MIT منتشر شده است.

## همکاری

پیشنهادات، گزارش مشکلات و درخواست‌های Pull همواره مورد استقبال قرار می‌گیرد. لطفاً برای همکاری، ابتدا مسئله مورد نظر را در قسمت Issues مطرح کنید.