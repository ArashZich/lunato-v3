import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import asyncio
import threading
import time
import schedule
import aiohttp
import random

from app.config import settings
from app.core.face_shape_data import get_recommended_frame_types
from app.db.connection import get_database
from app.db.repository import save_woocommerce_cache, get_woocommerce_cache


# تنظیمات لاگر
logger = logging.getLogger(__name__)

# کش برای محصولات
product_cache = None
last_cache_update = None
refresh_lock = asyncio.Lock()
update_scheduler = None

# نگهداری وضعیت بروزرسانی
update_status = {
    "last_update": None,
    "in_progress": False,
    "total_products": 0,
    "last_error": None
}


async def initialize_product_cache():
    """
    راه‌اندازی اولیه کش محصولات در شروع برنامه
    """
    global product_cache, last_cache_update, update_status

    logger.info("شروع راه‌اندازی اولیه کش محصولات WooCommerce")

    # بررسی وجود کش در دیتابیس
    try:
        db = get_database()
        cache_record = await db.woocommerce_cache.find_one({"type": "products_cache_meta"})

        if cache_record and "total_products" in cache_record and cache_record["total_products"] > 0:
            # بررسی اعتبار کش
            cache_age = datetime.now(timezone.utc) - \
                cache_record['last_update']
            if cache_age > timedelta(hours=24):
                logger.info(
                    f"کش محصولات در دیتابیس منقضی شده است (آخرین بروزرسانی: {cache_record['last_update']}). نیاز به بروزرسانی دارد.")
            else:
                # دریافت چانک‌های کش
                total_chunks = cache_record.get("total_chunks", 0)
                chunks_data = []

                for i in range(total_chunks):
                    chunk = await db.woocommerce_cache.find_one({"type": f"products_cache_chunk_{i}"})
                    if chunk and "data" in chunk:
                        chunks_data.extend(chunk["data"])

                if chunks_data:
                    logger.info(
                        f"کش محصولات از دیتابیس بازیابی شد: {len(chunks_data)} محصول (آخرین بروزرسانی: {cache_record['last_update']})")
                    product_cache = chunks_data
                    last_cache_update = cache_record["last_update"]
                    update_status["last_update"] = cache_record["last_update"]
                    update_status["total_products"] = len(chunks_data)

                    # بررسی محصولات معتبر پس از بازیابی
                    valid_products = []
                    for product in product_cache:
                        # تغییر از بررسی قیمت به بررسی stock_status
                        if (product.get("stock_status") == "instock" and
                            "/product/" in product.get("permalink", "") and
                                "/?post_type=product&p=" not in product.get("permalink", "")):
                            valid_products.append(product)
                        else:
                            logger.debug(
                                f"محصول نامعتبر از کش حذف شد: ID {product.get('id')}")

                    # به‌روزرسانی کش با محصولات معتبر
                    if len(valid_products) < len(product_cache):
                        logger.info(
                            f"{len(product_cache) - len(valid_products)} محصول نامعتبر از کش حذف شد")
                        product_cache = valid_products
                        update_status["total_products"] = len(valid_products)

                    # فقط راه‌اندازی زمان‌بندی بدون دانلود اولیه
                    start_scheduled_updates()
                    return
                else:
                    logger.info(
                        "کش محصولات در دیتابیس خالی است. انجام دانلود اولیه محصولات از WooCommerce...")
        else:
            logger.info(
                "کش محصولات در دیتابیس یافت نشد یا خالی است. انجام دانلود اولیه محصولات از WooCommerce...")
    except Exception as e:
        logger.warning(f"خطا در بررسی کش دیتابیس: {str(e)}")
        logger.info("به دلیل خطا در بررسی کش، انجام دانلود اولیه محصولات...")

    # بروزرسانی اولیه فقط اگر کش وجود نداشته باشد یا خالی باشد
    await refresh_product_cache(force=True)

    # راه‌اندازی بروزرسانی خودکار هفتگی
    start_scheduled_updates()


def start_scheduled_updates():
    """
    راه‌اندازی بروزرسانی زمان‌بندی شده محصولات
    """
    global update_scheduler

    # اگر قبلاً راه‌اندازی شده، آن را متوقف کنیم
    if update_scheduler and update_scheduler.is_alive():
        return

    # تنظیم زمان‌بندی بروزرسانی (روزی یکبار در ساعت 3 صبح)
    def run_scheduler():
        schedule.every().monday.at("03:00").do(run_async_update)  # دوشنبه‌ها ساعت 3 صبح
        schedule.every().thursday.at("03:00").do(
            run_async_update)  # پنجشنبه‌ها ساعت 3 صبح

        while True:
            schedule.run_pending()
            # بررسی هر 15 دقیقه
            time.sleep(900)

    def run_async_update():
        logger.info("اجرای زمان‌بندی شده بروزرسانی کش محصولات WooCommerce")
        asyncio.run(refresh_product_cache(force=True))

    # اجرای زمان‌بندی در یک ترد جداگانه
    update_scheduler = threading.Thread(target=run_scheduler, daemon=True)
    update_scheduler.start()

    logger.info(
        "زمان‌بندی بروزرسانی خودکار محصولات WooCommerce فعال شد (دوشنبه ها و پنجشنبه ها)")


async def refresh_product_cache(force=False) -> bool:
    """
    بروزرسانی کش محصولات WooCommerce.

    Args:
        force: اجبار به بروزرسانی حتی اگر کش معتبر باشد

    Returns:
        bool: نتیجه بروزرسانی
    """
    global product_cache, last_cache_update, update_status

    # بررسی وضعیت فعلی کش
    if not force and product_cache is not None and len(product_cache) > 0:
        logger.info(
            "کش محصولات WooCommerce از قبل معتبر است و نیاز به بروزرسانی ندارد")
        return True

    # از قفل برای جلوگیری از بروزرسانی‌های همزمان استفاده می‌کنیم
    async with refresh_lock:
        # بررسی مجدد پس از گرفتن قفل
        if not force and product_cache is not None and len(product_cache) > 0:
            logger.info("کش محصولات از قبل معتبر است (بررسی مجدد پس از قفل)")
            return True

        # تنظیم وضعیت بروزرسانی
        update_status["in_progress"] = True
        update_status["last_error"] = None

        try:
            logger.info(
                "شروع فرآیند دانلود و بروزرسانی کش محصولات از WooCommerce API")

            # دریافت محصولات از API
            products = await fetch_all_woocommerce_products()

            if not products:
                logger.error("خطا در دریافت محصولات از WooCommerce API")
                update_status["in_progress"] = False
                update_status["last_error"] = "خطا در دریافت محصولات از API"
                return False

            # بروزرسانی کش
            product_cache = products
            last_cache_update = datetime.now(timezone.utc)
            update_status["last_update"] = last_cache_update
            update_status["total_products"] = len(products)

            logger.info(
                f"دانلود محصولات از WooCommerce API با موفقیت انجام شد. تعداد محصولات: {len(products)}")

            # اضافه کردن لاگ برای محصولات موجود
            eyeglass_frames_count = sum(
                1 for p in products if is_eyeglass_frame(p))
            logger.info(f"تعداد فریم‌های عینک موجود: {eyeglass_frames_count}")

            # ذخیره در دیتابیس
            try:
                logger.info("در حال ذخیره محصولات دانلود شده در دیتابیس...")
                success = await save_woocommerce_cache(products, last_cache_update)
                if not success:
                    logger.warning(
                        "ذخیره کش محصولات در دیتابیس با مشکل مواجه شد")
            except Exception as db_error:
                logger.error(
                    f"خطا در ذخیره کش محصولات در دیتابیس: {str(db_error)}")

            logger.info(
                f"بروزرسانی کش محصولات WooCommerce با موفقیت انجام شد ({len(products)} محصول)")
            update_status["in_progress"] = False
            return True

        except Exception as e:
            logger.error(f"خطا در بروزرسانی کش محصولات WooCommerce: {str(e)}")
            update_status["in_progress"] = False
            update_status["last_error"] = str(e)
            return False


async def fetch_all_woocommerce_products() -> List[Dict[str, Any]]:
    """
    دریافت تمام محصولات از WooCommerce API و فیلتر کردن محصولات نامرتبط، ناموجود و بدون عکس.

    Returns:
        list: لیست محصولات فیلتر شده
    """
    try:
        logger.info("شروع دانلود محصولات از WooCommerce API...")

        # API پارامترها
        api_url = settings.WOOCOMMERCE_API_URL
        consumer_key = settings.WOOCOMMERCE_CONSUMER_KEY
        consumer_secret = settings.WOOCOMMERCE_CONSUMER_SECRET
        per_page = settings.WOOCOMMERCE_PER_PAGE

        # دسته‌بندی‌های مورد نظر
        categories = [
            {"id": 5215, "name": "computer-glasses"},
            {"id": 18, "name": "eyeglasses"},
            {"id": 17, "name": "sunglasses"},
            {"id": 5216, "name": "reading-glasses"}
        ]

        # مقداردهی اولیه لیست محصولات
        all_products = []
        total_downloaded = 0

        async with aiohttp.ClientSession() as session:
            # دانلود محصولات از هر دسته‌بندی
            for category in categories:
                page = 1
                category_products = []

                logger.info(
                    f"در حال دانلود محصولات دسته‌بندی {category['name']} (ID: {category['id']})...")

                while True:
                    # پارامترهای درخواست
                    params = {
                        "consumer_key": consumer_key,
                        "consumer_secret": consumer_secret,
                        "per_page": per_page,
                        "page": page,
                        "category": category["id"]
                    }

                    # ارسال درخواست
                    try:
                        async with session.get(api_url, params=params, timeout=30) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(
                                    f"خطا در WooCommerce API برای دسته‌بندی {category['name']}: {response.status} - {error_text}")
                                break

                            products = await response.json()

                            if not products:
                                logger.info(
                                    f"دانلود محصولات دسته‌بندی {category['name']} کامل شد. صفحه آخر: {page-1}")
                                break

                            total_downloaded += len(products)
                            logger.info(
                                f"دانلود صفحه {page} از دسته‌بندی {category['name']} با {len(products)} محصول انجام شد")
                            category_products.extend(products)

                            # بررسی تعداد محصولات دریافتی
                            if len(products) < per_page:
                                logger.info(
                                    f"دانلود محصولات دسته‌بندی {category['name']} کامل شد. صفحه آخر: {page}")
                                break

                            page += 1
                            logger.info(
                                f"در حال دانلود صفحه {page} از دسته‌بندی {category['name']}...")
                    except Exception as req_error:
                        logger.error(
                            f"خطا در ارسال درخواست به WooCommerce API برای دسته‌بندی {category['name']} (صفحه {page}): {str(req_error)}")
                        # کمی صبر کنیم و دوباره تلاش کنیم
                        logger.info(
                            f"تلاش مجدد برای دانلود صفحه {page} از دسته‌بندی {category['name']} پس از 2 ثانیه...")
                        await asyncio.sleep(2)
                        continue

                logger.info(
                    f"مجموع محصولات دانلود شده از دسته‌بندی {category['name']}: {len(category_products)}")
                all_products.extend(category_products)

        logger.info(f"پیش‌پردازش {len(all_products)} محصول دانلود شده...")

        # شمارنده‌های فیلتر
        # محصولات ناموجود (stock_status != instock)
        out_of_stock_count = 0
        invalid_permalink_count = 0   # محصولات با permalink نامعتبر
        unrelated_count = 0          # محصولات نامرتبط
        non_eyeglass_count = 0       # محصولات غیر فریم عینک
        lens_package_count = 0       # عدسی‌ها و پکیج‌های عدسی
        no_image_count = 0           # محصولات بدون تصویر
        valid_count = 0              # محصولات معتبر نهایی

        # پیش‌پردازش محصولات
        processed_products = []
        for product in all_products:
            # بررسی وضعیت موجودی - محصولات ناموجود را نادیده می‌گیریم
            if product.get("stock_status") != "instock":
                out_of_stock_count += 1
                continue

            # بررسی permalink - فقط محصولاتی که لینک معتبر دارند
            permalink = product.get("permalink", "")
            if "/?post_type=product&p=" in permalink or not "/product/" in permalink:
                invalid_permalink_count += 1
                continue

            # بررسی عدم ارتباط با محصولات نامرتبط
            if is_unrelated_product(product):
                unrelated_count += 1
                continue

            # بررسی اینکه فریم عینک باشد
            if not is_eyeglass_frame(product):
                non_eyeglass_count += 1
                continue

            # بررسی اینکه عدسی یا پکیج عدسی نباشد
            if is_lens_or_lens_package(product):
                lens_package_count += 1
                continue

            # بررسی وجود تصویر
            if not product.get("images"):
                no_image_count += 1
                continue

            # اضافه کردن محصول معتبر
            processed_products.append(product)
            valid_count += 1

        # لاگ‌های وضعیت فیلترها
        logger.info("===== آمار فیلتر محصولات =====")
        logger.info(f"تعداد کل محصولات دانلود شده: {len(all_products)}")
        logger.info(
            f"محصولات ناموجود (stock_status != instock): {out_of_stock_count}")
        logger.info(f"محصولات با لینک نامعتبر: {invalid_permalink_count}")
        logger.info(f"محصولات نامرتبط: {unrelated_count}")
        logger.info(f"محصولات غیر فریم عینک: {non_eyeglass_count}")
        logger.info(f"عدسی‌ها و پکیج‌های عدسی: {lens_package_count}")
        logger.info(f"محصولات بدون تصویر: {no_image_count}")
        logger.info(f"تعداد محصولات معتبر نهایی: {valid_count}")
        logger.info("================================")

        return processed_products

    except Exception as e:
        logger.error(f"خطا در دریافت محصولات از WooCommerce API: {str(e)}")
        return []


def is_valid_product(product: Dict[str, Any]) -> bool:
    """
    بررسی اعتبار کلی محصول.

    Args:
        product: محصول WooCommerce

    Returns:
        bool: True اگر محصول معتبر باشد
    """

    # تغییر بررسی قیمت به بررسی stock_status
    if product.get("stock_status") != "instock":
        logger.debug(
            f"محصول با ID {product.get('id')} ناموجود است (stock_status != instock)")
        return False

    # بررسی permalink - فقط محصولاتی که لینک آنها با الگوی /product/ شروع می‌شود
    permalink = product.get("permalink", "")
    if "/?post_type=product&p=" in permalink or not "/product/" in permalink:
        logger.debug(
            f"محصول با ID {product.get('id')} لینک نامعتبر دارد: {permalink}")
        return False

    # بررسی وجود تصویر
    if not product.get("images"):
        logger.debug(f"محصول با ID {product.get('id')} تصویر ندارد")
        return False

    # بررسی عدم ارتباط با محصولات نامرتبط
    if is_unrelated_product(product):
        logger.debug(f"محصول با ID {product.get('id')} نامرتبط است")
        return False

    # بررسی اینکه فریم عینک باشد و عدسی یا پکیج عدسی نباشد
    if not is_eyeglass_frame(product) or is_lens_or_lens_package(product):
        logger.debug(
            f"محصول با ID {product.get('id')} فریم عینک نیست یا عدسی/پکیج عدسی است")
        return False

    return True


def is_lens_or_lens_package(product: Dict[str, Any]) -> bool:
    """
    بررسی اینکه آیا محصول یک عدسی یا پکیج عدسی است.

    Args:
        product: محصول WooCommerce

    Returns:
        bool: True اگر محصول عدسی یا پکیج عدسی باشد
    """
    # بررسی نام محصول
    name = product.get("name", "").lower()
    description = product.get("description", "").lower()

    lens_keywords = [
        "عدسی", "عدسى", "lens", "lenses", "پکیج عدسی", "package lens",
        "optical", "محافظ", "بلوکات", "blue cut", "blue light",
        "آنتی رفلکس", "anti-reflective", "فتوکرومیک", "photochromic"
    ]

    # بررسی وجود کلیدواژه‌های عدسی در نام یا توضیحات
    for keyword in lens_keywords:
        if keyword in name or keyword in description:
            return True

    # بررسی دسته‌بندی‌های محصول
    categories = product.get("categories", [])
    for category in categories:
        category_name = category.get("name", "").lower()
        if "lens" in category_name or "عدسی" in category_name:
            return True

    # بررسی ویژگی‌های محصول
    attributes = product.get("attributes", [])
    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()
        if "lens" in attr_name or "عدسی" in attr_name:
            return True

    return False


async def get_all_products() -> List[Dict[str, Any]]:
    """
    دریافت تمام محصولات از کش. اگر کش منقضی شده باشد، آن را بروزرسانی می‌کند.

    Returns:
        list: لیست محصولات
    """
    global product_cache, last_cache_update

    # اگر کش موجود نیست، بروزرسانی کنیم
    if product_cache is None:
        await initialize_product_cache()

    # اگر کش قدیمی است (بیش از 24 ساعت)، بروزرسانی در پس‌زمینه
    if last_cache_update is None or datetime.now(timezone.utc) - last_cache_update > timedelta(hours=24):
        # بروزرسانی غیرمنتظر
        asyncio.create_task(refresh_product_cache())

    return product_cache if product_cache is not None else []


# def is_eyeglass_frame(product: Dict[str, Any]) -> bool:
#     """
#     بررسی اینکه آیا محصول یک فریم عینک است.

#     Args:
#         product: محصول WooCommerce

#     Returns:
#         bool: True اگر محصول فریم عینک باشد
#     """
#     # اگر قبلاً محاسبه شده، از آن استفاده کنیم
#     if "is_eyeglass_frame" in product:
#         return product["is_eyeglass_frame"]

#     # بررسی دسته‌بندی‌های محصول
#     categories = product.get("categories", [])
#     for category in categories:
#         category_id = category.get("id", 0)
#         category_name = category.get("name", "").lower()

#         # دسته‌بندی‌های مورد نظر ما
#         if category_id in [5215, 18, 17, 5216]:
#             # اما باید مطمئن شویم که عدسی نیست
#             if is_lens_or_lens_package(product):
#                 return False
#             return True

#         # بررسی نام دسته‌بندی
#         frame_keywords = ["عینک", "frame",
#                           "eyeglass", "glasses", "eyewear", "فریم"]
#         for keyword in frame_keywords:
#             if keyword in category_name:
#                 # اما باید مطمئن شویم که عدسی نیست
#                 if is_lens_or_lens_package(product):
#                     return False
#                 return True

#     # بررسی نام محصول
#     name = product.get("name", "").lower()
#     description = product.get("description", "").lower()

#     frame_keywords = ["عینک", "فریم", "eyeglass",
#                       "glasses", "frame", "eyewear"]
#     lens_keywords = ["عدسی", "lens", "package"]

#     # بررسی همزمان کلیدواژه‌های فریم و عدسی
#     has_frame_keyword = any(
#         keyword in name or keyword in description for keyword in frame_keywords)
#     has_lens_keyword = any(
#         keyword in name or keyword in description for keyword in lens_keywords)

#     # باید کلیدواژه فریم داشته باشد و کلیدواژه عدسی نداشته باشد
#     if has_frame_keyword and not has_lens_keyword:
#         return True

#     return False

def is_eyeglass_frame(product: Dict[str, Any]) -> bool:
    """
    بررسی اینکه آیا محصول یک فریم عینک است.
    """
    # بررسی ساده‌تر - فقط دسته‌بندی‌های اصلی عینک چک شوند
    categories = product.get("categories", [])
    for category in categories:
        category_id = category.get("id", 0)
        # اگر در یکی از دسته‌بندی‌های اصلی عینک است، معتبر تلقی شود
        if category_id in [5215, 18, 17, 5216]:  # دسته‌های عینک
            # فقط بررسی کنیم که نام محصول حاوی کلمه "عدسی" نباشد
            name = product.get("name", "").lower()
            if "lens" not in name and "عدسی" not in name and "پکیج" not in name:
                return True

    # بررسی اضافی اگر دسته‌بندی نداشت ولی در نام آن کلمه عینک یا فریم بود
    name = product.get("name", "").lower()
    if "عینک" in name or "فریم" in name or "eyeglasses" in name or "frame" in name:
        if "lens" not in name and "عدسی" not in name and "پکیج" not in name:
            return True

    return False


def get_frame_type(product: Dict[str, Any]) -> str:
    """
    استخراج نوع فریم از محصول.

    Args:
        product: محصول WooCommerce

    Returns:
        str: نوع فریم
    """
    # اگر قبلاً محاسبه شده، از آن استفاده کنیم
    if "frame_type" in product:
        return product["frame_type"]

    # بررسی ویژگی‌های محصول
    attributes = product.get("attributes", [])

    frame_type_attrs = ["شکل فریم", "نوع فریم",
                        "فرم فریم", "frame type", "frame shape"]

    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()

        # بررسی اینکه آیا این ویژگی مربوط به نوع فریم است
        is_frame_type_attr = any(
            frame_type in attr_name for frame_type in frame_type_attrs)

        if is_frame_type_attr:
            # دریافت مقدار ویژگی
            if "options" in attribute and attribute["options"]:
                # برگرداندن اولین گزینه
                return attribute["options"][0]

    # اگر نوع فریم خاصی پیدا نشد، سعی در استنباط از نام محصول
    name = product.get("name", "").lower()

    # نقشه نوع فریم به کلمات کلیدی
    try:
        # بارگیری نقشه از فایل داده
        with open(settings.FACE_SHAPE_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            frame_type_mappings = data.get('frame_type_mappings', {})
    except Exception as e:
        logger.error(f"خطا در بارگیری نقشه نوع فریم: {str(e)}")
        # نقشه پیش‌فرض
        frame_type_mappings = {
            "مستطیلی": ["مستطیل", "rectangular", "rectangle"],
            "مربعی": ["مربع", "square"],
            "گرد": ["گرد", "round", "circular"],
            "بیضی": ["بیضی", "oval"],
            "گربه‌ای": ["گربه", "cat eye", "cat-eye"],
            "هشت‌ضلعی": ["هشت", "octagonal", "octagon"],
            "هاوایی": ["هاوایی", "aviator"],
            "بدون‌فریم": ["بدون فریم", "rimless"]
        }

    for frame_type, keywords in frame_type_mappings.items():
        for keyword in keywords:
            if keyword in name:
                return frame_type

    # پیش‌فرض به یک نوع رایج
    return "مستطیلی"


def calculate_match_score(face_shape: str, frame_type: str) -> float:
    """
    محاسبه امتیاز تطابق بین شکل چهره و نوع فریم.

    Args:
        face_shape: شکل چهره
        frame_type: نوع فریم

    Returns:
        float: امتیاز تطابق (0-100)
    """
    # دریافت انواع فریم توصیه شده برای این شکل چهره
    recommended_types = get_recommended_frame_types(face_shape)

    if not recommended_types:
        return 50.0  # امتیاز متوسط پیش‌فرض

    # اگر نوع فریم در 2 نوع توصیه شده برتر باشد، امتیاز بالا
    if frame_type in recommended_types[:2]:
        return 90.0 + (recommended_types.index(frame_type) * -5.0)

    # اگر در لیست توصیه شده باشد اما نه در 2 نوع برتر، امتیاز متوسط
    if frame_type in recommended_types:
        position = recommended_types.index(frame_type)
        return 80.0 - (position * 5.0)

    # اگر در لیست توصیه شده نباشد، امتیاز پایین
    return 40.0


def filter_products_by_price(products: List[Dict[str, Any]], min_price: Optional[float] = None, max_price: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    فیلتر محصولات بر اساس قیمت.

    Args:
        products: لیست محصولات
        min_price: حداقل قیمت (اختیاری)
        max_price: حداکثر قیمت (اختیاری)

    Returns:
        list: محصولات فیلتر شده
    """
    if min_price is None and max_price is None:
        return products

    filtered_products = []

    for product in products:
        try:
            # بررسی وجود قیمت
            price_str = product.get("price", "")
            if not price_str:  # اگر قیمت خالی باشد، محصول را نادیده می‌گیریم
                continue

            # تبدیل قیمت به عدد
            price = float(price_str)

            # بررسی حداقل قیمت
            if min_price is not None and price < min_price:
                continue

            # بررسی حداکثر قیمت
            if max_price is not None and price > max_price:
                continue

            filtered_products.append(product)
        except (ValueError, TypeError):
            # در صورت خطا در تبدیل قیمت، محصول را نادیده می‌گیریم
            logger.warning(
                f"خطا در تبدیل قیمت برای محصول: {product.get('id', '')}")
            continue

    return filtered_products


async def get_eyeglass_frames(min_price: Optional[float] = None, max_price: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    دریافت همه فریم‌های عینک از کش.

    Args:
        min_price: حداقل قیمت (اختیاری)
        max_price: حداکثر قیمت (اختیاری)

    Returns:
        list: لیست فریم‌های عینک
    """
    # دریافت محصولات از کش
    products = await get_all_products()

    # فیلتر کردن فریم‌های عینک معتبر
    eyeglass_frames = []
    for product in products:

        # بررسی وضعیت موجودی
        if product.get("stock_status") != "instock":
            continue

        # بررسی permalink
        permalink = product.get("permalink", "")
        if "/?post_type=product&p=" in permalink or not "/product/" in permalink:
            continue

        # بررسی اینکه فریم عینک باشد
        if is_eyeglass_frame(product) and not is_lens_or_lens_package(product):
            eyeglass_frames.append(product)

    # فیلتر بر اساس قیمت (اگر درخواست شده باشد)
    if min_price is not None or max_price is not None:
        eyeglass_frames = filter_products_by_price(
            eyeglass_frames, min_price, max_price)

    logger.info(
        f"تعداد {len(eyeglass_frames)} فریم عینک معتبر از {len(products)} محصول یافت شد")
    return eyeglass_frames


def sort_products_by_match_score(products: List[Dict[str, Any]], face_shape: str) -> List[Dict[str, Any]]:
    """
    مرتب‌سازی محصولات بر اساس امتیاز تطابق با شکل چهره.

    Args:
        products: لیست محصولات
        face_shape: شکل چهره

    Returns:
        list: محصولات مرتب شده
    """
    # محاسبه امتیاز تطابق برای هر محصول
    for product in products:
        frame_type = get_frame_type(product)
        product["match_score"] = calculate_match_score(face_shape, frame_type)

    # مرتب‌سازی بر اساس امتیاز تطابق (نزولی)
    return sorted(products, key=lambda x: x.get("match_score", 0), reverse=True)


def is_unrelated_product(product: Dict[str, Any]) -> bool:
    """
    بررسی اینکه آیا محصول نامرتبط است (مانند "شارژ کیف پول" یا "ماوتفاوت محصول").

    Args:
        product: محصول WooCommerce

    Returns:
        bool: True اگر محصول نامرتبط باشد
    """
    # کلمات کلیدی برای فیلتر کردن محصولات نامرتبط
    unrelated_keywords = ["شارژ کیف پول",
                          "ماوتفاوت محصول", "شارژ کیف", "ماوتفاوت", "کیف پول"]

    # بررسی نام محصول
    name = product.get("name", "").lower()
    for keyword in unrelated_keywords:
        if keyword.lower() in name.lower():  # تغییر: تبدیل همه به lowercase برای مقایسه بهتر
            logger.debug(f"محصول نامرتبط تشخیص داده شد: {name}")
            return True

    # بررسی دسته‌بندی‌ها
    categories = product.get("categories", [])
    for category in categories:
        category_name = category.get("name", "").lower()
        for keyword in unrelated_keywords:
            if keyword.lower() in category_name.lower():  # تغییر: تبدیل همه به lowercase
                logger.debug(
                    f"محصول با دسته‌بندی نامرتبط تشخیص داده شد: {category_name}")
                return True

    # بررسی توضیحات کوتاه محصول
    short_description = product.get("short_description", "").lower()
    for keyword in unrelated_keywords:
        if keyword.lower() in short_description.lower():  # تغییر: بررسی توضیحات کوتاه
            logger.debug(f"محصول با توضیحات نامرتبط تشخیص داده شد")
            return True

    return False


async def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    """
    دریافت یک محصول خاص با شناسه از کش.

    Args:
        product_id: شناسه محصول

    Returns:
        dict: اطلاعات محصول یا None اگر پیدا نشود
    """
    # دریافت محصولات از کش
    products = await get_all_products()

    # جستجوی محصول با شناسه
    for product in products:
        if product.get("id") == product_id:
            # بررسی اعتبار محصول
            if is_valid_product(product):
                return product
            else:
                logger.debug(f"محصول با شناسه {product_id} نامعتبر است")
                return None

    # اگر در کش پیدا نشد، از API درخواست کنیم
    try:
        logger.info(f"دریافت محصول با شناسه {product_id} از WooCommerce API")

        # API پارامترها
        api_url = f"{settings.WOOCOMMERCE_API_URL}/{product_id}"
        consumer_key = settings.WOOCOMMERCE_CONSUMER_KEY
        consumer_secret = settings.WOOCOMMERCE_CONSUMER_SECRET

        # ارسال درخواست
        async with aiohttp.ClientSession() as session:
            params = {
                "consumer_key": consumer_key,
                "consumer_secret": consumer_secret
            }

            async with session.get(api_url, params=params, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"خطا در WooCommerce API: {response.status} - {error_text}")
                    return None

                product = await response.json()

                # بررسی اعتبار محصول
                if is_valid_product(product):
                    return product
                else:
                    logger.debug(f"محصول با شناسه {product_id} نامعتبر است")
                    return None

    except Exception as e:
        logger.error(f"خطا در دریافت محصول با شناسه {product_id}: {str(e)}")
        return None


async def get_products_by_category(category_id: int) -> List[Dict[str, Any]]:
    """
    دریافت محصولات یک دسته‌بندی خاص از کش.

    Args:
        category_id: شناسه دسته‌بندی

    Returns:
        list: لیست محصولات
    """
    # دریافت محصولات از کش
    products = await get_all_products()

    # فیلتر محصولات براساس دسته‌بندی و اعتبار
    category_products = []
    for product in products:
        # بررسی اعتبار محصول
        if not is_valid_product(product):
            continue

        # بررسی دسته‌بندی‌های محصول
        categories = product.get("categories", [])
        for category in categories:
            if category.get("id") == category_id:
                category_products.append(product)
                break

    return category_products


async def get_cache_status() -> Dict[str, Any]:
    """
    دریافت وضعیت فعلی کش محصولات.

    Returns:
        dict: وضعیت کش محصولات
    """
    global product_cache, last_cache_update, update_status

    return {
        "cache_initialized": product_cache is not None,
        "total_products": len(product_cache) if product_cache else 0,
        "last_update": last_cache_update,
        "update_in_progress": update_status["in_progress"],
        "last_error": update_status["last_error"],
        "eyeglass_frames_count": len([p for p in (product_cache or []) if p.get("is_eyeglass_frame", False)])
    }


async def get_recommended_frames(face_shape: str, min_price: Optional[float] = None, max_price: Optional[float] = None, limit: int = 15) -> Dict[str, Any]:
    """
    دریافت فریم‌های پیشنهادی بر اساس شکل چهره با ترکیبی از انواع مختلف عینک.
    """
    try:
        logger.info(f"دریافت فریم‌های پیشنهادی برای شکل چهره {face_shape}")

        # بررسی معتبر بودن شکل چهره
        valid_shapes = {"HEART", "OBLONG", "OVAL", "ROUND", "SQUARE"}
        if face_shape not in valid_shapes:
            logger.warning(
                f"شکل چهره {face_shape} معتبر نیست. استفاده از OVAL به عنوان پیش‌فرض.")
            face_shape = "OVAL"

        # دریافت انواع فریم توصیه شده برای این شکل چهره
        recommended_frame_types = get_recommended_frame_types(face_shape)

        if not recommended_frame_types:
            logger.warning(
                f"هیچ نوع فریمی برای شکل چهره {face_shape} توصیه نشده است")
            return {
                "success": False,
                "message": f"هیچ توصیه فریمی برای شکل چهره {face_shape} موجود نیست"
            }

        # دریافت فریم‌های عینک از کش
        all_frames = await get_eyeglass_frames(min_price, max_price)

        # فیلتر کردن محصولات نامرتبط به صورت دستی
        filtered_frames = [
            frame for frame in all_frames if not is_unrelated_product(frame)]
        if len(filtered_frames) < len(all_frames):
            logger.info(
                f"تعداد {len(all_frames) - len(filtered_frames)} محصول نامرتبط حذف شد")
            all_frames = filtered_frames

        logger.info(
            f"تعداد کل فریم‌های عینک پس از فیلتر اولیه: {len(all_frames)}")

        if not all_frames:
            logger.error("خطا در دریافت فریم‌های عینک از کش")
            return {
                "success": False,
                "message": "خطا در دریافت فریم‌های موجود"
            }

        # اگر تعداد فریم‌های موجود کمتر از حد درخواستی است، حداقل نیمی از فریم‌ها
        # را به فریم‌های پیشنهادی اختصاص می‌دهیم
        # تعدیل این قسمت از کد
        if len(all_frames) < limit * 2:
            logger.warning(
                f"تعداد کل فریم‌ها ({len(all_frames)}) کمتر از دو برابر حد درخواستی ({limit*2}) است")
            # تغییر از 'len(all_frames) // 2' به 'min(len(all_frames), limit)'
            recommended_limit = min(len(all_frames), limit)
        else:
            recommended_limit = limit

        # جداسازی عینک‌ها براساس دسته‌بندی
        eyeglasses_frames = []  # عینک طبی (ID: 18)
        sunglasses_frames = []  # عینک آفتابی (ID: 17)
        other_frames = []  # سایر انواع عینک

        for frame in all_frames:
            # بررسی اینکه آیا محصول نامرتبط است
            if is_unrelated_product(frame):
                logger.info(f"محصول نامرتبط حذف شد: {frame.get('name', '')}")
                continue

            # بررسی دسته‌بندی‌های هر محصول
            categories = frame.get("categories", [])
            category_ids = [cat.get("id") for cat in categories]

            if 17 in category_ids:  # عینک آفتابی
                sunglasses_frames.append(frame)
            elif 18 in category_ids:  # عینک طبی
                eyeglasses_frames.append(frame)
            else:  # سایر انواع
                other_frames.append(frame)

        logger.info(f"تعداد عینک‌های طبی: {len(eyeglasses_frames)}")
        logger.info(f"تعداد عینک‌های آفتابی: {len(sunglasses_frames)}")
        logger.info(f"تعداد سایر عینک‌ها: {len(other_frames)}")

        # محاسبه امتیاز تطابق برای هر دسته
        for frame in eyeglasses_frames + sunglasses_frames + other_frames:
            frame_type = get_frame_type(frame)
            frame["match_score"] = calculate_match_score(
                face_shape, frame_type)

        # مرتب‌سازی هر دسته براساس امتیاز تطابق
        eyeglasses_frames = sorted(
            eyeglasses_frames, key=lambda x: x.get("match_score", 0), reverse=True)
        sunglasses_frames = sorted(
            sunglasses_frames, key=lambda x: x.get("match_score", 0), reverse=True)
        other_frames = sorted(other_frames, key=lambda x: x.get(
            "match_score", 0), reverse=True)

        # محاسبه تعداد فریم‌ها از هر دسته براساس توزیع تعیین شده
        eyeglasses_count = int(limit * 0.4)  # 40% عینک طبی
        sunglasses_count = int(limit * 0.5)  # 50% عینک آفتابی
        other_count = limit - eyeglasses_count - sunglasses_count  # 10% سایر

        # تنظیم تعداد در صورت کمبود داده در هر دسته
        if len(eyeglasses_frames) < eyeglasses_count:
            shortfall = eyeglasses_count - len(eyeglasses_frames)
            eyeglasses_count = len(eyeglasses_frames)
            # توزیع کمبود بین دسته‌های دیگر
            sunglasses_count += int(shortfall * 0.8)
            other_count += shortfall - int(shortfall * 0.8)

        if len(sunglasses_frames) < sunglasses_count:
            shortfall = sunglasses_count - len(sunglasses_frames)
            sunglasses_count = len(sunglasses_frames)
            # اختصاص همه کمبود به دسته دیگر
            other_count += shortfall

        if len(other_frames) < other_count:
            shortfall = other_count - len(other_frames)
            other_count = len(other_frames)
            # اختصاص کمبود به عینک‌های طبی
            eyeglasses_count += shortfall

            # اگر عینک طبی کافی نباشد، به آفتابی اختصاص دهیم
            if len(eyeglasses_frames) < eyeglasses_count:
                shortfall = eyeglasses_count - len(eyeglasses_frames)
                eyeglasses_count = len(eyeglasses_frames)
                sunglasses_count += shortfall

        # تقسیم هر دسته به دو بخش: با امتیاز بالا و انتخاب تصادفی
        # برای هر دسته، 60% از محصولات بر اساس امتیاز و 40% به صورت تصادفی انتخاب می‌شوند
        # این قسمت را در تابع get_recommended_frames جایگزین کن
        # استفاده از تنوع بیشتر در انتخاب فریم‌ها
        def select_diverse_frames(frames, count):
            if not frames or count <= 0:
                return []

            # انتخاب فریم‌های با امتیاز بالا (50%)
            top_count = max(1, int(count * 0.5))
            top_frames = frames[:top_count]

            # انتخاب فریم‌های تصادفی از باقیمانده (50%)
            remaining_frames = frames[top_count:]
            random_count = count - top_count
            if remaining_frames and random_count > 0:
                # اگر تعداد فریم‌های باقیمانده کمتر از تعداد مورد نیاز است، همه را انتخاب کن
                if len(remaining_frames) <= random_count:
                    random_frames = remaining_frames
                else:
                    random_frames = random.sample(
                        remaining_frames, random_count)
            else:
                random_frames = []

            # ترکیب لیست‌ها
            selected = top_frames + random_frames
            return selected

        # انتخاب فریم‌ها با روش متنوع
        selected_eyeglasses = select_diverse_frames(
            eyeglasses_frames, eyeglasses_count)
        selected_sunglasses = select_diverse_frames(
            sunglasses_frames, sunglasses_count)
        selected_others = select_diverse_frames(other_frames, other_count)

        # ترکیب فریم‌های انتخاب شده
        selected_frames = selected_eyeglasses + selected_sunglasses + selected_others

        # ایجاد تنوع در نتایج نهایی با جابجایی تصادفی
        random.shuffle(selected_frames)

        logger.info(
            f"تعداد فریم‌های انتخاب شده (قبل از نهایی): {len(selected_frames)}")

        # اگر تعداد فریم‌های انتخاب شده کمتر از حد مورد نظر است
        if len(selected_frames) < limit:
            logger.warning(
                f"تعداد فریم‌های انتخابی ({len(selected_frames)}) کمتر از تعداد درخواستی ({limit}) است")

            # استفاده از تمام فریم‌های موجود
            all_sorted_frames = sorted(all_frames, key=lambda x: x.get(
                "match_score", 0) if "match_score" in x else calculate_match_score(face_shape, get_frame_type(x)), reverse=True)

            # فیلتر کردن فریم‌هایی که قبلاً انتخاب شده‌اند
            remaining_frames = [
                f for f in all_sorted_frames if f not in selected_frames]

            # اضافه کردن فریم‌های باقیمانده تا رسیدن به تعداد درخواستی
            frames_to_add = min(limit - len(selected_frames),
                                len(remaining_frames))
            if frames_to_add > 0:
                additional_frames = remaining_frames[:frames_to_add]
                selected_frames.extend(additional_frames)
                logger.info(
                    f"{frames_to_add} فریم اضافی برای رسیدن به تعداد درخواستی اضافه شد")

        # محدود کردن به تعداد درخواستی
        if len(selected_frames) > limit:
            selected_frames = selected_frames[:limit]

        # تبدیل به فرمت پاسخ مورد نظر
        recommended_frames = []
        for product in selected_frames:
            # بررسی وضعیت موجودی - محصولات ناموجود را نادیده می‌گیریم
            if product.get("stock_status") != "instock":
                continue

            # بررسی permalink
            permalink = product.get("permalink", "")
            if "/?post_type=product&p=" in permalink or not "/product/" in permalink:
                continue

            frame_type = product.get("frame_type", get_frame_type(product))
            match_score = product.get("match_score", 0)

            # تعیین نوع عینک (طبی، آفتابی، سایر)
            eyeglass_type = "سایر"
            categories = product.get("categories", [])
            category_ids = [cat.get("id") for cat in categories]

            if 17 in category_ids:
                eyeglass_type = "آفتابی"
            elif 18 in category_ids:
                eyeglass_type = "طبی"

            recommended_frames.append({
                "id": product["id"],
                "name": product["name"],
                "permalink": product["permalink"],
                "price": product.get("price", ""),
                "regular_price": product.get("regular_price", ""),
                "frame_type": frame_type,
                "eyeglass_type": eyeglass_type,  # اضافه کردن نوع عینک
                "images": [img["src"] for img in product.get("images", [])[:3]],
                "match_score": match_score
            })

        logger.info(
            f"تعداد نهایی فریم‌های پیشنهادی: {len(recommended_frames)}")
        logger.info(
            f"پیشنهاد فریم کامل شد: {len(recommended_frames)} توصیه (طبی: {len(selected_eyeglasses)}, آفتابی: {len(selected_sunglasses)}, سایر: {len(selected_others)})")

        return {
            "success": True,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommended_frames": recommended_frames,
            "total_matches": len(recommended_frames),
            "distribution": {
                "eyeglasses": len(selected_eyeglasses),
                "sunglasses": len(selected_sunglasses),
                "others": len(selected_others)
            }
        }

    except Exception as e:
        logger.error(f"خطا در تطبیق فریم: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تطبیق فریم: {str(e)}"
        }
