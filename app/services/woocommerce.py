import logging
import json
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import threading
import time
import schedule

from app.config import settings
from app.core.face_shape_data import get_recommended_frame_types
from app.db.connection import get_database


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
    global product_cache, last_cache_update, update_scheduler
    
    logger.info("شروع راه‌اندازی اولیه کش محصولات WooCommerce")
    
    # بررسی وجود کش در دیتابیس
    try:
        db = get_database()
        cache_record = await db.woocommerce_cache.find_one({"type": "products_cache"})
        
        if cache_record and "last_update" in cache_record:
            # بررسی تازگی کش
            last_update = cache_record["last_update"]
            if (datetime.utcnow() - last_update) < timedelta(hours=24):
                logger.info(f"استفاده از کش موجود در دیتابیس (بروزرسانی آخر: {last_update})")
                product_cache = cache_record["data"]
                last_cache_update = last_update
                update_status["last_update"] = last_update
                update_status["total_products"] = len(product_cache) if product_cache else 0
                
                # راه‌اندازی بروزرسانی خودکار
                start_scheduled_updates()
                return
            else:
                logger.info("کش محصولات در دیتابیس منقضی شده است. نیاز به بروزرسانی دارد.")
        else:
            logger.info("کش محصولات در دیتابیس یافت نشد. در حال دانلود محصولات از WooCommerce...")
    except Exception as e:
        logger.warning(f"خطا در بررسی کش دیتابیس: {str(e)}")
    
    # بروزرسانی اولیه
    await refresh_product_cache(force=True)
    
    # راه‌اندازی بروزرسانی خودکار
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
        schedule.every().day.at("03:00").do(run_async_update)
        
        while True:
            schedule.run_pending()
            # بررسی هر 15 دقیقه
            time.sleep(900)
    
    def run_async_update():
        logger.info("اجرای زمان‌بندی شده بروزرسانی کش محصولات WooCommerce")
        asyncio.run(refresh_product_cache(force=True))
    
    # اجرای زمان‌بندی در یک ترد جداگانه
    import time
    update_scheduler = threading.Thread(target=run_scheduler, daemon=True)
    update_scheduler.start()
    
    logger.info("زمان‌بندی بروزرسانی خودکار محصولات WooCommerce فعال شد (روزی یکبار در ساعت 3 صبح)")


async def refresh_product_cache(force=False):
    """
    بروزرسانی کش محصولات WooCommerce.
    
    Args:
        force: اجبار به بروزرسانی حتی اگر کش معتبر باشد
        
    Returns:
        bool: نتیجه بروزرسانی
    """
    global product_cache, last_cache_update, update_status
    
    # بررسی وضعیت فعلی کش
    if not force and product_cache is not None and last_cache_update is not None:
        # اگر کمتر از 24 ساعت از آخرین بروزرسانی گذشته باشد، نیازی به بروزرسانی نیست
        if datetime.utcnow() - last_cache_update < timedelta(hours=24):
            logger.info("کش محصولات WooCommerce معتبر است و نیاز به بروزرسانی ندارد")
            return True
    
    # از قفل برای جلوگیری از بروزرسانی‌های همزمان استفاده می‌کنیم
    async with refresh_lock:
        # بررسی مجدد پس از گرفتن قفل
        if not force and product_cache is not None and last_cache_update is not None:
            if datetime.utcnow() - last_cache_update < timedelta(hours=24):
                return True
                
        # تنظیم وضعیت بروزرسانی
        update_status["in_progress"] = True
        update_status["last_error"] = None
        
        try:
            logger.info("شروع فرآیند دانلود و بروزرسانی کش محصولات از WooCommerce API")
            
            # دریافت محصولات از API
            products = await fetch_all_woocommerce_products()
            
            if not products:
                logger.error("خطا در دریافت محصولات از WooCommerce API")
                update_status["in_progress"] = False
                update_status["last_error"] = "خطا در دریافت محصولات از API"
                return False
            
            # بروزرسانی کش
            product_cache = products
            last_cache_update = datetime.utcnow()
            update_status["last_update"] = last_cache_update
            update_status["total_products"] = len(products)
            
            logger.info(f"دانلود محصولات از WooCommerce API با موفقیت انجام شد. تعداد محصولات: {len(products)}")
            
            # ذخیره در دیتابیس با روش جدید چند بخشی
            try:
                logger.info("در حال ذخیره محصولات دانلود شده در دیتابیس...")
                success = await save_woocommerce_cache(products, last_cache_update)
                if not success:
                    logger.warning("ذخیره کش محصولات در دیتابیس با مشکل مواجه شد")
            except Exception as db_error:
                logger.error(f"خطا در ذخیره کش محصولات در دیتابیس: {str(db_error)}")
            
            logger.info(f"بروزرسانی کش محصولات WooCommerce با موفقیت انجام شد ({len(products)} محصول)")
            update_status["in_progress"] = False
            return True
            
        except Exception as e:
            logger.error(f"خطا در بروزرسانی کش محصولات WooCommerce: {str(e)}")
            update_status["in_progress"] = False
            update_status["last_error"] = str(e)
            return False


async def fetch_all_woocommerce_products() -> List[Dict[str, Any]]:
    """
    دریافت تمام محصولات از WooCommerce API.
    
    Returns:
        list: لیست محصولات
    """
    try:
        logger.info("شروع دانلود محصولات از WooCommerce API...")
        
        # API پارامترها
        api_url = settings.WOOCOMMERCE_API_URL
        consumer_key = settings.WOOCOMMERCE_CONSUMER_KEY
        consumer_secret = settings.WOOCOMMERCE_CONSUMER_SECRET
        per_page = settings.WOOCOMMERCE_PER_PAGE
        
        # مقداردهی اولیه لیست محصولات
        all_products = []
        page = 1
        
        async with aiohttp.ClientSession() as session:
            logger.info(f"در حال دانلود صفحه 1 از محصولات...")
            while True:
                # پارامترهای درخواست
                params = {
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "per_page": per_page,
                    "page": page
                }
                
                # ارسال درخواست
                try:
                    async with session.get(api_url, params=params, timeout=30) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"خطا در WooCommerce API: {response.status} - {error_text}")
                            break
                            
                        products = await response.json()
                        
                        if not products:
                            logger.info(f"دانلود محصولات از WooCommerce API کامل شد. صفحه آخر: {page-1}")
                            break
                            
                        logger.info(f"دانلود صفحه {page} با {len(products)} محصول انجام شد")
                        all_products.extend(products)
                        
                        # بررسی تعداد محصولات دریافتی
                        if len(products) < per_page:
                            logger.info(f"دانلود محصولات از WooCommerce API کامل شد. صفحه آخر: {page}")
                            break
                            
                        page += 1
                        logger.info(f"در حال دانلود صفحه {page} از محصولات...")
                except Exception as req_error:
                    logger.error(f"خطا در ارسال درخواست به WooCommerce API (صفحه {page}): {str(req_error)}")
                    # کمی صبر کنیم و دوباره تلاش کنیم
                    logger.info(f"تلاش مجدد برای دانلود صفحه {page} پس از 2 ثانیه...")
                    await asyncio.sleep(2)
                    continue
        
        logger.info(f"پیش‌پردازش {len(all_products)} محصول دانلود شده...")
        # پیش‌پردازش محصولات
        for product in all_products:
            # افزودن فیلد نوع فریم
            product["frame_type"] = get_frame_type(product)
            
            # افزودن فیلد آیا فریم عینک است
            product["is_eyeglass_frame"] = is_eyeglass_frame(product)
        
        eyeglass_count = sum(1 for p in all_products if p.get("is_eyeglass_frame", False))
        logger.info(f"دانلود و پیش‌پردازش {len(all_products)} محصول از WooCommerce API کامل شد. تعداد فریم عینک: {eyeglass_count}")
        return all_products
        
    except Exception as e:
        logger.error(f"خطا در دریافت محصولات از WooCommerce API: {str(e)}")
        return []


async def get_all_products() -> List[Dict[str, Any]]:
    """
    دریافت تمام محصولات از کش.
    
    Returns:
        list: لیست محصولات
    """
    global product_cache, last_cache_update
    
    # اگر کش موجود نیست، بروزرسانی کنیم
    if product_cache is None:
        await initialize_product_cache()
    
    # اگر کش قدیمی است (بیش از 24 ساعت)، بروزرسانی در پس‌زمینه
    if last_cache_update is None or datetime.utcnow() - last_cache_update > timedelta(hours=24):
        # بروزرسانی غیرمنتظر
        asyncio.create_task(refresh_product_cache())
    
    return product_cache if product_cache is not None else []


def is_eyeglass_frame(product: Dict[str, Any]) -> bool:
    """
    بررسی اینکه آیا محصول یک فریم عینک است.
    
    Args:
        product: محصول WooCommerce
        
    Returns:
        bool: True اگر محصول فریم عینک باشد
    """
    # اگر قبلاً محاسبه شده، از آن استفاده کنیم
    if "is_eyeglass_frame" in product:
        return product["is_eyeglass_frame"]
    
    # بررسی دسته‌بندی‌های محصول
    categories = product.get("categories", [])
    for category in categories:
        category_name = category.get("name", "").lower()
        if "عینک" in category_name or "frame" in category_name or "eyeglass" in category_name:
            return True
    
    # بررسی ویژگی‌های محصول
    attributes = product.get("attributes", [])
    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()
        if "frame" in attr_name or "شکل" in attr_name or "فریم" in attr_name or "نوع" in attr_name:
            return True
    
    # بررسی نام محصول
    name = product.get("name", "").lower()
    keywords = ["عینک", "فریم", "eyeglass", "glasses", "frame"]
    for keyword in keywords:
        if keyword in name:
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
    
    frame_type_attrs = ["شکل فریم", "نوع فریم", "فرم فریم", "frame type", "frame shape"]
    
    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()
        
        # بررسی اینکه آیا این ویژگی مربوط به نوع فریم است
        is_frame_type_attr = any(frame_type in attr_name for frame_type in frame_type_attrs)
        
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
            price = float(product.get("price", 0))
            
            # بررسی حداقل قیمت
            if min_price is not None and price < min_price:
                continue
                
            # بررسی حداکثر قیمت
            if max_price is not None and price > max_price:
                continue
                
            filtered_products.append(product)
        except (ValueError, TypeError):
            # در صورت خطا در تبدیل قیمت، محصول را نادیده می‌گیریم
            logger.warning(f"خطا در تبدیل قیمت برای محصول: {product.get('id', '')}")
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
    
    # فیلتر کردن فریم‌های عینک
    eyeglass_frames = [product for product in products if is_eyeglass_frame(product)]
    
    # فیلتر بر اساس قیمت (اگر درخواست شده باشد)
    if min_price is not None or max_price is not None:
        eyeglass_frames = filter_products_by_price(eyeglass_frames, min_price, max_price)
    
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


async def get_recommended_frames(face_shape: str, min_price: Optional[float] = None, max_price: Optional[float] = None, limit: int = 10) -> Dict[str, Any]:
    """
    دریافت فریم‌های پیشنهادی بر اساس شکل چهره.
    
    Args:
        face_shape: شکل چهره
        min_price: حداقل قیمت (اختیاری)
        max_price: حداکثر قیمت (اختیاری)
        limit: حداکثر تعداد توصیه‌ها
        
    Returns:
        dict: نتیجه عملیات شامل فریم‌های توصیه شده
    """
    try:
        logger.info(f"دریافت فریم‌های پیشنهادی برای شکل چهره {face_shape}")
        
        # دریافت انواع فریم توصیه شده برای این شکل چهره
        recommended_frame_types = get_recommended_frame_types(face_shape)
        
        if not recommended_frame_types:
            logger.warning(f"هیچ نوع فریمی برای شکل چهره {face_shape} توصیه نشده است")
            return {
                "success": False,
                "message": f"هیچ توصیه فریمی برای شکل چهره {face_shape} موجود نیست"
            }
        
        # دریافت فریم‌های عینک از کش
        eyeglass_frames = await get_eyeglass_frames(min_price, max_price)
        
        if not eyeglass_frames:
            logger.error("خطا در دریافت فریم‌های عینک از کش")
            return {
                "success": False,
                "message": "خطا در دریافت فریم‌های موجود"
            }
        
        # مرتب‌سازی بر اساس امتیاز تطابق
        sorted_frames = sort_products_by_match_score(eyeglass_frames, face_shape)
        
        # تبدیل به فرمت پاسخ مورد نظر
        recommended_frames = []
        for product in sorted_frames[:limit]:
            frame_type = product.get("frame_type", get_frame_type(product))
            match_score = product.get("match_score", 0)
            
            recommended_frames.append({
                "id": product["id"],
                "name": product["name"],
                "permalink": product["permalink"],
                "price": product.get("price", ""),
                "regular_price": product.get("regular_price", ""),
                "frame_type": frame_type,
                "images": [img["src"] for img in product.get("images", [])[:3]],
                "match_score": match_score
            })
        
        logger.info(f"تطبیق فریم کامل شد: {len(recommended_frames)} توصیه پیدا شد")
        
        return {
            "success": True,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommended_frames": recommended_frames,
            "total_matches": len(recommended_frames)
        }
        
    except Exception as e:
        logger.error(f"خطا در تطبیق فریم: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تطبیق فریم: {str(e)}"
        }


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
            return product
    
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
                    logger.error(f"خطا در WooCommerce API: {response.status} - {error_text}")
                    return None
                    
                product = await response.json()
                return product
                
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
    
    # فیلتر محصولات براساس دسته‌بندی
    category_products = []
    for product in products:
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
        "last_error": update_status["last_error"]
    }
    

async def save_woocommerce_cache(products: List[Dict[str, Any]], last_update: datetime) -> bool:
    """
    ذخیره کش محصولات WooCommerce در دیتابیس به صورت چند بخشی.
    
    Args:
        products: لیست محصولات
        last_update: زمان آخرین بروزرسانی
        
    Returns:
        bool: نتیجه عملیات ذخیره‌سازی
    """
    try:
        db = get_database()
        
        # حذف تمام رکوردهای قبلی مرتبط با کش
        logger.info("حذف رکوردهای قبلی کش از دیتابیس...")
        await db.woocommerce_cache.delete_many({"type": {"$regex": "^products_cache"}})
        
        # تعیین اندازه هر بخش (300 محصول در هر بخش)
        chunk_size = 300
        chunks = [products[i:i + chunk_size] for i in range(0, len(products), chunk_size)]
        
        logger.info(f"تقسیم {len(products)} محصول به {len(chunks)} بخش (هر بخش حداکثر {chunk_size} محصول)")
        
        # ذخیره هر بخش به صورت جداگانه
        for i, chunk in enumerate(chunks):
            await db.woocommerce_cache.insert_one({
                "type": f"products_cache_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "last_update": last_update,
                "data": chunk
            })
            logger.info(f"بخش {i+1} از {len(chunks)} با {len(chunk)} محصول ذخیره شد")
        
        # ذخیره متادیتا
        await db.woocommerce_cache.insert_one({
            "type": "products_cache_meta",
            "last_update": last_update,
            "total_products": len(products),
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "eyeglass_frames_count": sum(1 for p in products if p.get("is_eyeglass_frame", False))
        })
        
        logger.info(f"کش محصولات WooCommerce با موفقیت در دیتابیس ذخیره شد ({len(products)} محصول در {len(chunks)} بخش)")
        return True
        
    except Exception as e:
        logger.error(f"خطا در ذخیره کش محصولات در دیتابیس: {str(e)}")
        return False


async def get_woocommerce_cache() -> Tuple[Optional[List[Dict[str, Any]]], Optional[datetime]]:
    """
    دریافت کش محصولات WooCommerce از دیتابیس.
    
    Returns:
        tuple: (محصولات، تاریخ_آخرین_بروزرسانی) یا (None, None) در صورت عدم وجود کش
    """
    try:
        db = get_database()
        
        # دریافت متادیتای کش
        meta = await db.woocommerce_cache.find_one({"type": "products_cache_meta"})
        
        if not meta:
            logger.warning("متادیتای کش محصولات در دیتابیس یافت نشد")
            return None, None
        
        total_chunks = meta.get("total_chunks", 0)
        last_update = meta.get("last_update")
        
        # دریافت تمام بخش‌ها
        all_products = []
        for i in range(total_chunks):
            chunk = await db.woocommerce_cache.find_one({"type": f"products_cache_chunk_{i}"})
            if chunk and "data" in chunk:
                all_products.extend(chunk["data"])
                logger.debug(f"بخش {i+1} از {total_chunks} با {len(chunk['data'])} محصول بازیابی شد")
        
        if not all_products:
            logger.warning("داده‌های کش محصولات در دیتابیس یافت نشد")
            return None, None
        
        logger.info(f"{len(all_products)} محصول از کش دیتابیس بازیابی شد")
        return all_products, last_update
        
    except Exception as e:
        logger.error(f"خطا در دریافت کش محصولات از دیتابیس: {str(e)}")
        return None, None