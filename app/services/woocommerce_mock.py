# app/services/woocommerce_mock.py
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import random
import os

from app.config import settings
from app.core.face_shape_data import get_recommended_frame_types

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# کش برای محصولات مصنوعی
mock_product_cache = None
last_cache_update = None

# انواع فریم‌های عینک
FRAME_TYPES = [
    "مستطیلی", "مربعی", "گرد", "بیضی", "گربه‌ای",
    "هشت‌ضلعی", "هاوایی", "پایین‌بدون‌فریم", "بدون‌فریم"
]

# رنگ‌های فریم
FRAME_COLORS = ["مشکی", "قهوه‌ای", "طلایی",
                "نقره‌ای", "آبی", "قرمز", "سبز", "صورتی"]

# جنس‌های فریم
FRAME_MATERIALS = ["فلزی", "پلاستیکی", "استیل", "چوبی", "تیتانیوم", "کربنی"]

# برندهای فریم
BRANDS = ["RayBan", "Oakley", "Prada", "Gucci", "Chanel",
          "Versace", "Dior", "Tom Ford", "لوناتو", "عینک پلاس"]

# قیمت‌های پایه (بر حسب تومان)
BASE_PRICES = [
    500000, 750000, 1000000, 1250000, 1500000, 1800000,
    2000000, 2500000, 3000000, 3500000, 4000000, 5000000
]


async def initialize_product_cache():
    """
    راه‌اندازی اولیه کش محصولات مصنوعی
    """
    global mock_product_cache, last_cache_update

    logger.info("شروع راه‌اندازی داده‌های مصنوعی WooCommerce")

    # ایجاد داده‌های مصنوعی
    mock_product_cache = generate_mock_products(100)
    last_cache_update = datetime.utcnow()

    logger.info(f"داده‌های مصنوعی ایجاد شدند: {len(mock_product_cache)} محصول")


def generate_mock_products(num_products=100) -> List[Dict[str, Any]]:
    """
    تولید محصولات مصنوعی برای استفاده در محیط توسعه

    Args:
        num_products: تعداد محصولات مصنوعی که باید تولید شود

    Returns:
        list: لیست محصولات مصنوعی
    """
    logger.info(f"تولید {num_products} محصول مصنوعی برای WooCommerce")

    products = []
    for i in range(1, num_products + 1):
        # انتخاب تصادفی ویژگی‌ها
        frame_type = random.choice(FRAME_TYPES)
        frame_color = random.choice(FRAME_COLORS)
        frame_material = random.choice(FRAME_MATERIALS)
        brand = random.choice(BRANDS)
        base_price = random.choice(BASE_PRICES)

        # تعیین قیمت تصادفی با تخفیف احتمالی
        regular_price = base_price
        has_discount = random.random() < 0.3  # 30% احتمال داشتن تخفیف

        if has_discount:
            discount_percent = random.choice([10, 15, 20, 25, 30, 40, 50])
            price = int(regular_price * (100 - discount_percent) / 100)
        else:
            price = regular_price

        # ساخت نام محصول
        product_name = f"فریم {frame_type} {brand} مدل {frame_color} {frame_material}"

        # تعیین تصادفی تصاویر (از 1 تا 4 تصویر)
        num_images = random.randint(1, 4)
        images = []
        for j in range(1, num_images + 1):
            img_id = random.randint(1, 1000)
            images.append({
                "id": img_id,
                "src": f"https://lunato.shop/wp-content/uploads/frames/frame_{img_id}.jpg",
                "name": f"frame_{img_id}",
                "alt": f"تصویر {j} {product_name}"
            })

        # انتخاب تصادفی دسته‌بندی‌ها
        categories = [
            {"id": 15, "name": "فریم عینک", "slug": "eyeglass-frames"}
        ]

        # اضافه کردن دسته‌بندی‌های تصادفی دیگر
        if random.random() < 0.7:  # 70% احتمال داشتن دسته‌بندی جنسیت
            gender_cat = random.choice([
                {"id": 16, "name": "فریم زنانه", "slug": "women-frames"},
                {"id": 17, "name": "فریم مردانه", "slug": "men-frames"},
                {"id": 18, "name": "فریم یونیسکس", "slug": "unisex-frames"}
            ])
            categories.append(gender_cat)

        if random.random() < 0.5:  # 50% احتمال داشتن دسته‌بندی برند
            brand_id = 20 + BRANDS.index(brand)
            categories.append({
                "id": brand_id,
                "name": f"برند {brand}",
                "slug": f"brand-{brand.lower().replace(' ', '-')}"
            })

        # ساخت ویژگی‌های محصول
        attributes = [
            {
                "id": 1,
                "name": "شکل فریم",
                "options": [frame_type]
            },
            {
                "id": 2,
                "name": "رنگ",
                "options": [frame_color]
            },
            {
                "id": 3,
                "name": "جنس",
                "options": [frame_material]
            },
            {
                "id": 4,
                "name": "برند",
                "options": [brand]
            }
        ]

        # ساخت توضیحات محصول
        description = f"""<p>فریم عینک {brand} مدل {frame_type} {frame_color}</p>
<ul>
<li>جنس: {frame_material}</li>
<li>نوع فریم: {frame_type}</li>
<li>رنگ: {frame_color}</li>
<li>مناسب برای انواع شکل صورت</li>
<li>دارای گارانتی اصالت و سلامت کالا</li>
</ul>"""

        # ساخت محصول مصنوعی
        product = {
            "id": i,
            "name": product_name,
            "slug": f"frame-{i}-{brand.lower().replace(' ', '-')}",
            "permalink": f"https://lunato.shop/product/frame-{i}",
            "price": str(price),
            "regular_price": str(regular_price),
            "description": description,
            "short_description": f"فریم {frame_type} {brand} مدل {frame_color} {frame_material}",
            "categories": categories,
            "attributes": attributes,
            "images": images,
            "frame_type": frame_type,
            "is_eyeglass_frame": True
        }

        # تنوع بیشتر در داده‌ها - امتیاز تصادفی
        if random.random() < 0.7:  # 70% محصولات امتیاز دارند
            product["rating_count"] = random.randint(3, 50)
            product["average_rating"] = round(random.uniform(3.0, 5.0), 1)

        # تنوع بیشتر در داده‌ها - تگ‌های محصول
        if random.random() < 0.6:  # 60% محصولات تگ دارند
            num_tags = random.randint(1, 3)
            tags = []
            possible_tags = ["پرفروش", "جدید", "محبوب", "پیشنهاد ویژه",
                             "تخفیف", "فریم لوکس", "فریم سبک", "فریم کلاسیک"]
            selected_tags = random.sample(
                possible_tags, min(num_tags, len(possible_tags)))

            for j, tag_name in enumerate(selected_tags):
                tags.append({
                    "id": 100 + j,
                    "name": tag_name,
                    "slug": f"tag-{j}"
                })

            product["tags"] = tags

        # تنوع بیشتر در داده‌ها - موجودی محصول
        product["stock_status"] = "instock" if random.random(
        ) < 0.85 else "outofstock"  # 85% محصولات موجود هستند
        if product["stock_status"] == "instock":
            product["stock_quantity"] = random.randint(1, 20)

        # ابعاد محصول
        product["dimensions"] = {
            "length": str(random.randint(120, 145)),  # طول فریم به میلی‌متر
            "width": str(random.randint(30, 50)),    # عرض فریم به میلی‌متر
            # ارتفاع/ضخامت فریم به میلی‌متر
            "height": str(random.randint(5, 15))
        }

        products.append(product)

    logger.info(f"{num_products} محصول مصنوعی با موفقیت تولید شد")
    return products


async def refresh_product_cache(force=False):
    """
    بروزرسانی کش محصولات مصنوعی.

    Args:
        force: اجبار به بروزرسانی حتی اگر کش معتبر باشد

    Returns:
        bool: نتیجه بروزرسانی
    """
    global mock_product_cache, last_cache_update

    # بررسی وضعیت فعلی کش
    if not force and mock_product_cache is not None and last_cache_update is not None:
        # اگر کمتر از 1 روز از آخرین بروزرسانی گذشته باشد، نیازی به بروزرسانی نیست
        if datetime.utcnow() - last_cache_update < timedelta(days=1):
            logger.info(
                "کش محصولات مصنوعی معتبر است و نیاز به بروزرسانی ندارد")
            return True

    try:
        # در حالت توسعه، فقط داده‌های مصنوعی را بازتولید می‌کنیم
        logger.info("بروزرسانی داده‌های مصنوعی محصولات WooCommerce")

        # تولید داده‌های مصنوعی جدید
        mock_product_cache = generate_mock_products(100)
        last_cache_update = datetime.utcnow()

        logger.info("بروزرسانی کش محصولات مصنوعی با موفقیت انجام شد")
        return True

    except Exception as e:
        logger.error(f"خطا در بروزرسانی کش محصولات مصنوعی: {str(e)}")
        return False


async def get_all_products() -> List[Dict[str, Any]]:
    """
    دریافت تمام محصولات از کش مصنوعی.

    Returns:
        list: لیست محصولات
    """
    global mock_product_cache, last_cache_update

    # اگر کش موجود نیست، آن را راه‌اندازی می‌کنیم
    if mock_product_cache is None:
        await initialize_product_cache()

    # اگر کش قدیمی است (بیش از 1 روز)، آن را بروزرسانی می‌کنیم
    if last_cache_update is None or datetime.utcnow() - last_cache_update > timedelta(days=1):
        await refresh_product_cache()

    return mock_product_cache


def is_eyeglass_frame(product: Dict[str, Any]) -> bool:
    """
    بررسی اینکه آیا محصول یک فریم عینک است.

    Args:
        product: محصول

    Returns:
        bool: True اگر محصول فریم عینک باشد
    """
    # در داده‌های مصنوعی، همه محصولات فریم عینک هستند
    return product.get("is_eyeglass_frame", True)


def get_frame_type(product: Dict[str, Any]) -> str:
    """
    استخراج نوع فریم از محصول.

    Args:
        product: محصول

    Returns:
        str: نوع فریم
    """
    # اگر قبلاً نوع فریم مشخص شده، از آن استفاده می‌کنیم
    if "frame_type" in product:
        return product["frame_type"]

    # بررسی ویژگی‌های محصول
    attributes = product.get("attributes", [])

    frame_type_attrs = ["شکل فریم", "نوع فریم",
                        "فرم فریم", "frame type", "frame shape"]

    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()

        # بررسی اینکه آیا این ویژگی مربوط به نوع فریم است
        is_frame_type_attr = any(frame_type in attr_name.lower()
                                 for frame_type in frame_type_attrs)

        if is_frame_type_attr:
            # دریافت مقدار ویژگی
            if "options" in attribute and attribute["options"]:
                # برگرداندن اولین گزینه
                return attribute["options"][0]

    # اگر نوع فریم خاصی مشخص نشده، یک مقدار پیش‌فرض برمی‌گردانیم
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
    # بررسی معتبر بودن شکل چهره
    valid_shapes = {"HEART", "OBLONG", "OVAL", "ROUND", "SQUARE"}
    if face_shape not in valid_shapes:
        logger.warning(
            f"شکل چهره {face_shape} معتبر نیست. استفاده از OVAL به عنوان پیش‌فرض.")
        face_shape = "OVAL"

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
            price_str = product.get("price", "0")
            price = float(price_str) if price_str else 0

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

    # فیلتر کردن فریم‌های عینک (در داده‌های مصنوعی همه فریم عینک هستند)
    eyeglass_frames = [
        product for product in products if is_eyeglass_frame(product)]

    # فیلتر بر اساس قیمت (اگر درخواست شده باشد)
    if min_price is not None or max_price is not None:
        eyeglass_frames = filter_products_by_price(
            eyeglass_frames, min_price, max_price)

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
        eyeglass_frames = await get_eyeglass_frames(min_price, max_price)

        if not eyeglass_frames:
            logger.error("خطا در دریافت فریم‌های عینک از کش")
            return {
                "success": False,
                "message": "خطا در دریافت فریم‌های موجود"
            }

        # مرتب‌سازی بر اساس امتیاز تطابق
        sorted_frames = sort_products_by_match_score(
            eyeglass_frames, face_shape)

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

        logger.info(
            f"تطبیق فریم کامل شد: {len(recommended_frames)} توصیه پیدا شد")

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

    logger.warning(f"محصول با شناسه {product_id} در کش مصنوعی یافت نشد")
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
    global mock_product_cache, last_cache_update

    return {
        "cache_initialized": mock_product_cache is not None,
        "total_products": len(mock_product_cache) if mock_product_cache else 0,
        "last_update": last_cache_update,
        "update_in_progress": False,
        "last_error": None,
        "is_mock_data": True
    }
