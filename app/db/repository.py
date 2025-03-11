import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson.objectid import ObjectId
import uuid
import asyncio

from app.db.connection import get_database
from app.config import settings

# تنظیمات لاگر
logger = logging.getLogger(__name__)


async def save_request_info(
    path: str,
    method: str,
    client_info: Dict[str, Any],
    status_code: int,
    process_time: float,
    request_id: str
) -> bool:
    """
    ذخیره اطلاعات درخواست در دیتابیس.
    
    Args:
        path: مسیر درخواست
        method: متد HTTP
        client_info: اطلاعات کاربر
        status_code: کد وضعیت پاسخ
        process_time: زمان پردازش بر حسب ثانیه
        request_id: شناسه منحصر به فرد درخواست
        
    Returns:
        bool: نتیجه عملیات ذخیره‌سازی
    """
    # بررسی وضعیت ذخیره‌سازی اطلاعات تحلیلی
    if not settings.STORE_ANALYTICS:
        return False
    
    try:
        db = get_database()
        
        # ساخت داده درخواست
        request_data = {
            "request_id": request_id,
            "path": path,
            "method": method,
            "client_info": client_info,
            "status_code": status_code,
            "process_time": process_time,
            "created_at": datetime.utcnow()
        }
        
        # ذخیره در کالکشن درخواست‌ها
        await db.requests.insert_one(request_data)
        
        return True
        
    except Exception as e:
        logger.error(f"خطا در ذخیره اطلاعات درخواست: {e}")
        return False


async def save_analysis_result(
    user_id: str,
    request_id: str,
    face_shape: str,
    confidence: float,
    client_info: Dict[str, Any],
    task_id: Optional[str] = None
) -> str:
    """
    ذخیره نتیجه تحلیل چهره در دیتابیس.
    
    Args:
        user_id: شناسه کاربر
        request_id: شناسه درخواست
        face_shape: شکل چهره
        confidence: میزان اطمینان
        client_info: اطلاعات کاربر
        task_id: شناسه وظیفه Celery (اختیاری)
        
    Returns:
        str: شناسه رکورد ایجاد شده
    """
    # بررسی وضعیت ذخیره‌سازی اطلاعات تحلیلی
    if not settings.STORE_ANALYTICS:
        return str(uuid.uuid4())
    
    try:
        db = get_database()
        
        # ساخت داده تحلیل
        analysis_data = {
            "user_id": user_id,
            "request_id": request_id,
            "face_shape": face_shape,
            "confidence": confidence,
            "client_info": client_info,
            "task_id": task_id,
            "created_at": datetime.utcnow()
        }
        
        # ذخیره در کالکشن نتایج تحلیل
        result = await db.analysis_results.insert_one(analysis_data)
        
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"خطا در ذخیره نتیجه تحلیل: {e}")
        return str(uuid.uuid4())


async def save_recommendation(
    user_id: str,
    face_shape: str,
    recommended_frame_types: List[str],
    recommended_frames: List[Dict[str, Any]],
    client_info: Dict[str, Any],
    analysis_id: Optional[str] = None
) -> str:
    """
    ذخیره پیشنهادات فریم در دیتابیس.
    """
    # بررسی وضعیت ذخیره‌سازی اطلاعات تحلیلی
    if not settings.STORE_ANALYTICS:
        return str(uuid.uuid4())
    
    try:
        db = get_database()
        
        # تبدیل مدل‌های Pydantic به دیکشنری
        frame_data = []
        for frame in recommended_frames:
            try:
                # اگر شیء دارای متد dict است از آن استفاده کنیم
                if hasattr(frame, 'dict'):
                    frame_dict = frame.dict()
                # اگر یک دیکشنری است، از آن به طور مستقیم استفاده کنیم
                elif isinstance(frame, dict):
                    frame_dict = frame
                else:
                    # تبدیل دستی به دیکشنری اگر نوع دیگری باشد
                    frame_dict = {
                        "id": getattr(frame, "id", None),
                        "name": getattr(frame, "name", ""),
                        "frame_type": getattr(frame, "frame_type", ""),
                        "match_score": getattr(frame, "match_score", 0)
                    }
                
                frame_data.append(frame_dict)
            except Exception as e:
                logger.warning(f"خطا در تبدیل فریم برای ذخیره در دیتابیس: {str(e)}")
                continue
        
        # ساخت داده پیشنهاد
        recommendation_data = {
            "user_id": user_id,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommended_frames": frame_data,
            "client_info": client_info,
            "analysis_id": analysis_id,
            "created_at": datetime.utcnow()
        }
        
        # ذخیره در کالکشن پیشنهادات
        result = await db.recommendations.insert_one(recommendation_data)
        
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"خطا در ذخیره پیشنهادات: {str(e)}")
        return str(uuid.uuid4())


async def get_analytics_summary(start_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    دریافت خلاصه اطلاعات تحلیلی.
    
    Args:
        start_date: تاریخ شروع برای محدود کردن نتایج (اختیاری)
        
    Returns:
        dict: خلاصه اطلاعات تحلیلی
    """
    try:
        db = get_database()
        
        # تعیین دوره زمانی
        period = "all"
        if start_date:
            now = datetime.utcnow()
            delta = now - start_date
            if delta.days <= 1:
                period = "today"
            elif delta.days <= 7:
                period = "week"
            elif delta.days <= 31:
                period = "month"
        
        # شرط تطبیق تاریخ
        date_match = {"created_at": {"$gte": start_date}} if start_date else {}
        
        # تعداد کل درخواست‌ها
        total_requests = await db.requests.count_documents(date_match)
        
        # تعداد کاربران منحصر به فرد
        unique_users_pipeline = [
            {"$match": date_match},
            {"$group": {"_id": "$client_info.ip_address"}},
            {"$count": "count"}
        ]
        
        unique_users_result = await db.requests.aggregate(unique_users_pipeline).to_list(1)
        total_unique_users = unique_users_result[0]["count"] if unique_users_result else 0
        
        # تعداد تحلیل‌ها به تفکیک شکل چهره
        face_shapes = {}
        face_shapes_pipeline = [
            {"$match": date_match},
            {"$group": {"_id": "$face_shape", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        async for doc in db.analysis_results.aggregate(face_shapes_pipeline):
            if doc["_id"]:  # اطمینان از وجود مقدار
                face_shapes[doc["_id"]] = doc["count"]
        
        # تعداد به تفکیک دستگاه
        devices = {}
        devices_pipeline = [
            {"$match": date_match},
            {"$group": {"_id": "$client_info.device_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        async for doc in db.requests.aggregate(devices_pipeline):
            if doc["_id"]:  # اطمینان از وجود مقدار
                devices[doc["_id"]] = doc["count"]
        
        # تعداد به تفکیک مرورگر
        browsers = {}
        browsers_pipeline = [
            {"$match": date_match},
            {"$group": {"_id": "$client_info.browser_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        async for doc in db.requests.aggregate(browsers_pipeline):
            if doc["_id"]:  # اطمینان از وجود مقدار
                browsers[doc["_id"]] = doc["count"]
        
        # تعداد به تفکیک سیستم عامل
        os_pipeline = [
            {"$match": date_match},
            {"$group": {"_id": "$client_info.os_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        operating_systems = {}
        async for doc in db.requests.aggregate(os_pipeline):
            if doc["_id"]:  # اطمینان از وجود مقدار
                operating_systems[doc["_id"]] = doc["count"]
        
        # انواع فریم پیشنهادی
        frame_types = {}
        frame_types_pipeline = [
            {"$match": date_match},
            {"$unwind": "$recommended_frame_types"},
            {"$group": {"_id": "$recommended_frame_types", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        async for doc in db.recommendations.aggregate(frame_types_pipeline):
            if doc["_id"]:  # اطمینان از وجود مقدار
                frame_types[doc["_id"]] = doc["count"]
        
        # میانگین زمان پردازش
        avg_process_time_pipeline = [
            {"$match": date_match},
            {"$group": {"_id": None, "avg_time": {"$avg": "$process_time"}}}
        ]
        
        avg_process_time_result = await db.requests.aggregate(avg_process_time_pipeline).to_list(1)
        avg_process_time = round(avg_process_time_result[0]["avg_time"], 3) if avg_process_time_result else 0
        
        # آخرین درخواست
        latest_request_pipeline = [
            {"$match": date_match},
            {"$sort": {"created_at": -1}},
            {"$limit": 1},
            {"$project": {"_id": 0, "created_at": 1}}
        ]
        
        latest_request = await db.requests.aggregate(latest_request_pipeline).to_list(1)
        last_update_time = latest_request[0]["created_at"] if latest_request else None
        
        return {
            "total_requests": total_requests,
            "total_unique_users": total_unique_users,
            "face_shapes": face_shapes,
            "devices": devices,
            "browsers": browsers,
            "operating_systems": operating_systems,
            "frame_types": frame_types,
            "average_process_time": avg_process_time,
            "last_update_time": last_update_time,
            "period": period
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت خلاصه اطلاعات تحلیلی: {e}")
        return {
            "total_requests": 0,
            "total_unique_users": 0,
            "face_shapes": {},
            "devices": {},
            "browsers": {},
            "operating_systems": {},
            "frame_types": {},
            "average_process_time": 0,
            "last_update_time": None,
            "period": "all"
        }


async def get_detailed_analytics(
    start_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100
) -> Dict[str, Any]:
    """
    دریافت اطلاعات تحلیلی تفصیلی.
    
    Args:
        start_date: تاریخ شروع برای محدود کردن نتایج (اختیاری)
        skip: تعداد رکوردهای نادیده گرفته شده (برای صفحه‌بندی)
        limit: حداکثر تعداد رکوردهای برگشتی
        
    Returns:
        dict: اطلاعات تحلیلی تفصیلی
    """
    try:
        db = get_database()
        
        # تعیین دوره زمانی
        period = "all"
        if start_date:
            now = datetime.utcnow()
            delta = now - start_date
            if delta.days <= 1:
                period = "today"
            elif delta.days <= 7:
                period = "week"
            elif delta.days <= 31:
                period = "month"
        
        # شرط تطبیق تاریخ
        match_condition = {"created_at": {"$gte": start_date}} if start_date else {}
        
        # بدست آوردن تعداد کل رکوردها
        total = await db.analysis_results.count_documents(match_condition)
        
        # دریافت رکوردهای تحلیل
        pipeline = [
            {"$match": match_condition},
            {"$lookup": {
                "from": "recommendations",
                "localField": "_id",
                "foreignField": "analysis_id",
                "as": "recommendations"
            }},
            {"$project": {
                "_id": 0,
                "request_id": 1,
                "user_id": 1,
                "face_shape": 1,
                "confidence": 1,
                "created_at": 1,
                "client_info.device_type": 1,
                "client_info.browser_name": 1,
                "recommended_frame_types": {
                    "$ifNull": [{"$arrayElemAt": ["$recommendations.recommended_frame_types", 0]}, []]
                }
            }},
            {"$sort": {"created_at": -1}},
            {"$skip": skip},
            {"$limit": limit}
        ]
        
        items = []
        async for doc in db.analysis_results.aggregate(pipeline):
            # استخراج اطلاعات از نتیجه پایپلاین
            items.append({
                "request_id": doc.get("request_id", ""),
                "user_id": doc.get("user_id", ""),
                "face_shape": doc.get("face_shape", ""),
                "confidence": doc.get("confidence", 0),
                "device_type": doc.get("client_info", {}).get("device_type", "unknown"),
                "browser_name": doc.get("client_info", {}).get("browser_name", None),
                "recommended_frame_types": doc.get("recommended_frame_types", []),
                "created_at": doc.get("created_at", datetime.utcnow())
            })
        
        return {
            "total": total,
            "period": period,
            "skip": skip,
            "limit": limit,
            "items": items
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت اطلاعات تحلیلی تفصیلی: {e}")
        return {
            "total": 0,
            "period": period,
            "skip": skip,
            "limit": limit,
            "items": []
        }


async def get_time_based_analytics(
    group_by: str = "day",
    start_date: Optional[datetime] = None,
    face_shape: Optional[str] = None
) -> Dict[str, Any]:
    """
    دریافت اطلاعات تحلیلی بر اساس زمان.
    
    Args:
        group_by: نحوه گروه‌بندی بر اساس زمان (hour, day, week, month)
        start_date: تاریخ شروع برای محدود کردن نتایج (اختیاری)
        face_shape: فیلتر بر اساس شکل چهره (اختیاری)
        
    Returns:
        dict: اطلاعات تحلیلی بر اساس زمان
    """
    try:
        db = get_database()
        
        # تعیین دوره زمانی
        period = "all"
        if start_date:
            now = datetime.utcnow()
            delta = now - start_date
            if delta.days <= 1:
                period = "today"
            elif delta.days <= 7:
                period = "week"
            elif delta.days <= 31:
                period = "month"
        
        # ساخت شرط فیلتر
        match_condition = {}
        if start_date:
            match_condition["created_at"] = {"$gte": start_date}
        if face_shape:
            match_condition["face_shape"] = face_shape
        
        # ساخت عبارت گروه‌بندی بر اساس زمان
        if group_by == "hour":
            time_format = "%Y-%m-%d %H:00"
            time_group = {
                "year": {"$year": "$created_at"},
                "month": {"$month": "$created_at"},
                "day": {"$dayOfMonth": "$created_at"},
                "hour": {"$hour": "$created_at"}
            }
        elif group_by == "day":
            time_format = "%Y-%m-%d"
            time_group = {
                "year": {"$year": "$created_at"},
                "month": {"$month": "$created_at"},
                "day": {"$dayOfMonth": "$created_at"}
            }
        elif group_by == "week":
            time_format = "Week %U, %Y"
            time_group = {
                "year": {"$year": "$created_at"},
                "week": {"$week": "$created_at"}
            }
        elif group_by == "month":
            time_format = "%Y-%m"
            time_group = {
                "year": {"$year": "$created_at"},
                "month": {"$month": "$created_at"}
            }
        else:
            # پیش‌فرض: گروه‌بندی روزانه
            group_by = "day"
            time_format = "%Y-%m-%d"
            time_group = {
                "year": {"$year": "$created_at"},
                "month": {"$month": "$created_at"},
                "day": {"$dayOfMonth": "$created_at"}
            }
        
        # پایپلاین اصلی برای گروه‌بندی زمانی
        pipeline = [
            {"$match": match_condition},
            {"$group": {
                "_id": time_group,
                "count": {"$sum": 1},
                "face_shapes": {"$push": "$face_shape"},
                "avg_confidence": {"$avg": "$confidence"}
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1, "_id.day": 1}}
        ]
        
        # اجرای پایپلاین و پردازش نتایج
        data_points = []
        async for doc in db.analysis_results.aggregate(pipeline):
            # ساخت رشته نمایش زمان
            time_id = doc["_id"]
            if group_by == "hour":
                time_str = f"{time_id['year']}-{time_id['month']:02d}-{time_id['day']:02d} {time_id['hour']:02d}:00"
            elif group_by == "day":
                time_str = f"{time_id['year']}-{time_id['month']:02d}-{time_id['day']:02d}"
            elif group_by == "week":
                time_str = f"Week {time_id.get('week', 0)}, {time_id['year']}"
            elif group_by == "month":
                time_str = f"{time_id['year']}-{time_id['month']:02d}"
            
            # محاسبه توزیع شکل چهره
            face_shapes_distribution = {}
            for shape in doc["face_shapes"]:
                if shape:
                    face_shapes_distribution[shape] = face_shapes_distribution.get(shape, 0) + 1
            
            # افزودن نقطه داده
            data_points.append({
                "time_period": time_str,
                "count": doc["count"],
                "face_shape_distribution": face_shapes_distribution,
                "avg_confidence": round(doc.get("avg_confidence", 0), 1)
            })
        
        return {
            "group_by": group_by,
            "period": period,
            "face_shape_filter": face_shape,
            "data_points": data_points
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت اطلاعات تحلیلی بر اساس زمان: {e}")
        return {
            "group_by": group_by,
            "period": period,
            "face_shape_filter": face_shape,
            "data_points": []
        }


async def get_popular_frames(period: str = "month", limit: int = 10) -> Dict[str, Any]:
    """
    دریافت فریم‌های محبوب بر اساس پیشنهادهای ارائه شده.
    
    Args:
        period: دوره زمانی (today, week, month, all)
        limit: تعداد فریم‌های محبوب برای نمایش
        
    Returns:
        dict: فریم‌های محبوب
    """
    try:
        # تبدیل دوره زمانی به تاریخ شروع
        start_date = None
        if period == "today":
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start_date = datetime.utcnow() - timedelta(days=7)
        elif period == "month":
            start_date = datetime.utcnow() - timedelta(days=30)
        
        # دریافت اطلاعات از دیتابیس
        db = get_database()
        
        pipeline = [
            # فیلتر بر اساس دوره زمانی
            {"$match": {"created_at": {"$gte": start_date}}} if start_date else {"$match": {}},
            # باز کردن آرایه فریم‌های پیشنهادی
            {"$unwind": "$recommended_frames"},
            # گروه‌بندی بر اساس شناسه فریم
            {"$group": {
                "_id": "$recommended_frames.id",
                "name": {"$first": "$recommended_frames.name"},
                "frame_type": {"$first": "$recommended_frames.frame_type"},
                "avg_match_score": {"$avg": "$recommended_frames.match_score"},
                "count": {"$sum": 1}
            }},
            # مرتب‌سازی بر اساس تعداد پیشنهاد
            {"$sort": {"count": -1}},
            # محدود کردن تعداد نتایج
            {"$limit": limit}
        ]
        
        if start_date is None:
            # حذف شرط تاریخ اگر همه زمان‌ها مورد نظر است
            pipeline = pipeline[1:]
        
        # اجرای پایپلاین
        popular_frames = []
        async for doc in db.recommendations.aggregate(pipeline):
            popular_frames.append({
                "id": doc["_id"],
                "name": doc["name"],
                "frame_type": doc["frame_type"],
                "avg_match_score": round(doc["avg_match_score"], 1),
                "recommendation_count": doc["count"]
            })
        
        return {
            "period": period,
            "popular_frames": popular_frames
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت محبوب‌ترین فریم‌ها: {str(e)}")
        return {
            "period": period,
            "popular_frames": []
        }


async def get_conversion_stats(period: str = "month") -> Dict[str, Any]:
    """
    دریافت آمار تبدیل (نسبت درخواست‌های موفق به کل درخواست‌ها).
    
    Args:
        period: دوره زمانی (today, week, month, all)
        
    Returns:
        dict: آمار تبدیل
    """
    try:
        # تبدیل دوره زمانی به تاریخ شروع
        start_date = None
        if period == "today":
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start_date = datetime.utcnow() - timedelta(days=7)
        elif period == "month":
            start_date = datetime.utcnow() - timedelta(days=30)
        
        # دریافت اطلاعات از دیتابیس
        db = get_database()
        
        # شرط تاریخ
        date_match = {"created_at": {"$gte": start_date}} if start_date else {}
        
        # تعداد کل درخواست‌ها
        total_requests = await db.requests.count_documents(date_match)
        
        # تعداد درخواست‌های موفق (کد وضعیت 200)
        success_requests = await db.requests.count_documents({
            **date_match,
            "status_code": 200
        })
        
        # تعداد تحلیل‌های موفق چهره
        successful_analyses = await db.analysis_results.count_documents(date_match)
        
        # تعداد پیشنهادات ارائه شده
        total_recommendations = await db.recommendations.count_documents(date_match)
        
        # محاسبه نرخ‌های تبدیل
        success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 0
        analysis_to_request_ratio = (successful_analyses / total_requests * 100) if total_requests > 0 else 0
        recommendation_to_analysis_ratio = (total_recommendations / successful_analyses * 100) if successful_analyses > 0 else 0
        
        return {
            "period": period,
            "total_requests": total_requests,
            "successful_requests": success_requests,
            "success_rate": round(success_rate, 1),
            "successful_analyses": successful_analyses,
            "analysis_to_request_ratio": round(analysis_to_request_ratio, 1),
            "total_recommendations": total_recommendations,
            "recommendation_to_analysis_ratio": round(recommendation_to_analysis_ratio, 1)
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت آمار تبدیل: {str(e)}")
        return {
            "period": period,
            "total_requests": 0,
            "successful_requests": 0,
            "success_rate": 0,
            "successful_analyses": 0,
            "analysis_to_request_ratio": 0,
            "total_recommendations": 0,
            "recommendation_to_analysis_ratio": 0
        }


async def save_woocommerce_cache(data: List[Dict[str, Any]], last_update: datetime) -> bool:
    """
    ذخیره کش محصولات WooCommerce در دیتابیس.
    
    Args:
        data: لیست محصولات
        last_update: زمان آخرین بروزرسانی
        
    Returns:
        bool: نتیجه عملیات ذخیره‌سازی
    """
    try:
        db = get_database()
        
        # حذف رکورد قبلی
        await db.woocommerce_cache.delete_many({"type": "products_cache"})
        
        # افزودن رکورد جدید
        await db.woocommerce_cache.insert_one({
            "type": "products_cache",
            "last_update": last_update,
            "data": data
        })
        
        logger.info(f"کش محصولات WooCommerce در دیتابیس بروزرسانی شد ({len(data)} محصول)")
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
        
        # دریافت رکورد کش
        cache_record = await db.woocommerce_cache.find_one({"type": "products_cache"})
        
        if cache_record and "data" in cache_record and "last_update" in cache_record:
            return cache_record["data"], cache_record["last_update"]
        
        return None, None
        
    except Exception as e:
        logger.error(f"خطا در دریافت کش محصولات از دیتابیس: {str(e)}")
        return None, None


async def check_and_update_request_analytics():
    """
    بررسی و بروزرسانی داده‌های تحلیلی درخواست‌ها.
    این تابع برای استفاده در زمان اجرای سیستم به صورت دوره‌ای فراخوانی می‌شود.
    """
    try:
        db = get_database()
        
        # بررسی درخواست‌های فاقد زمان پردازش
        missing_time_count = await db.requests.count_documents({"process_time": {"$exists": False}})
        
        if missing_time_count > 0:
            logger.info(f"تعداد {missing_time_count} درخواست فاقد زمان پردازش یافت شد. در حال بروزرسانی...")
            
            # بروزرسانی با مقدار پیش‌فرض
            await db.requests.update_many(
                {"process_time": {"$exists": False}},
                {"$set": {"process_time": 0.5}}  # مقدار پیش‌فرض معقول
            )
        
        # بررسی درخواست‌های فاقد client_info
        missing_client_info_count = await db.requests.count_documents({"client_info": {"$exists": False}})
        
        if missing_client_info_count > 0:
            logger.info(f"تعداد {missing_client_info_count} درخواست فاقد client_info یافت شد. در حال بروزرسانی...")
            
            # بروزرسانی با مقدار پیش‌فرض
            await db.requests.update_many(
                {"client_info": {"$exists": False}},
                {"$set": {"client_info": {
                    "device_type": "unknown",
                    "browser_name": "unknown",
                    "os_name": "unknown"
                }}}
            )
        
        # بررسی درخواست‌های فاقد created_at
        missing_created_at_count = await db.requests.count_documents({"created_at": {"$exists": False}})
        
        if missing_created_at_count > 0:
            logger.info(f"تعداد {missing_created_at_count} درخواست فاقد created_at یافت شد. در حال بروزرسانی...")
            
            # بروزرسانی با زمان فعلی
            await db.requests.update_many(
                {"created_at": {"$exists": False}},
                {"$set": {"created_at": datetime.utcnow()}}
            )
        
        return True
        
    except Exception as e:
        logger.error(f"خطا در بررسی و بروزرسانی داده‌های تحلیلی: {str(e)}")
        return False


async def get_frame_recommendations_by_face_shape() -> Dict[str, List[Dict[str, Any]]]:
    """
    دریافت پیشنهادات فریم به تفکیک شکل چهره.
    
    Returns:
        dict: پیشنهادات فریم به تفکیک شکل چهره
    """
    try:
        db = get_database()
        
        # دریافت شکل‌های چهره موجود
        face_shapes_pipeline = [
            {"$group": {"_id": "$face_shape"}},
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"_id": 1}}
        ]
        
        face_shapes = []
        async for doc in db.analysis_results.aggregate(face_shapes_pipeline):
            face_shapes.append(doc["_id"])
        
        # دریافت پیشنهادات فریم برای هر شکل چهره
        results = {}
        
        for shape in face_shapes:
            # دریافت فریم‌های پیشنهادی برای این شکل چهره
            pipeline = [
                {"$match": {"face_shape": shape}},
                {"$unwind": "$recommended_frames"},
                {"$group": {
                    "_id": "$recommended_frames.id",
                    "name": {"$first": "$recommended_frames.name"},
                    "frame_type": {"$first": "$recommended_frames.frame_type"},
                    "avg_match_score": {"$avg": "$recommended_frames.match_score"},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"avg_match_score": -1}},
                {"$limit": 5}
            ]
            
            recommendations = []
            async for doc in db.recommendations.aggregate(pipeline):
                recommendations.append({
                    "id": doc["_id"],
                    "name": doc["name"],
                    "frame_type": doc["frame_type"],
                    "avg_match_score": round(doc["avg_match_score"], 1),
                    "count": doc["count"]
                })
            
            if recommendations:
                results[shape] = recommendations
        
        return results
        
    except Exception as e:
        logger.error(f"خطا در دریافت پیشنهادات فریم به تفکیک شکل چهره: {str(e)}")
        return {}


async def create_database_indexes():
    """
    ایجاد ایندکس‌های مورد نیاز در پایگاه داده.
    این تابع در زمان راه‌اندازی سیستم فراخوانی می‌شود.
    """
    try:
        db = get_database()
        
        # ایندکس برای کالکشن درخواست‌ها
        await db.requests.create_index("request_id", unique=True)
        await db.requests.create_index("created_at")
        await db.requests.create_index("client_info.device_type")
        await db.requests.create_index("client_info.browser_name")
        await db.requests.create_index("client_info.os_name")
        await db.requests.create_index("status_code")
        
        # ایندکس برای کالکشن نتایج تحلیل
        await db.analysis_results.create_index("user_id")
        await db.analysis_results.create_index("request_id")
        await db.analysis_results.create_index("face_shape")
        await db.analysis_results.create_index("created_at")
        await db.analysis_results.create_index([("confidence", -1)])
        
        # ایندکس برای کالکشن پیشنهادات
        await db.recommendations.create_index("user_id")
        await db.recommendations.create_index("face_shape")
        await db.recommendations.create_index("analysis_id")
        await db.recommendations.create_index("created_at")
        await db.recommendations.create_index([("recommended_frames.match_score", -1)])
        
        # ایندکس برای کالکشن کش محصولات WooCommerce
        await db.woocommerce_cache.create_index("type", unique=True)
        await db.woocommerce_cache.create_index("last_update")
        
        logger.info("ایندکس‌های دیتابیس با موفقیت ایجاد شدند")
        return True
        
    except Exception as e:
        logger.error(f"خطا در ایجاد ایندکس‌های دیتابیس: {str(e)}")
        return False


async def get_face_shape_distribution_by_device():
    """
    دریافت توزیع شکل چهره به تفکیک نوع دستگاه.
    
    Returns:
        dict: توزیع شکل چهره به تفکیک نوع دستگاه
    """
    try:
        db = get_database()
        
        pipeline = [
            {"$match": {"client_info.device_type": {"$exists": True}, "face_shape": {"$exists": True}}},
            {"$group": {
                "_id": {
                    "device_type": "$client_info.device_type",
                    "face_shape": "$face_shape"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        
        # اجرای پایپلاین
        results = {}
        async for doc in db.analysis_results.aggregate(pipeline):
            device_type = doc["_id"]["device_type"]
            face_shape = doc["_id"]["face_shape"]
            count = doc["count"]
            
            if device_type not in results:
                results[device_type] = {}
            
            results[device_type][face_shape] = count
        
        return results
        
    except Exception as e:
        logger.error(f"خطا در دریافت توزیع شکل چهره به تفکیک نوع دستگاه: {str(e)}")
        return {}


async def get_confidence_stats_by_face_shape():
    """
    دریافت آمار میزان اطمینان به تفکیک شکل چهره.
    
    Returns:
        dict: آمار میزان اطمینان به تفکیک شکل چهره
    """
    try:
        db = get_database()
        
        pipeline = [
            {"$match": {"face_shape": {"$exists": True}, "confidence": {"$exists": True}}},
            {"$group": {
                "_id": "$face_shape",
                "avg_confidence": {"$avg": "$confidence"},
                "min_confidence": {"$min": "$confidence"},
                "max_confidence": {"$max": "$confidence"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        
        # اجرای پایپلاین
        results = {}
        async for doc in db.analysis_results.aggregate(pipeline):
            face_shape = doc["_id"]
            results[face_shape] = {
                "avg_confidence": round(doc["avg_confidence"], 1),
                "min_confidence": round(doc["min_confidence"], 1),
                "max_confidence": round(doc["max_confidence"], 1),
                "count": doc["count"]
            }
        
        return results
        
    except Exception as e:
        logger.error(f"خطا در دریافت آمار میزان اطمینان به تفکیک شکل چهره: {str(e)}")
        return {}


async def clear_old_analytics_data(days: int = 90):
    """
    حذف داده‌های تحلیلی قدیمی.
    
    Args:
        days: تعداد روزهای حفظ داده‌ها (پیش‌فرض: 90 روز)
        
    Returns:
        dict: نتیجه عملیات حذف
    """
    try:
        db = get_database()
        
        # تاریخ مرزی برای حذف
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # حذف درخواست‌های قدیمی
        requests_result = await db.requests.delete_many({"created_at": {"$lt": cutoff_date}})
        
        # حذف نتایج تحلیل قدیمی
        analysis_result = await db.analysis_results.delete_many({"created_at": {"$lt": cutoff_date}})
        
        # حذف پیشنهادات قدیمی
        recommendations_result = await db.recommendations.delete_many({"created_at": {"$lt": cutoff_date}})
        
        logger.info(f"داده‌های تحلیلی قدیمی‌تر از {cutoff_date} حذف شدند")
        
        return {
            "deleted_requests": requests_result.deleted_count,
            "deleted_analyses": analysis_result.deleted_count,
            "deleted_recommendations": recommendations_result.deleted_count,
            "cutoff_date": cutoff_date
        }
        
    except Exception as e:
        logger.error(f"خطا در حذف داده‌های تحلیلی قدیمی: {str(e)}")
        return {
            "deleted_requests": 0,
            "deleted_analyses": 0,
            "deleted_recommendations": 0,
            "error": str(e)
        }