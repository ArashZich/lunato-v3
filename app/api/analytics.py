from fastapi import APIRouter, Depends, HTTPException, Request, Query
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from app.db.repository import get_analytics_summary, get_detailed_analytics, get_time_based_analytics
from app.models.database import AnalyticsSummary, DetailedAnalytics, TimeBasedAnalytics
from app.db.connection import get_database
from app.config import settings

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# تعریف روتر
router = APIRouter()


@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary_api(
    period: Optional[str] = Query("all", description="دوره زمانی (today, week, month, all)")
):
    """
    دریافت خلاصه اطلاعات تحلیلی سیستم.
    
    این API اطلاعات کلی در مورد تحلیل‌های انجام شده، شکل‌های چهره، و دستگاه‌های کاربران را ارائه می‌دهد.
    
    پارامترها:
        - period: دوره زمانی (today: امروز، week: هفته اخیر، month: ماه اخیر، all: تمام زمان‌ها)
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
        
        # دریافت خلاصه اطلاعات تحلیلی
        summary = await get_analytics_summary(start_date)
        
        # بازگرداندن نتیجه
        return AnalyticsSummary(**summary)
        
    except Exception as e:
        logger.error(f"خطا در دریافت خلاصه اطلاعات تحلیلی: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در دریافت اطلاعات تحلیلی: {str(e)}")


@router.get("/analytics/detailed", response_model=DetailedAnalytics)
async def get_detailed_analytics_api(
    period: Optional[str] = Query("all", description="دوره زمانی (today, week, month, all)"),
    skip: int = Query(0, description="تعداد رکوردهای نادیده گرفته شده (برای صفحه‌بندی)"),
    limit: int = Query(100, description="حداکثر تعداد رکوردهای برگشتی")
):
    """
    دریافت اطلاعات تحلیلی تفصیلی سیستم.
    
    این API اطلاعات جزئی‌تر در مورد تحلیل‌های انجام شده و پیشنهادات ارائه می‌دهد.
    
    پارامترها:
        - period: دوره زمانی (today: امروز، week: هفته اخیر، month: ماه اخیر، all: تمام زمان‌ها)
        - skip: تعداد رکوردهای نادیده گرفته شده (برای صفحه‌بندی)
        - limit: حداکثر تعداد رکوردهای برگشتی
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
        
        # دریافت اطلاعات تحلیلی تفصیلی
        detailed = await get_detailed_analytics(start_date, skip, limit)
        
        # بازگرداندن نتیجه
        return DetailedAnalytics(**detailed)
        
    except Exception as e:
        logger.error(f"خطا در دریافت اطلاعات تحلیلی تفصیلی: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در دریافت اطلاعات تحلیلی: {str(e)}")


@router.get("/analytics/time-based", response_model=TimeBasedAnalytics)
async def get_time_based_analytics_api(
    group_by: str = Query("day", description="گروه‌بندی بر اساس زمان (hour, day, week, month)"),
    period: Optional[str] = Query("month", description="دوره زمانی (today, week, month, all)"),
    face_shape: Optional[str] = Query(None, description="فیلتر بر اساس شکل چهره")
):
    """
    دریافت اطلاعات تحلیلی بر اساس زمان.
    
    این API اطلاعات تحلیلی را بر اساس زمان گروه‌بندی می‌کند.
    
    پارامترها:
        - group_by: نحوه گروه‌بندی بر اساس زمان (hour: ساعت، day: روز، week: هفته، month: ماه)
        - period: دوره زمانی (today: امروز، week: هفته اخیر، month: ماه اخیر، all: تمام زمان‌ها)
        - face_shape: فیلتر بر اساس شکل چهره (اختیاری)
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
        
        # دریافت اطلاعات تحلیلی بر اساس زمان
        time_based = await get_time_based_analytics(group_by, start_date, face_shape)
        
        # بازگرداندن نتیجه
        return TimeBasedAnalytics(**time_based)
        
    except Exception as e:
        logger.error(f"خطا در دریافت اطلاعات تحلیلی بر اساس زمان: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در دریافت اطلاعات تحلیلی: {str(e)}")


@router.get("/analytics/frame-popularity")
async def get_frame_popularity_api(
    period: Optional[str] = Query("month", description="دوره زمانی (today, week, month, all)"),
    limit: int = Query(10, description="تعداد فریم‌های محبوب برای نمایش")
):
    """
    دریافت محبوب‌ترین فریم‌ها بر اساس پیشنهادهای ارائه شده.
    
    پارامترها:
        - period: دوره زمانی (today: امروز، week: هفته اخیر، month: ماه اخیر، all: تمام زمان‌ها)
        - limit: تعداد فریم‌های محبوب برای نمایش
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
        raise HTTPException(status_code=500, detail=f"خطا در دریافت محبوب‌ترین فریم‌ها: {str(e)}")


@router.get("/analytics/conversion-stats")
async def get_conversion_stats_api(
    period: Optional[str] = Query("month", description="دوره زمانی (today, week, month, all)")
):
    """
    دریافت آمار تبدیل (نسبت درخواست‌های موفق به کل درخواست‌ها).
    
    پارامترها:
        - period: دوره زمانی (today: امروز، week: هفته اخیر، month: ماه اخیر، all: تمام زمان‌ها)
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
        raise HTTPException(status_code=500, detail=f"خطا در دریافت آمار تبدیل: {str(e)}")