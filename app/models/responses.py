from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ClientInfo(BaseModel):
    """اطلاعات دستگاه و مرورگر کاربر"""
    device_type: str = Field(..., description="نوع دستگاه (mobile, tablet, desktop)")
    os_name: Optional[str] = Field(None, description="نام سیستم‌عامل")
    os_version: Optional[str] = Field(None, description="نسخه سیستم‌عامل")
    browser_name: Optional[str] = Field(None, description="نام مرورگر")
    browser_version: Optional[str] = Field(None, description="نسخه مرورگر")
    ip_address: Optional[str] = Field(None, description="آدرس IP")
    language: Optional[str] = Field(None, description="زبان مرورگر")


class BaseResponse(BaseModel):
    """پاسخ پایه برای تمام APIs"""
    success: bool = Field(..., description="وضعیت موفقیت درخواست")
    message: str = Field(..., description="پیام توضیحی")


class FaceCoordinates(BaseModel):
    """مختصات چهره در تصویر"""
    x: int = Field(..., description="مختصات X گوشه بالا چپ")
    y: int = Field(..., description="مختصات Y گوشه بالا چپ")
    width: int = Field(..., description="عرض چهره")
    height: int = Field(..., description="ارتفاع چهره")
    center_x: int = Field(..., description="مختصات X مرکز چهره")
    center_y: int = Field(..., description="مختصات Y مرکز چهره")
    aspect_ratio: float = Field(..., description="نسبت ابعاد (عرض/ارتفاع)")


class RecommendedFrame(BaseModel):
    """مدل فریم پیشنهادی"""
    id: int = Field(..., description="شناسه محصول")
    name: str = Field(..., description="نام محصول")
    permalink: str = Field(..., description="لینک محصول")
    price: str = Field(..., description="قیمت محصول")
    regular_price: Optional[str] = Field(None, description="قیمت عادی محصول (قبل از تخفیف)")
    frame_type: str = Field(..., description="نوع/شکل فریم")
    images: List[str] = Field(..., description="تصاویر محصول")
    match_score: float = Field(..., description="امتیاز تطابق با شکل صورت")


class FaceAnalysisResponse(BaseResponse):
    """پاسخ تحلیل چهره و پیشنهاد فریم"""
    face_coordinates: Optional[FaceCoordinates] = Field(None, description="مختصات چهره")
    face_shape: Optional[str] = Field(None, description="شکل تشخیص داده شده چهره")
    confidence: Optional[float] = Field(None, description="میزان اطمینان تشخیص")
    description: Optional[str] = Field(None, description="توضیحات مربوط به شکل چهره")
    recommendation: Optional[str] = Field(None, description="توصیه‌های مربوط به فریم مناسب")
    client_info: Optional[ClientInfo] = Field(None, description="اطلاعات دستگاه و مرورگر کاربر")
    recommended_frame_types: Optional[List[str]] = Field(None, description="انواع فریم‌های پیشنهادی")
    recommended_frames: Optional[List[RecommendedFrame]] = Field(None, description="فریم‌های پیشنهادی")
    task_id: Optional[str] = Field(None, description="شناسه وظیفه در صورت پردازش غیرهمزمان")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "تحلیل چهره با موفقیت انجام شد",
                "face_coordinates": {
                    "x": 100,
                    "y": 80,
                    "width": 200,
                    "height": 220,
                    "center_x": 200,
                    "center_y": 190,
                    "aspect_ratio": 0.91
                },
                "face_shape": "OVAL",
                "confidence": 85.5,
                "description": "صورت بیضی متعادل‌ترین شکل صورت است. پهنای گونه‌ها با پهنای پیشانی و فک متناسب است.",
                "recommendation": "اکثر فریم‌ها برای این نوع صورت مناسب هستند، اما فریم‌های مستطیلی و مربعی بهترین گزینه هستند.",
                "client_info": {
                    "device_type": "mobile",
                    "os_name": "iOS",
                    "os_version": "15.0",
                    "browser_name": "Safari",
                    "browser_version": "15.0",
                    "ip_address": "192.168.1.1",
                    "language": "fa-IR"
                },
                "recommended_frame_types": ["مستطیلی", "مربعی", "هشت‌ضلعی"],
                "recommended_frames": [
                    {
                        "id": 123,
                        "name": "فریم طبی مدل مستطیلی کلاسیک",
                        "permalink": "https://lunato.shop/product/classic-rectangular",
                        "price": "2500000",
                        "regular_price": "3000000",
                        "frame_type": "مستطیلی",
                        "images": ["https://lunato.shop/wp-content/uploads/frame1.jpg"],
                        "match_score": 92.5
                    }
                ]
            }
        }


class FrameRecommendationResponse(BaseResponse):
    """پاسخ پیشنهاد فریم بر اساس شکل صورت"""
    face_shape: str = Field(..., description="شکل صورت مورد استفاده برای پیشنهادات")
    recommended_frame_types: List[str] = Field(..., description="انواع فریم‌های پیشنهادی")
    recommended_frames: List[RecommendedFrame] = Field(..., description="فریم‌های پیشنهادی")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "پیشنهادات فریم با موفقیت ایجاد شد",
                "face_shape": "OVAL",
                "recommended_frame_types": ["مستطیلی", "مربعی", "هشت‌ضلعی"],
                "recommended_frames": [
                    {
                        "id": 123,
                        "name": "فریم طبی مدل مستطیلی کلاسیک",
                        "permalink": "https://lunato.shop/product/classic-rectangular",
                        "price": "2500000",
                        "regular_price": "3000000",
                        "frame_type": "مستطیلی",
                        "images": ["https://lunato.shop/wp-content/uploads/frame1.jpg"],
                        "match_score": 92.5
                    }
                ]
            }
        }


class HealthResponse(BaseResponse):
    """پاسخ برای endpoint سلامت سیستم"""
    version: str = Field(..., description="نسخه API")
    environment: str = Field(..., description="محیط اجرا")
    timestamp: datetime = Field(..., description="زمان پاسخ")
    database_status: Optional[str] = Field(None, description="وضعیت اتصال به دیتابیس")
    celery_status: Optional[str] = Field(None, description="وضعیت سرویس Celery")