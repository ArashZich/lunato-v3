from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class FaceAnalysisRequest(BaseModel):
    """
    مدل درخواست برای تحلیل چهره.
    """
    image: Optional[str] = Field(None, description="تصویر به صورت base64 encoded (اختیاری)")
    include_frames: bool = Field(True, description="آیا فریم‌های پیشنهادی در پاسخ گنجانده شود")
    min_price: Optional[float] = Field(None, description="حداقل قیمت (تومان)")
    max_price: Optional[float] = Field(None, description="حداکثر قیمت (تومان)")
    limit: Optional[int] = Field(10, description="حداکثر تعداد فریم‌های پیشنهادی")
    async_process: bool = Field(False, description="پردازش به صورت غیرهمزمان انجام شود")
    user_id: Optional[str] = Field(None, description="شناسه کاربر (اختیاری)")
    
    class Config:
        schema_extra = {
            "example": {
                "include_frames": True,
                "min_price": 1000000,
                "max_price": 5000000,
                "limit": 5,
                "async_process": False
            }
        }


class FrameRecommendationRequest(BaseModel):
    """
    مدل درخواست برای پیشنهاد فریم بر اساس شکل چهره.
    """
    face_shape: str = Field(..., description="شکل چهره (مانند OVAL, ROUND, و غیره)")
    min_price: Optional[float] = Field(None, description="حداقل قیمت (تومان)")
    max_price: Optional[float] = Field(None, description="حداکثر قیمت (تومان)")
    limit: Optional[int] = Field(10, description="حداکثر تعداد فریم‌های پیشنهادی")
    user_id: Optional[str] = Field(None, description="شناسه کاربر (اختیاری)")
    
    class Config:
        schema_extra = {
            "example": {
                "face_shape": "OVAL",
                "min_price": 1000000,
                "max_price": 5000000,
                "limit": 5
            }
        }