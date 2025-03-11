# app/services/tasks.py
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from celery import shared_task

from app.celery_app import app
from app.config import settings
from app.db.repository import save_analysis_result, save_recommendation
from app.utils.image_processing import base64_to_opencv
from app.core.face_detection import detect_face

# حذف این واردسازی‌ها از سطح بالای فایل
# from app.core.face_analysis import analyze_face_shape, get_recommended_frame_types
# from app.services.classifier import predict_face_shape
# from app.services.woocommerce import get_recommended_frames

# تنظیمات لاگر
logger = logging.getLogger(__name__)


@shared_task(name="app.services.tasks.detect_face")
def detect_face_task(image_data: str, user_id: str, request_id: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    وظیفه تشخیص چهره در تصویر.
    
    Args:
        image_data: تصویر به صورت base64
        user_id: شناسه کاربر
        request_id: شناسه درخواست
        client_info: اطلاعات کاربر
        
    Returns:
        dict: نتیجه تشخیص چهره
    """
    try:
        logger.info(f"شروع تشخیص چهره برای درخواست {request_id}")
        
        # تبدیل تصویر به فرمت OpenCV
        image = base64_to_opencv(image_data)
        
        if image is None:
            logger.error(f"خطا در تبدیل تصویر برای درخواست {request_id}")
            return {
                "success": False,
                "message": "تصویر نامعتبر است",
                "request_id": request_id
            }
        
        # تشخیص چهره
        detection_result = detect_face(image)
        
        if not detection_result.get("success", False):
            logger.warning(f"تشخیص چهره ناموفق بود برای درخواست {request_id}: {detection_result.get('message')}")
            return {
                "success": False,
                "message": detection_result.get("message", "خطا در تشخیص چهره"),
                "request_id": request_id
            }
        
        # نتیجه موفقیت‌آمیز
        result = {
            "success": True,
            "face": detection_result.get("face"),
            "message": "چهره با موفقیت تشخیص داده شد",
            "request_id": request_id,
            "user_id": user_id
        }
        
        # ارسال وظیفه بعدی برای تحلیل چهره
        analyze_face_shape_task.delay(
            image_data=image_data,
            face_coordinates=detection_result.get("face"),
            user_id=user_id,
            request_id=request_id,
            client_info=client_info
        )
        
        return result
        
    except Exception as e:
        logger.error(f"خطا در تشخیص چهره: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تشخیص چهره: {str(e)}",
            "request_id": request_id
        }


@shared_task(name="app.services.tasks.analyze_face_shape")
async def analyze_face_shape_task(image_data: str, face_coordinates: Dict[str, int], user_id: str, request_id: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    وظیفه تحلیل شکل چهره.
    
    Args:
        image_data: تصویر به صورت base64
        face_coordinates: مختصات چهره در تصویر
        user_id: شناسه کاربر
        request_id: شناسه درخواست
        client_info: اطلاعات کاربر
        
    Returns:
        dict: نتیجه تحلیل شکل چهره
    """
    try:
        logger.info(f"شروع تحلیل شکل چهره برای درخواست {request_id}")
        
        # واردسازی تأخیری برای جلوگیری از واردسازی دایره‌ای
        from app.core.face_analysis import analyze_face_shape, get_recommended_frame_types
        from app.services.classifier import predict_face_shape
        
        # تبدیل تصویر به فرمت OpenCV
        image = base64_to_opencv(image_data)
        
        if image is None:
            logger.error(f"خطا در تبدیل تصویر برای درخواست {request_id}")
            return {
                "success": False,
                "message": "تصویر نامعتبر است",
                "request_id": request_id
            }
        
        # تحلیل شکل چهره با استفاده از مدل scikit-learn اگر موجود باشد
        try:
            face_shape, confidence, shape_details = predict_face_shape(image, face_coordinates)
            logger.info(f"شکل چهره با استفاده از مدل ML تشخیص داده شد: {face_shape}")
        except Exception as model_error:
            logger.warning(f"خطا در استفاده از مدل ML: {str(model_error)}. استفاده از روش قوانین...")
            # استفاده از تحلیل مبتنی بر قوانین به عنوان پشتیبان
            analysis_result = analyze_face_shape(image, face_coordinates)
            
            if not analysis_result.get("success", False):
                logger.warning(f"تحلیل شکل چهره ناموفق بود برای درخواست {request_id}: {analysis_result.get('message')}")
                return {
                    "success": False,
                    "message": analysis_result.get("message", "خطا در تحلیل شکل چهره"),
                    "request_id": request_id
                }
                
            face_shape = analysis_result.get("face_shape")
            confidence = analysis_result.get("confidence")
            shape_details = analysis_result.get("shape_metrics", {})
        
        # ذخیره نتیجه تحلیل در دیتابیس
        analysis_id = await save_analysis_result(
            user_id=user_id,
            request_id=request_id,
            face_shape=face_shape,
            confidence=confidence,
            client_info=client_info,
            task_id=str(analyze_face_shape_task.request.id)
        )
        
        # دریافت انواع فریم پیشنهادی
        recommended_frame_types = get_recommended_frame_types(face_shape)
        
        # نتیجه موفقیت‌آمیز
        result = {
            "success": True,
            "face_shape": face_shape,
            "confidence": confidence,
            "shape_details": shape_details,
            "recommended_frame_types": recommended_frame_types,
            "message": "شکل چهره با موفقیت تحلیل شد",
            "request_id": request_id,
            "user_id": user_id,
            "analysis_id": analysis_id
        }
        
        # ارسال وظیفه بعدی برای پیشنهاد فریم
        match_frames_task.delay(
            face_shape=face_shape,
            user_id=user_id,
            request_id=request_id,
            client_info=client_info,
            analysis_id=analysis_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"خطا در تحلیل شکل چهره: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تحلیل شکل چهره: {str(e)}",
            "request_id": request_id
        }


@shared_task(name="app.services.tasks.match_frames")
async def match_frames_task(face_shape: str, user_id: str, request_id: str, client_info: Dict[str, Any], analysis_id: Optional[str] = None) -> Dict[str, Any]:
    """
    وظیفه پیشنهاد فریم عینک مناسب.
    
    Args:
        face_shape: شکل چهره
        user_id: شناسه کاربر
        request_id: شناسه درخواست
        client_info: اطلاعات کاربر
        analysis_id: شناسه تحلیل (اختیاری)
        
    Returns:
        dict: نتیجه پیشنهاد فریم
    """
    try:
        logger.info(f"شروع پیشنهاد فریم برای درخواست {request_id} با شکل چهره {face_shape}")
        
        # واردسازی تأخیری برای جلوگیری از واردسازی دایره‌ای
        from app.core.face_analysis import get_recommended_frame_types
        from app.services.woocommerce import get_recommended_frames
        
        # دریافت انواع فریم پیشنهادی
        recommended_frame_types = get_recommended_frame_types(face_shape)
        
        # دریافت فریم‌های پیشنهادی از WooCommerce
        frames_result = await get_recommended_frames(face_shape)
        
        if not frames_result.get("success", False):
            logger.warning(f"دریافت فریم‌های پیشنهادی ناموفق بود برای درخواست {request_id}: {frames_result.get('message')}")
            return {
                "success": False,
                "message": frames_result.get("message", "خطا در دریافت فریم‌های پیشنهادی"),
                "request_id": request_id
            }
            
        recommended_frames = frames_result.get("recommended_frames", [])
        
        # ذخیره پیشنهادات در دیتابیس
        recommendation_id = await save_recommendation(
            user_id=user_id,
            face_shape=face_shape,
            recommended_frame_types=recommended_frame_types,
            recommended_frames=recommended_frames,
            client_info=client_info,
            analysis_id=analysis_id
        )
        
        # نتیجه موفقیت‌آمیز
        result = {
            "success": True,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommended_frames": recommended_frames,
            "message": "فریم‌های مناسب با موفقیت پیشنهاد شدند",
            "request_id": request_id,
            "user_id": user_id,
            "analysis_id": analysis_id,
            "recommendation_id": recommendation_id
        }
        
        return result
        
    except Exception as e:
        logger.error(f"خطا در پیشنهاد فریم: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در پیشنهاد فریم: {str(e)}",
            "request_id": request_id
        }