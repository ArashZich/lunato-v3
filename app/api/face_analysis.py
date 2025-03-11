from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, Query, Path
import logging
import numpy as np
import uuid
from typing import Optional
import base64
import json

from app.models.responses import FaceAnalysisResponse, ClientInfo
from app.models.enums import FaceShapeEnum 
from app.utils.client_info import extract_client_info
from app.utils.image_processing import validate_image_file, read_image_file
from app.core.face_detection import get_face_image
from app.core.face_analysis import generate_full_analysis
from app.core.frame_matching import get_combined_result
from app.services.tasks import detect_face_task, analyze_face_shape_task, match_frames_task
from app.db.repository import save_analysis_result, save_recommendation
from celery.result import AsyncResult

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# تعریف روتر
router = APIRouter()


@router.post("/analyze", response_model=FaceAnalysisResponse)
async def analyze_face(
    file: UploadFile = File(...),
    include_frames: bool = Form(True),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
    limit: int = Form(10),
    async_process: bool = Form(False),
    request: Request = None
):
    """
    آنالیز تصویر چهره و تشخیص شکل صورت.
    
    این API:
    1. تصویر را دریافت می‌کند
    2. چهره را در تصویر تشخیص می‌دهد
    3. شکل چهره را تحلیل می‌کند
    4. در صورت درخواست، فریم‌های عینک مناسب را پیشنهاد می‌دهد
    
    پارامتر async_process مشخص می‌کند که آیا پردازش به صورت غیرهمزمان انجام شود یا نه.
    در حالت غیرهمزمان، یک شناسه وظیفه برگردانده می‌شود که می‌توان با آن وضعیت پردازش را بررسی کرد.
    """
    try:
        # استخراج اطلاعات مرورگر و دستگاه کاربر
        client_info = extract_client_info(request) if request else None
        client_info_dict = client_info.dict() if client_info else {}
        
        # بررسی اعتبار تصویر
        is_valid = await validate_image_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail="فایل تصویر نامعتبر است")
        
        # ایجاد شناسه کاربر
        user_id = str(uuid.uuid4())
        
        # ایجاد شناسه درخواست
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # پردازش غیرهمزمان
        if async_process:
            # خواندن تصویر و تبدیل به base64
            image_content = await file.read()
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            await file.seek(0)  # بازگشت به ابتدای فایل
            
            # ارسال وظیفه به Celery
            task = detect_face_task.delay(
                image_data=image_base64,
                user_id=user_id,
                request_id=request_id,
                client_info=client_info_dict
            )
            
            # برگرداندن پاسخ با شناسه وظیفه
            return FaceAnalysisResponse(
                success=True,
                message="پردازش تصویر آغاز شد. برای بررسی وضعیت، از شناسه وظیفه استفاده کنید.",
                task_id=task.id,
                client_info=client_info
            )
        
        # پردازش همزمان
        else:
            # خواندن تصویر
            image = await read_image_file(file)
            if image is None:
                raise HTTPException(status_code=400, detail="خطا در خواندن فایل تصویر")
            
            # تشخیص چهره در تصویر
            success, detection_result, face_image = get_face_image(image)
            
            if not success:
                return FaceAnalysisResponse(
                    success=False,
                    message=detection_result.get("message", "خطا در تشخیص چهره"),
                    client_info=client_info
                )
            
            # آنالیز شکل چهره
            face_coordinates = detection_result.get("face")
            analysis_result = generate_full_analysis(image, face_coordinates)
            
            if not analysis_result.get("success", False):
                return FaceAnalysisResponse(
                    success=False,
                    message=analysis_result.get("message", "خطا در تحلیل شکل چهره"),
                    face_coordinates=face_coordinates,
                    client_info=client_info
                )
            
            # ذخیره نتیجه تحلیل در دیتابیس
            face_shape = analysis_result.get("face_shape")
            confidence = analysis_result.get("confidence")
            
            analysis_id = await save_analysis_result(
                user_id=user_id,
                request_id=request_id,
                face_shape=face_shape,
                confidence=confidence,
                client_info=client_info_dict
            )
            
            # ساخت پاسخ پایه
            response = FaceAnalysisResponse(
                success=True,
                message="تحلیل چهره با موفقیت انجام شد",
                face_coordinates=face_coordinates,
                face_shape=face_shape,
                confidence=confidence,
                description=analysis_result.get("description"),
                recommendation=analysis_result.get("recommendation"),
                client_info=client_info,
                recommended_frame_types=analysis_result.get("recommended_frame_types", [])
            )
            
            # اگر شامل پیشنهاد فریم باشد
            if include_frames:
                # دریافت فریم‌های پیشنهادی
                frames_result = await get_combined_result(
                    face_shape=face_shape,
                    min_price=min_price,
                    max_price=max_price,
                    limit=limit
                )
                
                # ذخیره پیشنهادات در دیتابیس
                if frames_result.get("success", False):
                    await save_recommendation(
                        user_id=user_id,
                        face_shape=face_shape,
                        recommended_frame_types=frames_result.get("recommended_frame_types", []),
                        recommended_frames=frames_result.get("recommended_frames", []),
                        client_info=client_info_dict,
                        analysis_id=analysis_id
                    )
                
                # افزودن فریم‌های پیشنهادی به پاسخ
                response.recommended_frames = frames_result.get("recommended_frames", [])
                
                if not frames_result.get("success", False):
                    logger.warning(f"خطا در دریافت فریم‌های پیشنهادی: {frames_result.get('message')}")
            
            return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطا در پردازش درخواست آنالیز چهره: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در پردازش درخواست: {str(e)}")


@router.get("/analyze/{task_id}", response_model=FaceAnalysisResponse)
async def get_analysis_status(task_id: str):
    """
    بررسی وضعیت پردازش غیرهمزمان.
    
    با استفاده از شناسه وظیفه، وضعیت پردازش تصویر را بررسی می‌کند.
    این API برای زمانی است که درخواست تحلیل به صورت غیرهمزمان ارسال شده باشد.
    """
    try:
        # بررسی وضعیت وظیفه
        task_result = AsyncResult(task_id)
        
        # اگر وظیفه هنوز کامل نشده است
        if not task_result.ready():
            return FaceAnalysisResponse(
                success=True,
                message=f"پردازش در حال انجام است. وضعیت: {task_result.state}",
                task_id=task_id
            )
        
        # اگر وظیفه با خطا مواجه شده است
        if task_result.failed():
            return FaceAnalysisResponse(
                success=False,
                message=f"پردازش با خطا مواجه شد: {str(task_result.result)}",
                task_id=task_id
            )
        
        # دریافت نتیجه وظیفه
        result = task_result.result
        
        # بررسی موفقیت‌آمیز بودن نتیجه
        if not result.get("success", False):
            return FaceAnalysisResponse(
                success=False,
                message=result.get("message", "خطا در پردازش تصویر"),
                task_id=task_id
            )
        
        # ساخت پاسخ
        return FaceAnalysisResponse(
            success=True,
            message="پردازش تصویر با موفقیت انجام شد",
            task_id=task_id,
            face_shape=result.get("face_shape"),
            confidence=result.get("confidence"),
            recommended_frame_types=result.get("recommended_frame_types", []),
            recommended_frames=result.get("recommended_frames", [])
        )
        
    except Exception as e:
        logger.error(f"خطا در بررسی وضعیت پردازش: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در بررسی وضعیت پردازش: {str(e)}")


@router.get("/face-shapes/{face_shape}/frames", response_model=FaceAnalysisResponse)
async def get_frames_for_face_shape(
    face_shape: FaceShapeEnum = Path(..., description="شکل چهره"),  # استفاده از Enum
    min_price: Optional[float] = Query(None, description="حداقل قیمت (تومان)"),
    max_price: Optional[float] = Query(None, description="حداکثر قیمت (تومان)"),
    limit: int = Query(10, description="حداکثر تعداد فریم‌های پیشنهادی"),
    request: Request = None
):
    """
    دریافت فریم‌های پیشنهادی براساس شکل چهره.
    
    این API برای زمانی است که کاربر از قبل شکل چهره خود را می‌داند و فقط نیاز به پیشنهاد فریم دارد.
    
    شکل‌های چهره موجود:
    - OVAL: بیضی
    - ROUND: گرد
    - SQUARE: مربعی
    - HEART: قلبی
    - OBLONG: کشیده
    - DIAMOND: لوزی
    - TRIANGLE: مثلثی
    """
    try:
        # استخراج اطلاعات مرورگر و دستگاه کاربر
        client_info = extract_client_info(request) if request else None
        client_info_dict = client_info.dict() if client_info else {}
        
        # تبدیل به حروف بزرگ برای استاندارد‌سازی (اختیاری - پیش‌فرض در Enum)
        # face_shape = face_shape.upper()
        
        # ایجاد شناسه کاربر
        user_id = str(uuid.uuid4())
        
        # دریافت فریم‌های پیشنهادی و اطلاعات شکل چهره
        result = await get_combined_result(
            face_shape=face_shape,
            min_price=min_price,
            max_price=max_price,
            limit=limit
        )
        
        # ذخیره پیشنهادات در دیتابیس
        if result.get("success", False):
            await save_recommendation(
                user_id=user_id,
                face_shape=face_shape,
                recommended_frame_types=result.get("recommended_frame_types", []),
                recommended_frames=result.get("recommended_frames", []),
                client_info=client_info_dict
            )
        
        if not result.get("success", False):
            return FaceAnalysisResponse(
                success=False,
                message=result.get("message", "خطا در دریافت فریم‌های پیشنهادی"),
                face_shape=face_shape,
                client_info=client_info
            )
        
        # ساخت پاسخ
        return FaceAnalysisResponse(
            success=True,
            message="فریم‌های پیشنهادی دریافت شد",
            face_shape=face_shape,
            description=result.get("description"),
            recommendation=result.get("recommendation"),
            client_info=client_info,
            recommended_frame_types=result.get("recommended_frame_types", []),
            recommended_frames=result.get("recommended_frames", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطا در دریافت فریم‌های پیشنهادی: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در پردازش درخواست: {str(e)}")