from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class RequestRecord(BaseModel):
    """مدل رکورد درخواست‌ها"""
    request_id: str
    path: str
    method: str
    client_info: Dict[str, Any]
    status_code: int
    process_time: float
    created_at: datetime


class AnalysisRecord(BaseModel):
    """مدل رکورد نتایج تحلیل چهره"""
    user_id: str
    request_id: str
    face_shape: str
    confidence: float
    client_info: Dict[str, Any]
    task_id: Optional[str] = None
    created_at: datetime


class RecommendationRecord(BaseModel):
    """مدل رکورد پیشنهادات فریم"""
    user_id: str
    face_shape: str
    recommended_frame_types: List[str]
    recommended_frames: List[Dict[str, Any]]
    client_info: Dict[str, Any]
    analysis_id: Optional[str] = None
    created_at: datetime


class AnalyticsSummary(BaseModel):
    """مدل خلاصه اطلاعات تحلیلی"""
    total_requests: int = 0
    total_unique_users: int = 0
    face_shapes: Dict[str, int] = {}
    devices: Dict[str, int] = {}
    browsers: Dict[str, int] = {}
    frame_types: Dict[str, int] = {}
    operating_systems: Dict[str, int] = {}
    average_process_time: float = 0
    last_update_time: Optional[datetime] = None
    period: Optional[str] = "all"


class AnalyticsDetailItem(BaseModel):
    """مدل یک آیتم تحلیلی جزئی"""
    request_id: str
    user_id: str
    face_shape: str
    confidence: float
    device_type: str
    browser_name: Optional[str] = None
    recommended_frame_types: List[str] = []
    created_at: datetime


class DetailedAnalytics(BaseModel):
    """مدل اطلاعات تحلیلی تفصیلی"""
    total: int = 0
    period: str = "all"
    skip: int = 0
    limit: int = 100
    items: List[AnalyticsDetailItem] = []


class TimeDataPoint(BaseModel):
    """مدل نقطه داده زمانی"""
    time_period: str
    count: int
    face_shape_distribution: Optional[Dict[str, int]] = None
    avg_confidence: Optional[float] = None


class TimeBasedAnalytics(BaseModel):
    """مدل اطلاعات تحلیلی بر اساس زمان"""
    group_by: str
    period: str
    face_shape_filter: Optional[str] = None
    data_points: List[TimeDataPoint] = []


class PopularFrame(BaseModel):
    """مدل فریم محبوب"""
    id: int
    name: str
    frame_type: str
    avg_match_score: float
    recommendation_count: int


class FramePopularity(BaseModel):
    """مدل محبوبیت فریم‌ها"""
    period: str
    popular_frames: List[PopularFrame] = []


class ConversionStats(BaseModel):
    """مدل آمار تبدیل"""
    period: str
    total_requests: int
    successful_requests: int
    success_rate: float
    successful_analyses: int
    analysis_to_request_ratio: float
    total_recommendations: int
    recommendation_to_analysis_ratio: float


class WooCommerceCache(BaseModel):
    """مدل کش محصولات WooCommerce"""
    type: str = "products_cache"
    last_update: datetime
    data: List[Dict[str, Any]]


class WooCommerceCacheStatus(BaseModel):
    """مدل وضعیت کش محصولات WooCommerce"""
    cache_initialized: bool
    total_products: int
    last_update: Optional[datetime] = None
    update_in_progress: bool = False
    last_error: Optional[str] = None
    eyeglass_frames_count: int = 0
    frame_types_count: Dict[str, int] = {}