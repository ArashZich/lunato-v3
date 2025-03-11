# app/models/woocommerce.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class WooCommerceProductImage(BaseModel):
    """مدل تصویر محصول WooCommerce"""
    id: int
    src: str
    name: Optional[str] = None
    alt: Optional[str] = None


class WooCommerceProductAttribute(BaseModel):
    """مدل ویژگی محصول WooCommerce"""
    id: int
    name: str
    options: List[str]


class WooCommerceProductCategory(BaseModel):
    """مدل دسته‌بندی محصول WooCommerce"""
    id: int
    name: str
    slug: str


class WooCommerceProduct(BaseModel):
    """مدل محصول WooCommerce"""
    id: int
    name: str
    slug: str
    permalink: str
    price: str
    regular_price: Optional[str] = None
    sale_price: Optional[str] = None
    description: Optional[str] = None
    short_description: Optional[str] = None
    categories: List[WooCommerceProductCategory] = []
    attributes: List[WooCommerceProductAttribute] = []
    images: List[WooCommerceProductImage] = []
    frame_type: Optional[str] = None
    is_eyeglass_frame: Optional[bool] = None
    match_score: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True


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


class RecommendedFrame(BaseModel):
    """مدل فریم پیشنهادی برای نمایش به کاربر"""
    id: int
    name: str
    permalink: str
    price: str
    regular_price: Optional[str] = None
    frame_type: str
    images: List[str]
    match_score: float