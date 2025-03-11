# app/services/__init__.py
import os

# تعیین نوع سرویس WooCommerce بر اساس متغیر محیطی
use_mock_data = os.environ.get('USE_MOCK_DATA', 'false').lower() == 'true'

if use_mock_data:
    from app.services.woocommerce_mock import (
        get_all_products,
        get_eyeglass_frames,
        get_recommended_frames,
        get_product_by_id,
        get_products_by_category,
        get_cache_status,
        is_eyeglass_frame,
        get_frame_type,
        calculate_match_score,
        filter_products_by_price,
        sort_products_by_match_score
    )
else:
    from app.services.woocommerce import (
        get_all_products,
        get_eyeglass_frames,
        get_recommended_frames,
        get_product_by_id,
        get_products_by_category,
        get_cache_status,
        is_eyeglass_frame,
        get_frame_type,
        calculate_match_score,
        filter_products_by_price,
        sort_products_by_match_score
    )
