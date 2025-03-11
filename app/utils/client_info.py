import logging
from typing import Optional
import re
from fastapi import Request
from app.models.responses import ClientInfo

# تنظیمات لاگر
logger = logging.getLogger(__name__)


def extract_client_info(request: Optional[Request] = None) -> Optional[ClientInfo]:
    """
    استخراج اطلاعات کاربر از هدرهای درخواست.
    
    Args:
        request: درخواست FastAPI
        
    Returns:
        ClientInfo: اطلاعات کاربر یا None در صورت خطا
    """
    if request is None:
        return None
    
    try:
        headers = request.headers
        
        # استخراج آدرس IP
        ip_address = headers.get("X-Forwarded-For") or headers.get("X-Real-IP") or request.client.host if request.client else None
        
        # استخراج اطلاعات User-Agent
        user_agent = headers.get("User-Agent", "")
        
        # استخراج اطلاعات مرورگر و دستگاه
        device_type = _detect_device_type(user_agent)
        os_info = _detect_os(user_agent)
        browser_info = _detect_browser(user_agent)
        
        # استخراج زبان مرورگر
        language = headers.get("Accept-Language", "").split(',')[0] if headers.get("Accept-Language") else None
        
        return ClientInfo(
            device_type=device_type,
            os_name=os_info.get("name"),
            os_version=os_info.get("version"),
            browser_name=browser_info.get("name"),
            browser_version=browser_info.get("version"),
            ip_address=ip_address,
            language=language
        )
        
    except Exception as e:
        logger.error(f"خطا در استخراج اطلاعات کاربر: {str(e)}")
        
        # برگرداندن اطلاعات حداقلی در صورت خطا
        return ClientInfo(
            device_type="unknown",
            os_name=None,
            os_version=None,
            browser_name=None,
            browser_version=None,
            ip_address=None,
            language=None
        )


def _detect_device_type(user_agent: str) -> str:
    """
    تشخیص نوع دستگاه از User-Agent.
    
    Args:
        user_agent: رشته User-Agent
        
    Returns:
        str: نوع دستگاه (mobile, tablet, desktop)
    """
    user_agent = user_agent.lower()
    
    # تشخیص موبایل
    if any(device in user_agent for device in [
        "android", "iphone", "ipod", "windows phone", "blackberry", "mobile", "mobi"
    ]):
        # تشخیص تبلت
        if any(tablet in user_agent for tablet in ["ipad", "tablet"]):
            return "tablet"
        return "mobile"
    
    # تشخیص تبلت
    if any(tablet in user_agent for tablet in ["ipad", "tablet"]):
        return "tablet"
    
    # پیش‌فرض دسکتاپ
    return "desktop"


def _detect_os(user_agent: str) -> dict:
    """
    تشخیص سیستم عامل از User-Agent.
    
    Args:
        user_agent: رشته User-Agent
        
    Returns:
        dict: اطلاعات سیستم عامل
    """
    user_agent = user_agent.lower()
    
    # Windows
    windows_match = re.search(r"windows nt (\d+\.\d+)", user_agent)
    if windows_match:
        version = windows_match.group(1)
        windows_versions = {
            "10.0": "10/11",
            "6.3": "8.1",
            "6.2": "8",
            "6.1": "7",
            "6.0": "Vista",
            "5.2": "XP/Server 2003",
            "5.1": "XP",
            "5.0": "2000"
        }
        return {
            "name": "Windows",
            "version": windows_versions.get(version, version)
        }
    
    # MacOS
    mac_match = re.search(r"mac os x (\d+[._]\d+[._]?\d*)", user_agent)
    if mac_match:
        version = mac_match.group(1).replace("_", ".")
        return {
            "name": "macOS",
            "version": version
        }
    
    # iOS
    ios_match = re.search(r"(?:iphone|ipad|ipod).+?os (\d+[._]\d+[._]?\d*)", user_agent)
    if ios_match:
        version = ios_match.group(1).replace("_", ".")
        return {
            "name": "iOS",
            "version": version
        }
    
    # Android
    android_match = re.search(r"android (\d+(?:\.\d+)+)", user_agent)
    if android_match:
        version = android_match.group(1)
        return {
            "name": "Android",
            "version": version
        }
    
    # Linux
    if "linux" in user_agent:
        return {
            "name": "Linux",
            "version": None
        }
    
    return {
        "name": None,
        "version": None
    }


def _detect_browser(user_agent: str) -> dict:
    """
    تشخیص مرورگر از User-Agent.
    
    Args:
        user_agent: رشته User-Agent
        
    Returns:
        dict: اطلاعات مرورگر
    """
    user_agent = user_agent.lower()
    
    # Chrome
    chrome_match = re.search(r"(?:chrome|crios)/(\d+(?:\.\d+)+)", user_agent)
    if chrome_match and "edge" not in user_agent and "opr" not in user_agent and "samsung" not in user_agent:
        return {
            "name": "Chrome",
            "version": chrome_match.group(1)
        }
    
    # Firefox
    firefox_match = re.search(r"(?:firefox|fxios)/(\d+(?:\.\d+)+)", user_agent)
    if firefox_match:
        return {
            "name": "Firefox",
            "version": firefox_match.group(1)
        }
    
    # Safari
    safari_match = re.search(r"safari/(\d+(?:\.\d+)+)", user_agent)
    if safari_match and "chrome" not in user_agent and "android" not in user_agent:
        webkit_match = re.search(r"version/(\d+(?:\.\d+)+)", user_agent)
        if webkit_match:
            return {
                "name": "Safari",
                "version": webkit_match.group(1)
            }
    
    # Edge
    edge_match = re.search(r"(?:edge|edg|edgios)/(\d+(?:\.\d+)+)", user_agent)
    if edge_match:
        return {
            "name": "Edge",
            "version": edge_match.group(1)
        }
    
    # Opera
    opera_match = re.search(r"(?:opr|opera)/(\d+(?:\.\d+)+)", user_agent)
    if opera_match:
        return {
            "name": "Opera",
            "version": opera_match.group(1)
        }
    
    # Samsung Internet
    samsung_match = re.search(r"samsungbrowser/(\d+(?:\.\d+)+)", user_agent)
    if samsung_match:
        return {
            "name": "Samsung Internet",
            "version": samsung_match.group(1)
        }
    
    # IE
    ie_match = re.search(r"msie (\d+(?:\.\d+)+)|trident.+?rv:(\d+(?:\.\d+)+)", user_agent)
    if ie_match:
        version = ie_match.group(1) or ie_match.group(2)
        return {
            "name": "Internet Explorer",
            "version": version
        }
    
    return {
        "name": None,
        "version": None
    }