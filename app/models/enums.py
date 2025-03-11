# app/models/enums.py
from enum import Enum, auto


class FaceShapeEnum(str, Enum):
    """انواع شکل‌های چهره"""
    OVAL = "OVAL"  # بیضی
    ROUND = "ROUND"  # گرد
    SQUARE = "SQUARE"  # مربعی
    HEART = "HEART"  # قلبی
    OBLONG = "OBLONG"  # کشیده

    @classmethod
    def get_description(cls, shape_name):
        """دریافت توضیحات فارسی شکل چهره"""
        descriptions = {
            cls.OVAL: "بیضی",
            cls.ROUND: "گرد",
            cls.SQUARE: "مربعی",
            cls.HEART: "قلبی",
            cls.OBLONG: "کشیده"
        }
        return descriptions.get(shape_name, "نامشخص")
