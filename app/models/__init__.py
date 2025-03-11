# Import model modules
from app.models.requests import FaceAnalysisRequest, FrameRecommendationRequest
from app.models.responses import BaseResponse, FaceAnalysisResponse, FaceCoordinates, RecommendedFrame, HealthResponse
from app.models.database import RequestRecord, AnalysisRecord, RecommendationRecord, AnalyticsSummary