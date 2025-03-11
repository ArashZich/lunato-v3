# Import database modules
from app.db.connection import connect_to_mongo, close_mongo_connection, get_database
from app.db.repository import save_request_info, save_analysis_result, save_recommendation, get_analytics_summary