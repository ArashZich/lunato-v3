print("شروع اسکریپت راه‌اندازی MongoDB");

// ایجاد کاربر مدیر
db.createUser({
  user: "eyeglass_admin",
  pwd: "secure_password123",
  roles: [
    { role: "readWrite", db: "eyeglass_recommendation" },
    { role: "dbAdmin", db: "eyeglass_recommendation" },
  ],
});

print("کاربر eyeglass_admin با موفقیت ایجاد شد");

// انتخاب دیتابیس
db = db.getSiblingDB("eyeglass_recommendation");

// ایجاد کالکشن‌ها
db.createCollection("requests");
db.createCollection("analysis_results");
db.createCollection("recommendations");
db.createCollection("woocommerce_cache"); // اضافه کردن کالکشن جدید

print("کالکشن‌های مورد نیاز ایجاد شدند");

// ایجاد ایندکس‌ها
// ایندکس برای کالکشن درخواست‌ها
db.requests.createIndex({ request_id: 1 }, { unique: true });
db.requests.createIndex({ created_at: 1 });
db.requests.createIndex({ "client_info.device_type": 1 });
db.requests.createIndex({ "client_info.browser_name": 1 });
db.requests.createIndex({ "client_info.os_name": 1 });
db.requests.createIndex({ status_code: 1 });

// ایندکس برای کالکشن نتایج تحلیل
db.analysis_results.createIndex({ user_id: 1 });
db.analysis_results.createIndex({ request_id: 1 });
db.analysis_results.createIndex({ face_shape: 1 });
db.analysis_results.createIndex({ created_at: 1 });
db.analysis_results.createIndex({ confidence: -1 });

// ایندکس برای کالکشن پیشنهادات
db.recommendations.createIndex({ user_id: 1 });
db.recommendations.createIndex({ face_shape: 1 });
db.recommendations.createIndex({ analysis_id: 1 });
db.recommendations.createIndex({ created_at: 1 });
db.recommendations.createIndex({ "recommended_frames.match_score": -1 });

// ایندکس برای کالکشن کش محصولات WooCommerce
db.woocommerce_cache.createIndex({ type: 1 }, { unique: true });
db.woocommerce_cache.createIndex({ last_update: 1 });

print("ایندکس‌های مورد نیاز ایجاد شدند");
print("راه‌اندازی MongoDB با موفقیت به پایان رسید");
