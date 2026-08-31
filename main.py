# =========================================================
# 🌿 مشروع: تصنيف الصور والتعرف على الأوراق والأنواع
# دورة تطبيقات الذكاء الاصطناعي - Computer Vision Project
# =========================================================

# 1️⃣ استدعاء المكتبات البرمجية المطلوبة
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

print("✅ تم تحميل المكتبات بنجاح!")
print(f"إصدار TensorFlow: {tf.__version__}")

# 2️⃣ إنشاء صورة اختبارية وتحديد المسار
sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 245
sample_image[40:180, 80:144] = [34, 139, 34]  # رسم شكل ورقة باللون الأخضر

image_filename = 'sample_leaf.jpg'
plt.imsave(image_filename, sample_image)
print(f"✅ تم تجهيز الصورة الاختباريّة باسم '{image_filename}'")

# 3️⃣ تحميل نموذج الذكاء الاصطناعي المدرب جاهزاً (MobileNetV2)
model = MobileNetV2(weights='imagenet')
print("✅ تم تحميل نموذج الذكاء الاصطناعي!")

# 4️⃣ دالة تحليل الصورة والتنبؤ بالنتيجة
def classify_image(img_path):
    # تحميل وتعديل قياسات الصورة
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)
    
    # عرض الصورة المدخلة
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title("الصورة المدخلة")
    plt.show()
    
    # التنبؤ بواسطة الذكاء الاصطناعي
    predictions = model.predict(img_preprocessed)
    results = decode_predictions(predictions, top=3)[0]
    
    print("\n--- 📊 نتائج تحليل الذكاء الاصطناعي ---")
    for i, (imagenet_id, label, score) in enumerate(results):
        print(f"{i+1}. التصنيف: {label:<20} | نسبة الثقة: {score*100:.2f}%")

# تشغيل الفحص
classify_image(image_filename)
