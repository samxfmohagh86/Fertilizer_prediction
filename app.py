from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# تحميل النماذج والمشفرات
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    le_soil = joblib.load('label_encoder_soil.joblib')
    le_crop = joblib.load('label_encoder_crop.joblib')
    le_fertilizer = joblib.load('label_encoder_fertilizer.joblib')
    
    print("✅ تم تحميل جميع النماذج بنجاح")
    print(f"فئات التربة: {list(le_soil.classes_)}")
    print(f"فئات المحاصيل: {list(le_crop.classes_)}")
    print(f"فئات الأسمدة: {list(le_fertilizer.classes_)}")
    
except Exception as e:
    print(f"❌ خطأ في تحميل النماذج: {str(e)}")
    raise e

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'success',
        'message': 'الخادم يعمل بشكل طبيعي',
        'models_loaded': True
    })

@app.route('/info', methods=['GET'])
def get_info():
    return jsonify({
        'status': 'success',
        'soil_types': list(le_soil.classes_),
        'crop_types': list(le_crop.classes_),
        'fertilizer_types': list(le_fertilizer.classes_)
    })

@app.route('/predict', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.get_json()
        
        # التحقق من وجود جميع الحقول المطلوبة
        required_fields = ['temperature', 'moisture', 'rainfall', 'ph', 'nitrogen', 
                          'phosphorous', 'potassium', 'carbon', 'soil_type', 'crop_type']
        
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({
                    'status': 'error',
                    'error': f'الحقل {field} مطلوب'
                }), 400
        
        # تحويل البيانات إلى تنسيق مناسب للنموذج
        try:
            # تحويل البيانات الرقمية
            temperature = float(data['temperature'])
            moisture = float(data['moisture'])
            rainfall = float(data['rainfall'])
            ph = float(data['ph'])
            nitrogen = float(data['nitrogen'])
            phosphorous = float(data['phosphorous'])
            potassium = float(data['potassium'])
            carbon = float(data['carbon'])
            
            # تحويل النصوص باستخدام LabelEncoders
            soil_type_encoded = le_soil.transform([data['soil_type']])[0]
            crop_type_encoded = le_crop.transform([data['crop_type']])[0]
            
        except (ValueError, KeyError) as e:
            return jsonify({
                'status': 'error',
                'error': 'قيم غير صالحة في البيانات المدخلة'
            }), 400
        
        # تجهيز بيانات الإدخال للنموذج
        input_features = np.array([[
            temperature, moisture, rainfall, ph, nitrogen, 
            phosphorous, potassium, carbon, soil_type_encoded, crop_type_encoded
        ]])
        
        # تطبيق المعايرة (Scaler)
        input_scaled = scaler.transform(input_features)
        
        # الحصول على التنبؤات والاحتمالات
        prediction_encoded = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # فك التشفير للحصول على اسم السماد
        predicted_fertilizer = le_fertilizer.inverse_transform([prediction_encoded])[0]
        
        # إنشاء قاموس بالاحتمالات لكل سماد
        all_probabilities = {
            le_fertilizer.inverse_transform([i])[0]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # العثور على أعلى احتمال
        confidence = max(all_probabilities.values())
        
        return jsonify({
            'status': 'success',
            'fertilizer': predicted_fertilizer,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        })
        
    except Exception as e:
        print(f"❌ خطأ في التنبؤ: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'حدث خطأ أثناء المعالجة: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
