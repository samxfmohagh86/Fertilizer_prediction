from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
from supabase import create_client, Client

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# إعداد Supabase
supabase_url = os.environ.get('SUPABASE_URL', 'YOUR_SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY', 'YOUR_SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# تحميل النماذج والمشفرات
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    le_soil = joblib.load('label_encoder_soil.joblib')
    le_crop = joblib.load('label_encoder_crop.joblib')
    le_fertilizer = joblib.load('label_encoder_fertilizer.joblib')
    
    logger.info("✅ تم تحميل جميع النماذج بنجاح")
    logger.info(f"فئات التربة: {list(le_soil.classes_)}")
    logger.info(f"فئات المحاصيل: {list(le_crop.classes_)}")
    logger.info(f"فئات الأسمدة: {list(le_fertilizer.classes_)}")
    
except Exception as e:
    logger.error(f"❌ خطأ في تحميل النماذج: {str(e)}")
    raise e

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'success',
        'message': 'الخادم يعمل بشكل طبيعي',
        'models_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/info', methods=['GET'])
def get_info():
    """إرجاع معلومات عن أنواع التربة والمحاصيل المتاحة"""
    return jsonify({
        'status': 'success',
        'soil_types': list(le_soil.classes_),
        'crop_types': list(le_crop.classes_),
        'fertilizer_types': list(le_fertilizer.classes_)
    })

@app.route('/history', methods=['GET'])
def get_history():
    """جلب آخر 10 سجلات من Supabase"""
    try:
        response = supabase.table('fertilizer_data')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(10)\
            .execute()
        
        return jsonify({
            'status': 'success',
            'data': response.data,
            'count': len(response.data)
        })
    except Exception as e:
        logger.error(f"خطأ في جلب السجلات: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'خطأ في جلب السجلات: {str(e)}'
        }), 500

def calculate_additional_features(temperature, moisture, rainfall, nitrogen, phosphorous, potassium):
    """حساب الميزات الإضافية بناءً على المعادلات"""
    try:
        # حساب نسبة NPK
        npk_ratio = nitrogen / (phosphorous + potassium + 1e-8)
        
        # حساب توازن المغذيات
        nutrient_balance = (nitrogen + phosphorous + potassium) / 3
        
        # حساب المؤشر البيئي
        environmental_index = (temperature * moisture * rainfall) / 1000
        
        return npk_ratio, nutrient_balance, environmental_index
    except Exception as e:
        logger.error(f"خطأ في حساب الميزات الإضافية: {e}")
        return 0.0, 0.0, 0.0

@app.route('/predict', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.get_json()
        logger.info(f"📨 received prediction request: {data}")
        
        # التحقق من وجود جميع الحقول المطلوبة
        required_fields = ['temperature', 'moisture', 'rainfall', 'ph', 'nitrogen', 
                          'phosphorous', 'potassium', 'carbon', 'soil_type', 'crop_type']
        
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({
                    'status': 'error',
                    'error': f'الحقل {field} مطلوب'
                }), 400
        
        # تحويل البيانات إلى تنسيق مناسب للنموذج
        try:
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
            logger.error(f"خطأ في تحويل البيانات: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': 'قيم غير صالحة في البيانات المدخلة'
            }), 400
        
        # حساب الميزات الإضافية
        npk_ratio, nutrient_balance, environmental_index = calculate_additional_features(
            temperature, moisture, rainfall, nitrogen, phosphorous, potassium
        )
        
        # إنشاء مصفوفة الميزات الكاملة
        input_features = np.array([[
            temperature, moisture, rainfall, ph, nitrogen, 
            phosphorous, potassium, carbon, soil_type_encoded, 
            crop_type_encoded, npk_ratio, nutrient_balance, environmental_index
        ]])
        
        logger.info(f"🔢 بيانات الإدخال المحولة: {input_features[0]}")
        
        # تطبيق المعايرة (Scaler)
        try:
            input_scaled = scaler.transform(input_features)
            logger.info("✅ تم تطبيق المعايرة بنجاح")
        except Exception as e:
            logger.error(f"❌ خطأ في المعايرة: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': f'خطأ في معالجة البيانات: {str(e)}'
            }), 500
        
        # الحصول على التنبؤات والاحتمالات
        try:
            prediction_encoded = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            predicted_fertilizer = le_fertilizer.inverse_transform([prediction_encoded])[0]
            
            # إنشاء قاموس بالاحتمالات لكل سماد
            all_probabilities = {
                le_fertilizer.inverse_transform([i])[0]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            confidence = max(all_probabilities.values())
            
            logger.info(f"🎯 السماد الموصى به: {predicted_fertilizer} (ثقة: {confidence:.2f})")
            
            response_data = {
                'status': 'success',
                'fertilizer': predicted_fertilizer,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'calculated_features': {
                    'npk_ratio': round(npk_ratio, 2),
                    'nutrient_balance': round(nutrient_balance, 2),
                    'environmental_index': round(environmental_index, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ خطأ في التنبؤ: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': f'خطأ في النموذج: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"❌ خطأ عام في التنبؤ: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'حدث خطأ أثناء المعالجة: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
