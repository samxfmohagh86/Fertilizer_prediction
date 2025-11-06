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
    
    # الحصول على أسماء الميزات التي تم تدريب النموذج عليها
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_.tolist()
        print(f"الميزات المتوقعة: {feature_names}")
    else:
        # إذا لم تكن أسماء الميزات متاحة، نفترض الميزات القياسية
        feature_names = ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 
                        'Phosphorous', 'Potassium', 'Carbon', 'Soil_encoded', 'Crop_encoded',
                        'NPK_ratio', 'Nutrient_balance', 'Environmental_index']
        print("⚠️ استخدام الميزات الافتراضية")
    
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

def calculate_additional_features(temperature, moisture, rainfall, nitrogen, phosphorous, potassium):
    """حساب الميزات الإضافية بناءً على المعادلات"""
    try:
        # حساب نسبة NPK
        npk_ratio = nitrogen / (phosphorous + potassium + 1e-8)  # تجنب القسمة على الصفر
        
        # حساب توازن المغذيات
        nutrient_balance = (nitrogen + phosphorous + potassium) / 3
        
        # حساب المؤشر البيئي
        environmental_index = (temperature * moisture * rainfall) / 1000
        
        return npk_ratio, nutrient_balance, environmental_index
    except Exception as e:
        print(f"خطأ في حساب الميزات الإضافية: {e}")
        return 0.0, 0.0, 0.0

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
        
        # حساب الميزات الإضافية
        npk_ratio, nutrient_balance, environmental_index = calculate_additional_features(
            temperature, moisture, rainfall, nitrogen, phosphorous, potassium
        )
        
        # إنشاء مصفوفة الميزات الكاملة (13 ميزة)
        input_features = np.array([[
            temperature,      # Temperature
            moisture,         # Moisture
            rainfall,         # Rainfall
            ph,               # PH
            nitrogen,         # Nitrogen
            phosphorous,      # Phosphorous
            potassium,        # Potassium
            carbon,           # Carbon
            soil_type_encoded, # Soil_encoded
            crop_type_encoded, # Crop_encoded
            npk_ratio,        # NPK_ratio
            nutrient_balance, # Nutrient_balance
            environmental_index # Environmental_index
        ]])
        
        print(f"🔢 شكل بيانات الإدخال: {input_features.shape}")
        print(f"📊 بيانات الإدخال: {input_features[0]}")
        
        # تطبيق المعايرة (Scaler)
        try:
            input_scaled = scaler.transform(input_features)
            print(f"✅ تم تطبيق المعايرة بنجاح")
        except Exception as e:
            print(f"❌ خطأ في المعايرة: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': f'خطأ في معالجة البيانات: {str(e)}'
            }), 500
        
        # الحصول على التنبؤات والاحتمالات
        try:
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
            
            print(f"🎯 السماد الموصى به: {predicted_fertilizer}")
            print(f"📈 مستوى الثقة: {confidence:.2f}")
            
            return jsonify({
                'status': 'success',
                'fertilizer': predicted_fertilizer,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'calculated_features': {
                    'npk_ratio': round(npk_ratio, 2),
                    'nutrient_balance': round(nutrient_balance, 2),
                    'environmental_index': round(environmental_index, 2)
                }
            })
            
        except Exception as e:
            print(f"❌ خطأ في التنبؤ: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': f'خطأ في النموذج: {str(e)}'
            }), 500
        
    except Exception as e:
        print(f"❌ خطأ عام في التنبؤ: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'حدث خطأ أثناء المعالجة: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
