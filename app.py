from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# تحميل النماذج عند بدء التشغيل
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    le_soil = joblib.load('label_encoder_soil.joblib')
    le_crop = joblib.load('label_encoder_crop.joblib')
    le_fertilizer = joblib.load('label_encoder_fertilizer.joblib')
    models_loaded = True
except Exception as e:
    print(f"خطأ في تحميل النماذج: {e}")
    models_loaded = False

def predict_fertilizer(temperature, moisture, rainfall, ph, nitrogen, 
                      phosphorous, potassium, carbon, soil_type, crop_type):
    """
    تنبؤ السماد الموصى به بناءً على المعطيات المدخلة
    """
    try:
        if not models_loaded:
            return {'status': 'error', 'error': 'النماذج غير محملة'}
        
        # ترميز المدخلات الفئوية
        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded = le_crop.transform([crop_type])[0]

        # المعطيات الأساسية
        input_data = {
            'Temperature': float(temperature),
            'Moisture': float(moisture),
            'Rainfall': float(rainfall),
            'PH': float(ph),
            'Nitrogen': float(nitrogen),
            'Phosphorous': float(phosphorous),
            'Potassium': float(potassium),
            'Carbon': float(carbon),
            'Soil_encoded': soil_encoded,
            'Crop_encoded': crop_encoded
        }
        
        # إضافة الميزات المحسوبة إذا كان النموذج يستخدمها
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            
            if 'NPK_ratio' in features:
                input_data['NPK_ratio'] = nitrogen / (phosphorous + potassium + 1e-8)

            if 'Nutrient_balance' in features:
                input_data['Nutrient_balance'] = (nitrogen + phosphorous + potassium) / 3

            if 'Environmental_index' in features:
                input_data['Environmental_index'] = temperature * moisture * rainfall
        
        # تحويل المدخلات إلى DataFrame
        input_df = pd.DataFrame([input_data])
        
        # تطبيق التحجيم
        input_scaled = scaler.transform(input_df)

        # التنبؤ
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # فك ترميز اسم السماد
        fertilizer_name = le_fertilizer.inverse_transform([prediction_encoded])[0]
        confidence = float(np.max(prediction_proba))
        
        # قاموس الاحتمالات
        all_probabilities = {
            fert: float(prob) for fert, prob in zip(le_fertilizer.classes_, prediction_proba)
        }

        return {
            'status': 'success',
            'fertilizer': fertilizer_name,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
    
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استخراج البيانات من الطلب
        data = request.get_json()
        
        # التحقق من وجود جميع الحقول المطلوبة
        required_fields = ['temperature', 'moisture', 'rainfall', 'ph', 
                          'nitrogen', 'phosphorous', 'potassium', 'carbon', 
                          'soil_type', 'crop_type']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'error': f'الحقل {field} مفقود'})
        
        # استدعاء دالة التنبؤ
        result = predict_fertilizer(
            data['temperature'], data['moisture'], data['rainfall'], data['ph'],
            data['nitrogen'], data['phosphorous'], data['potassium'], data['carbon'],
            data['soil_type'], data['crop_type']
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)