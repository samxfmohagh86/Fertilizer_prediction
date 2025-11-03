import numpy as np
import pandas as pd
import joblib

def predict_fertilizer(temperature, moisture, rainfall, ph, nitrogen, 
                      phosphorous, potassium, carbon, soil_type, crop_type):
    """
    Predict fertilizer recommendation based on input parameters
    """
    try:
        # Load saved artifacts
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        le_soil = joblib.load('label_encoder_soil.joblib')
        le_crop = joblib.load('label_encoder_crop.joblib')
        le_fertilizer = joblib.load('label_encoder_fertilizer.joblib')
        
        # Encode categorical inputs
        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded = le_crop.transform([crop_type])[0]

        # Base input features
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
        
        # Add engineered features if model uses them
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            
            if 'NPK_ratio' in features:
                input_data['NPK_ratio'] = nitrogen / (phosphorous + potassium + 1e-8)

            if 'Nutrient_balance' in features:
                input_data['Nutrient_balance'] = (nitrogen + phosphorous + potassium) / 3

            if 'Environmental_index' in features:
                input_data['Environmental_index'] = temperature * moisture * rainfall
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale values
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Decode fertilizer label
        fertilizer_name = le_fertilizer.inverse_transform([prediction_encoded])[0]
        confidence = float(np.max(prediction_proba))
        
        # Probabilities dictionary
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


# ✅ التصحيح النهائي - استخدام المفاتيح الصحيحة
def display_result(result):
    if result['status'] == 'success':
        print("🔮 نتائج التنبؤ:")
        print(f"السماد الموصى به: {result['fertilizer']}")
        print(f"درجة الثقة: {result['confidence']:.2%}")
        print("\nاحتمالات جميع الأسمدة:")
        for fert, prob in result['all_probabilities'].items():
            print(f"  - {fert}: {prob:.2%}")
    else:
        print(f"❌ حدث خطأ: {result['error']}")


# 🔍 تحقق من الميزات التي يتوقعها النموذج
def check_model_features():
    try:
        model = joblib.load('model.joblib')
        if hasattr(model, 'feature_names_in_'):
            print("📋 الميزات التي يتوقعها النموذج:")
            for i, feature in enumerate(model.feature_names_in_):
                print(f"  {i+1}. {feature}")
        else:
            print("ℹ️ النموذج لا يحتوي على معلومات الميزات")
    except Exception as e:
        print(f"❌ خطأ في التحقق من الميزات: {e}")


# ✅ مثال تشغيل
result = predict_fertilizer(25, 60, 200, 6.5, 50, 20, 30, 1.2, "Loamy Soil", "wheat")
display_result(result)

# تشغيل التحقق
check_model_features()
