from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # هذا مهم للسماح لطلبات React بالوصول إلى الخادم

# تحميل النموذج والبيانات الأخرى اللازمة
model = pickle.load(open('fertilizer_model.pkl', 'rb'))
soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'success', 'message': 'Server is running'})

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'status': 'success',
        'soil_types': soil_types,
        'crop_types': crop_types
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # استخراج البيانات من الطلب
        temperature = float(data['temperature'])
        moisture = float(data['moisture'])
        rainfall = float(data['rainfall'])
        ph = float(data['ph'])
        nitrogen = float(data['nitrogen'])
        phosphorous = float(data['phosphorous'])
        potassium = float(data['potassium'])
        carbon = float(data['carbon'])
        soil_type = data['soil_type']
        crop_type = data['crop_type']

        # تحويل النوع النصي إلى أرقام
        soil_type_encoded = soil_types.index(soil_type)
        crop_type_encoded = crop_types.index(crop_type)

        # تجهيز البيانات للإدخال في النموذج
        input_data = np.array([[temperature, moisture, rainfall, ph, nitrogen, phosphorous, potassium, carbon, soil_type_encoded, crop_type_encoded]])

        # التنبؤ
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        # الحصول على جميع الاحتمالات لكل سماد
        all_probabilities = probabilities[0]
        fertilizer_classes = model.classes_

        # إنشاء قاموس بالاحتمالات لكل سماد
        prob_dict = {fertilizer: prob for fertilizer, prob in zip(fertilizer_classes, all_probabilities)}

        # العثور على السماد الموصى به (أعلى احتمال)
        recommended_fertilizer = prediction[0]
        confidence = np.max(probabilities)

        return jsonify({
            'status': 'success',
            'fertilizer': recommended_fertilizer,
            'confidence': confidence,
            'all_probabilities': prob_dict
        })

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
