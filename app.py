from flask import Flask, request, jsonify
import joblib
import numpy as np

# تحميل النموذج والـ scaler
model = joblib.load("svm_fire_model.joblib")
scaler = joblib.load("fire_scaler.joblib")

# إنشاء التطبيق
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "data" not in data:
        return jsonify({"error": "Missing 'data' field"}), 400
    try:
        features = np.array(data["data"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# تشغيل التطبيق
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)