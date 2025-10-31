import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

# ------------------ CONFIG ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TEMPLATES_FOLDER = os.path.join(BASE_DIR, "templates")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ------------------ DATABASE MODELS ------------------
class Farmer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    contact = db.Column(db.String(50))

class SoilPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey('farmer.id'))
    soil_type = db.Column(db.String(50))
    crop_recommendation = db.Column(db.String(200))
    confidence = db.Column(db.Float)

class DiseasePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey('farmer.id'))
    disease_name = db.Column(db.String(100))
    treatment = db.Column(db.String(200))
    confidence = db.Column(db.Float)

# ------------------ HELPER FUNCTIONS ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ------------------ LOAD MODELS ------------------
SOIL_MODEL_PATH = os.path.join(MODELS_FOLDER, "soil_model.h5")
DISEASE_MODEL_PATH = os.path.join(MODELS_FOLDER, "disease_model.h5")

try:
    soil_model = tf.keras.models.load_model(SOIL_MODEL_PATH)
except Exception as e:
    soil_model = None
    print(f"Error loading soil model: {e}")

try:
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
except Exception as e:
    disease_model = None
    print(f"Error loading disease model: {e}")

# ------------------ PREDICTION FUNCTIONS ------------------
SOIL_TYPES = ["Sandy", "Loamy", "Clayey"]
CROP_RECOMMENDATIONS = {
    "Sandy": ["Wheat", "Barley", "Millets"],
    "Loamy": ["Rice", "Maize", "Sugarcane"],
    "Clayey": ["Rice", "Soybean", "Cotton"]
}

DISEASE_CLASSES = ["Healthy", "Powdery Mildew", "Leaf Spot", "Rust"]
TREATMENTS = {
    "Healthy": ["No treatment needed"],
    "Powdery Mildew": ["Use sulfur-based fungicide", "Improve air circulation"],
    "Leaf Spot": ["Remove infected leaves", "Apply copper fungicide"],
    "Rust": ["Use resistant varieties", "Apply fungicide"]
}

def predict_soil(img_path):
    if soil_model is None:
        return "Model not loaded", 0.0, []
    img = prepare_image(img_path)
    pred = soil_model.predict(img)
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    soil_type = SOIL_TYPES[class_idx]
    recommendations = CROP_RECOMMENDATIONS.get(soil_type, [])
    return soil_type, confidence, recommendations

def predict_disease(img_path):
    if disease_model is None:
        return "Model not loaded", 0.0, []
    img = prepare_image(img_path)
    pred = disease_model.predict(img)
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    disease_name = DISEASE_CLASSES[class_idx]
    treatment = TREATMENTS.get(disease_name, [])
    return disease_name, confidence, treatment

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    
    file = request.files['file']
    prediction_type = request.form.get("prediction_type")

    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if prediction_type == "soil":
            soil_type, confidence, recommendations = predict_soil(file_path)
            response = {
                "success": True,
                "prediction": soil_type,
                "confidence": confidence,
                "recommendations": recommendations
            }
        elif prediction_type == "disease":
            disease_name, confidence, treatment = predict_disease(file_path)
            response = {
                "success": True,
                "prediction": disease_name,
                "confidence": confidence,
                "recommendations": treatment
            }
        else:
            return jsonify({"success": False, "error": "Invalid prediction type"})

        return jsonify(response)
    else:
        return jsonify({"success": False, "error": "Invalid file type"})

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ------------------ RUN ------------------
if __name__ == "_main_":
    # Ensure app context when creating database
    with app.app_context():
        db.create_all()
    app.run(debug=True)