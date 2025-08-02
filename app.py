from flask import Flask, request, jsonify
import tempfile
import os
import numpy as np
from collections import defaultdict
from PIL import Image
import base64
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None

def load_model():
    """Load YOLO model with error handling"""
    global model
    try:
        from ultralytics import YOLO
        
        # Try different possible paths for the model
        possible_paths = [
            'best.pt',
            './best.pt',
            os.path.join(os.path.dirname(__file__), 'best.pt'),
            os.path.join(os.getcwd(), 'best.pt')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"Found model at: {model_path}")
                break
        
        if model_path is None:
            logger.error("Model file 'best.pt' not found in any expected location")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Files in current directory: {os.listdir('.')}")
            raise FileNotFoundError("Model file 'best.pt' not found")
        
        # Load model with explicit device setting
        logger.info("Loading YOLO model...")
        model = YOLO(model_path)
        
        # Force CPU usage to avoid CUDA issues on Render
        model.to('cpu')
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# -------------------- Class Mapping & Weights --------------------
class_mapping = {
    0: 'fine dust',
    1: 'garbagebag',
    2: 'liquid',
    3: 'paper_waste',
    4: 'plastic_bottles',
    5: 'plasticbags',
    6: 'stains'
}

original_weights = {
    0: 1,
    1: 5,
    2: 4,
    3: 2,
    4: 3,
    5: 4,
    6: 3
}

total_weight = sum(original_weights.values())
normalized_weights = {cls: (wt / total_weight * 10) for cls, wt in original_weights.items()}

# -------------------- Health Check Endpoint --------------------
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Render"""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 503
    return jsonify({"status": "healthy"}), 200

# -------------------- Predict Endpoint --------------------
@app.route("/predict", methods=["POST"])
def predict():
    global model
    
    if model is None:
        return jsonify({"error": "Model not loaded. Please try again later."}), 503
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            file.save(tmp_file.name)
            image_path = tmp_file.name

        # Run inference
        logger.info("Running YOLO inference...")
        results = model(image_path, device='cpu')  # Explicitly use CPU
        result = results[0]

        # Check if any detections were made
        if result.boxes is None or len(result.boxes) == 0:
            # Clean up temp file
            os.unlink(image_path)
            return jsonify({
                "cleanliness_score": 10.0,
                "raw_score": 0.0,
                "breakdown": [],
                "message": "No objects detected - area appears clean!"
            })

        # Process detections
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        raw_score = 0
        class_confidence_dict = defaultdict(list)

        for cls_id, conf in zip(class_ids, confidences):
            weight = normalized_weights.get(cls_id, 1)
            raw_score += weight * conf
            class_confidence_dict[cls_id].append(conf)

        cleanliness_score = max(0, round(10 - raw_score, 2))

        breakdown = []
        for cls_id, conf_list in class_confidence_dict.items():
            count = len(conf_list)
            avg_conf = float(np.mean(conf_list))
            breakdown.append({
                "class": class_mapping.get(cls_id, str(cls_id)),
                "count": count,
                "avg_conf": round(avg_conf, 2),
                "weight": round(normalized_weights[cls_id], 2)
            })

        # Save annotated image
        try:
            annotated_frame = result.plot()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as output_img:
                import cv2
                cv2.imwrite(output_img.name, annotated_frame)

                with open(output_img.name, "rb") as img_file:
                    encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Clean up temp files
                os.unlink(output_img.name)
        except Exception as e:
            logger.warning(f"Could not create annotated image: {str(e)}")
            encoded_img = None

        # Clean up temp file
        os.unlink(image_path)

        response = {
            "cleanliness_score": cleanliness_score,
            "raw_score": round(raw_score, 2),
            "breakdown": breakdown
        }
        
        if encoded_img:
            response["image_base64"] = encoded_img

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        # Clean up temp file if it exists
        if 'image_path' in locals() and os.path.exists(image_path):
            os.unlink(image_path)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# -------------------- Root Endpoint --------------------
@app.route("/")
def home():
    return jsonify({
        "message": "🧼 Cleanliness Score YOLOv8 API is up and running!",
        "endpoints": {
            "/": "Home page",
            "/health": "Health check",
            "/predict": "POST - Upload image for cleanliness scoring"
        }
    })

# -------------------- Initialize Model on Startup --------------------
@app.before_first_request
def initialize():
    """Initialize model when first request is made"""
    logger.info("Initializing model...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

# -------------------- Main Entry --------------------
if __name__ == "__main__":
    # Load model on startup
    logger.info("Starting application...")
    success = load_model()
    if not success:
        logger.error("Failed to load model. Exiting...")
        sys.exit(1)
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)