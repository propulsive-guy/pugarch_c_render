from flask import Flask, request, jsonify
import tempfile
import os
import numpy as np
from collections import defaultdict
from PIL import Image
import base64
import logging
import sys
import gc
import psutil

# Minimal logging to reduce memory usage
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def load_model():
    """Load YOLO model with memory optimization"""
    global model
    try:
        logger.warning("Loading model...")
        
        # Check available memory
        try:
            memory_mb = get_memory_usage()
            logger.warning(f"Memory before loading: {memory_mb:.1f} MB")
        except:
            pass
        
        from ultralytics import YOLO
        
        # Try to find model file
        possible_paths = ['best.pt', './best.pt', os.path.join(os.path.dirname(__file__), 'best.pt')]
        model_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            logger.error("Model file not found")
            return False
        
        # Load model with memory optimization
        model = YOLO(model_path)
        
        # Force CPU and optimize for inference
        model.to('cpu')
        model.model.eval()  # Set to evaluation mode
        
        # Disable unnecessary features to save memory
        if hasattr(model, 'predictor'):
            model.predictor.args.save = False
            model.predictor.args.save_txt = False
            model.predictor.args.save_conf = False
            model.predictor.args.save_crop = False
            model.predictor.args.show = False
            model.predictor.args.verbose = False
        
        # Force garbage collection
        gc.collect()
        
        try:
            memory_mb = get_memory_usage()
            logger.warning(f"Memory after loading: {memory_mb:.1f} MB")
        except:
            pass
        
        logger.warning("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

# Lightweight class mapping
class_mapping = {0: 'fine dust', 1: 'garbagebag', 2: 'liquid', 3: 'paper_waste', 4: 'plastic_bottles', 5: 'plasticbags', 6: 'stains'}
normalized_weights = {0: 0.45, 1: 2.27, 2: 1.82, 3: 0.91, 4: 1.36, 5: 1.82, 6: 1.36}

@app.route("/health", methods=["GET"])
def health_check():
    """Minimal health check"""
    try:
        memory_mb = get_memory_usage()
        return jsonify({
            "status": "healthy" if model is not None else "unhealthy",
            "model_loaded": model is not None,
            "memory_mb": round(memory_mb, 1)
        }), 200 if model is not None else 503
    except:
        return jsonify({"status": "healthy" if model is not None else "unhealthy"}), 200 if model is not None else 503

@app.route("/predict", methods=["POST"])
def predict():
    """Memory-optimized prediction endpoint"""
    global model
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    image_path = None
    output_path = None
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            file.save(tmp_file.name)
            image_path = tmp_file.name

        # Verify image
        try:
            test_img = Image.open(image_path)
            # Resize large images to save memory
            if test_img.size[0] > 1280 or test_img.size[1] > 1280:
                test_img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                test_img.save(image_path, "JPEG", quality=85)
            test_img.close()
        except Exception as e:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
            return jsonify({"error": f"Invalid image: {str(e)}"}), 400

        # Force garbage collection before inference
        gc.collect()

        # Run inference with minimal settings
        logger.warning("Starting inference...")
        results = model.predict(
            source=image_path,
            device='cpu',
            verbose=False,
            save=False,
            show=False,
            conf=0.25,  # Lower confidence threshold
            iou=0.45,   # IoU threshold
            max_det=100,  # Limit detections
            agnostic_nms=True,
            augment=False,  # Disable augmentation to save memory
            half=False,     # Disable half precision
            imgsz=640       # Standard size
        )
        
        result = results[0]
        logger.warning("Inference completed")

        # Check detections
        if result.boxes is None or len(result.boxes) == 0:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
            return jsonify({
                "cleanliness_score": 10.0,
                "raw_score": 0.0,
                "breakdown": [],
                "message": "No objects detected - area appears clean!"
            })

        # Process detections efficiently
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
                "weight": round(normalized_weights.get(cls_id, 1), 2)
            })

        response = {
            "cleanliness_score": cleanliness_score,
            "raw_score": round(raw_score, 2),
            "breakdown": breakdown
        }

        # Only create annotated image for smaller responses
        try:
            if len(breakdown) <= 10:  # Only if not too many objects
                annotated_frame = result.plot(
                    line_width=2,
                    font_size=12,
                    pil=False,
                    img=None,
                    labels=True,
                    boxes=True,
                    conf=True
                )
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as output_img:
                    import cv2
                    # Compress heavily to save memory
                    cv2.imwrite(output_img.name, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    output_path = output_img.name

                    with open(output_path, "rb") as img_file:
                        encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    response["image_base64"] = encoded_img
        except Exception as e:
            logger.warning(f"Could not create annotated image: {e}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    finally:
        # Cleanup
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        
        # Force garbage collection
        gc.collect()

@app.route("/")
def home():
    """Minimal home endpoint"""
    return jsonify({
        "message": "🧼 Cleanliness API Running",
        "model_loaded": model is not None,
        "endpoints": {"/": "Home", "/health": "Health", "/predict": "POST - Analyze image"}
    })

# Initialize model
def initialize_model():
    """Initialize model on startup"""
    logger.warning("Initializing...")
    success = load_model()
    if not success:
        logger.error("Model loading failed")

# Initialize when imported
try:
    initialize_model()
except Exception as e:
    logger.error(f"Initialization failed: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)