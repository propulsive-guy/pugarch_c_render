from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile
import io
import os

app = Flask(__name__)

# Load lightweight YOLOv8 model
model = YOLO("best.pt")

@app.route("/")
def home():
    return "🧼 Cleanliness Score API is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read and convert image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)

        # Run inference without warmup to reduce memory load
        results = model.predict(image_array, conf=0.25, verbose=False)[0]

        # Count detections
        counts = {}
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            counts[cls_name] = counts.get(cls_name, 0) + 1

        dirty = counts.get("dirty", 0)
        clean = counts.get("clean", 0)
        total = dirty + clean
        score = (clean / total) * 100 if total > 0 else 0

        # Annotate image
        annotated = results.plot()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, annotated)

        # Send file and metadata
        return send_file(
            temp_file.name,
            mimetype="image/jpeg",
            download_name="result.jpg"
        ), 200, {
            "X-Cleanliness-Score": f"{score:.2f}",
            "X-Detection-Counts": str(counts)
        }

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
