# app.py
from flask import Flask, request, send_file, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import os
import io

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["X-Detections"]  # Important for accessing custom headers
    }
})


class ConstellationDetector:
    def __init__(self, model_path):
        """
        Initialize the constellation detector with the trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        # Load the model and labels
        self.model = tf.keras.models.load_model(model_path)
        
        # Load labels and thresholds
        model_dir = Path(model_path).parent
        with open(model_dir / 'constellation_labels.txt', 'r') as f:
            self.labels = f.read().splitlines()
        
        # Load thresholds if available, otherwise use default
        metrics_path = model_dir / 'validation_metrics' / 'validation_metrics.csv'
        if metrics_path.exists():
            import pandas as pd
            metrics_df = pd.read_csv(metrics_path)
            self.thresholds = dict(zip(metrics_df['Label'], metrics_df['Threshold']))
        else:
            self.thresholds = {label: 0.5 for label in self.labels}
    
    def preprocess_image(self, image_data):
        """
        Preprocess the image for model input.
        
        Args:
            image_data: Raw image data from request
            
        Returns:
            processed_image: Preprocessed image ready for model input
            original_image: Original image for visualization
        """
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_image = image.copy()
        
        # Resize to model input size
        processed_image = cv2.resize(image, (224, 224))
        processed_image = processed_image.astype('float32') / 255.0
        
        return np.expand_dims(processed_image, axis=0), original_image
    
    def process_image(self, image_data):
        """
        Process the uploaded image and return the result with highlighted constellations.
        
        Args:
            image_data: Raw image data from request
            
        Returns:
            output_image_data: Processed image data
            detected_constellations: List of detected constellations with confidence scores
        """
        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_data)
        
        # Get predictions
        predictions = self.model.predict(input_tensor)
        
        # Get detected constellations
        detected_constellations = []
        for i, label in enumerate(self.labels):
            if predictions[0][i] > self.thresholds[label]:
                confidence = predictions[0][i] * 100
                detected_constellations.append((label, confidence))
        
        # Highlight detected constellations
        highlighted_image = self.highlight_constellations(original_image, detected_constellations)
        
        # Encode the output image
        _, buffer = cv2.imencode('.jpg', highlighted_image)
        output_image_data = buffer.tobytes()
        
        return output_image_data, detected_constellations
    
    def highlight_constellations(self, image, detected_constellations):
        """
        Add visual highlights and labels for detected constellations.
        
        Args:
            image: Original image
            detected_constellations: List of (constellation_name, confidence) tuples
            
        Returns:
            highlighted_image: Image with highlights and labels
        """
        highlighted_image = image.copy()
        
        # Add text with detected constellations
        for i, (const, conf) in enumerate(detected_constellations):
            # Position text at the top of the image
            y_position = 30 + (i * 30)
            cv2.putText(
                highlighted_image,
                f"{const} ({conf:.1f}%)",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return highlighted_image

# Initialize the detector globally
MODEL_PATH = r"model\constellation_model_densenet.keras"
detector = ConstellationDetector(MODEL_PATH)

# @app.route('/')
# def index():
#     """Render the upload page"""
#     return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_constellations():
    """Handle image upload and constellation detection"""
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return {'error': 'No image selected'}, 400
    
    try:
        # Read the image data
        image_data = image_file.read()
        
        # Process the image
        output_image_data, detected_constellations = detector.process_image(image_data)
        
        # Create response with both image and detections
        response_data = {
            'detections': [
                {'constellation': const, 'confidence': conf}
                for const, conf in detected_constellations
            ]
        }
        
        # Send both the processed image and detection results
        return send_file(
            io.BytesIO(output_image_data),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='processed_image.jpg'
        )
    
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)