# -*- coding: utf-8 -*-
"""
Nüshu Object Detection Web Application
Flask-based web interface for the Nüshu object detection system
"""

import os
import io
import base64
import time
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Import our custom classes from the same file since import might cause issues
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Copy the classes from nvshu_local.py
class NushuCustomImageMapping:
    def __init__(self, nushu_images_dir="nushu_images"):
        """
        Initialize the Nüshu Custom Image Mapping

        Args:
            nushu_images_dir: Directory containing Nüshu character images
                              Each image should be named as "class_name.png" (or .jpg)
        """
        self.nushu_images_dir = nushu_images_dir
        self.nushu_images = {}
        self.default_image = None

        # Create the directory if it doesn't exist
        if not os.path.exists(nushu_images_dir):
            os.makedirs(nushu_images_dir)
            print(f"Created directory: {nushu_images_dir}")

        # Load COCO class names (simplified for demo)
        self.coco_classes = [
            'person', 'laptop', 'cell phone', 'microwave', 'book', 'chair', 'cup', 'bottle'
        ]

    def load_images(self):
        """Load all Nüshu character images from the directory"""
        print(f"Loading Nüshu character images from {self.nushu_images_dir}...")

        # Check if directory exists
        if not os.path.exists(self.nushu_images_dir):
            print(f"Warning: Directory {self.nushu_images_dir} does not exist!")
            return

        # Load all images from the directory
        for filename in os.listdir(self.nushu_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract class name from filename (without extension)
                class_name = os.path.splitext(filename)[0].lower().replace('_', ' ')

                # Load the image
                try:
                    image_path = os.path.join(self.nushu_images_dir, filename)
                    image = Image.open(image_path).convert("RGBA")
                    self.nushu_images[class_name] = image
                    print(f"Loaded image for class: {class_name}")

                    # Use the first image as default if not set
                    if self.default_image is None:
                        self.default_image = image
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")

        # If no images were loaded, create a placeholder default image
        if not self.nushu_images:
            print("No images found. Creating a placeholder default image.")
            self.default_image = self.create_placeholder_image()

        print(f"Loaded {len(self.nushu_images)} Nüshu character images.")

    def create_placeholder_image(self):
        """Create a placeholder image for when custom images are not available"""
        img = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        # Use a simple rectangle as placeholder since we may not have Chinese fonts
        draw.rectangle([20, 20, 80, 80], fill=(255, 255, 0, 255))
        draw.text((30, 40), "Nu", fill=(0, 0, 0, 255))
        return img

    def get_image_for_class(self, class_name):
        """
        Get the Nüshu character image for a class

        Args:
            class_name: The class name to get the image for

        Returns:
            PIL Image containing the Nüshu character
        """
        # Try exact match first
        if class_name.lower() in self.nushu_images:
            return self.nushu_images[class_name.lower()]

        # Try to find similar class names
        for name in self.nushu_images:
            if name in class_name.lower() or class_name.lower() in name:
                return self.nushu_images[name]

        # Use default image if no match found
        return self.default_image

class NushuObjectDetectionSystem:
    def __init__(self, nushu_mapping):
        """
        Initialize the Nüshu Object Detection System

        Args:
            nushu_mapping: NushuCustomImageMapping object
        """
        # Load the Nüshu data
        self.nushu_mapping = nushu_mapping
        self.nushu_mapping.load_images()

        # Load YOLOv8 model
        print("Loading YOLOv8 object detection model...")
        self.model = YOLO('yolov8n.pt')  # Using nano version for speed, can use 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' for better accuracy
        print("YOLOv8 model loaded successfully!")

        # YOLO COCO class names
        self.coco_class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect_objects(self, image):
        """
        Detect objects in the given image using YOLOv8

        Args:
            image: Image to detect objects in (BGR format from OpenCV)

        Returns:
            List of (class_name, confidence, box) tuples
        """
        # Run YOLO inference
        results = self.model(image, conf=0.3)  # confidence threshold of 0.3
        
        detections = []
        
        # Process YOLO results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates (normalized to 0-1)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    h, w = image.shape[:2]
                    
                    # Convert to normalized coordinates for consistency
                    norm_box = [y1/h, x1/w, y2/h, x2/w]  # [ymin, xmin, ymax, xmax]
                    
                    # Get class info
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.coco_class_names[class_id] if class_id < len(self.coco_class_names) else f"class_{class_id}"
                    
                    detections.append((class_name, confidence, norm_box))
        
        return detections

    def get_class_name(self, class_id):
        """Get class name from COCO class ID"""
        if 0 <= class_id < len(self.coco_class_names):
            return self.coco_class_names[class_id]
        return f"class_{class_id}"

    def draw_detections_with_custom_images(self, image, detections):
        """
        Draw object detections on image using custom Nüshu images

        Args:
            image: OpenCV image (BGR format)
            detections: List of (class_name, confidence, box) tuples

        Returns:
            Image with custom Nüshu images and labels drawn (BGR format)
        """
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Get image dimensions
        height, width, _ = image.shape

        # Try to use default font
        try:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        except:
            font = None
            small_font = None

        # Draw each detection
        for class_name, confidence, box in detections:
            # Get coordinates (box is [y_min, x_min, y_max, x_max] in normalized coordinates)
            y_min, x_min, y_max, x_max = box
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)

            # Draw bounding box
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 255, 0), width=3)

            # Get Nüshu image for this class
            nushu_img = self.nushu_mapping.get_image_for_class(class_name)

            if nushu_img:
                # Resize the Nüshu image to an appropriate size
                nushu_size = min(80, (y_max - y_min) // 2)
                nushu_size = max(40, nushu_size)  # Ensure minimum size
                nushu_img = nushu_img.resize((nushu_size, nushu_size), Image.LANCZOS)

                # Create a background for text
                text = f"{class_name} ({confidence:.2f})"
                
                # Position the label at the top of the bounding box
                label_x = x_min
                label_y = max(0, y_min - 75)

                # Draw background rectangle for text and image
                bg_width = max(150, nushu_size + 10)
                bg_height = 30 + nushu_size
                
                draw.rectangle(
                    [(label_x, label_y), (label_x + bg_width, label_y + bg_height)],
                    fill=(0, 0, 0, 180)
                )

                # Draw text
                if font:
                    draw.text(
                        (label_x + 5, label_y + 5),
                        text,
                        font=small_font,
                        fill=(255, 255, 255)
                    )
                else:
                    draw.text(
                        (label_x + 5, label_y + 5),
                        text,
                        fill=(255, 255, 255)
                    )

                # Paste the Nüshu image onto the main image
                pil_image.paste(
                    nushu_img,
                    (label_x + 5, label_y + 35),
                    nushu_img
                )

        # Convert back to OpenCV image
        result_image = np.array(pil_image)
        return cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    def process_image(self, image_path):
        """
        Process a single image file

        Args:
            image_path: Path to the image file

        Returns:
            Processed image with Nüshu labels and list of detections
        """
        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, []

        # Detect objects
        detections = self.detect_objects(image)

        # Draw detections with custom Nüshu images
        annotated_image = self.draw_detections_with_custom_images(image, detections)

        return annotated_image, detections

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('nushu_images', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables for the detection system
nushu_system = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_detection_system():
    """Initialize the Nüshu detection system"""
    global nushu_system
    print("Initializing Nüshu detection system...")
    nushu_mapping = NushuCustomImageMapping("nushu_images")
    nushu_system = NushuObjectDetectionSystem(nushu_mapping)
    print("Detection system initialized!")

def image_to_base64(image):
    """Convert PIL image to base64 string for web display"""
    if isinstance(image, np.ndarray):
        # Convert OpenCV image to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            annotated_image, detections = nushu_system.process_image(filepath)
            
            if annotated_image is not None:
                # Save the result
                result_filename = f"result_{timestamp}.jpg"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                cv2.imwrite(result_path, annotated_image)
                
                # Convert images to base64 for display
                original_img = cv2.imread(filepath)
                original_b64 = image_to_base64(original_img)
                result_b64 = image_to_base64(annotated_image)
                
                # Process detection results
                detection_results = []
                for class_name, confidence, box in detections:
                    # Get Nüshu image for this class
                    nushu_img = nushu_system.nushu_mapping.get_image_for_class(class_name)
                    nushu_b64 = image_to_base64(nushu_img) if nushu_img else None
                    
                    detection_results.append({
                        'class_name': class_name,
                        'confidence': round(confidence, 2),
                        'nushu_image': nushu_b64
                    })
                
                return jsonify({
                    'success': True,
                    'original_image': original_b64,
                    'result_image': result_b64,
                    'detections': detection_results,
                    'num_objects': len(detections)
                })
            else:
                return jsonify({'error': 'Failed to process image'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/nushu_images')
def show_nushu_images():
    """Show available Nüshu character images"""
    nushu_images = []
    for class_name, img in nushu_system.nushu_mapping.nushu_images.items():
        img_b64 = image_to_base64(img)
        nushu_images.append({
            'class_name': class_name,
            'image': img_b64
        })
    
    return jsonify({
        'images': nushu_images,
        'total': len(nushu_images)
    })

@app.route('/webcam')
def webcam_page():
    """Webcam detection page"""
    return render_template('webcam.html')

@app.route('/process_webcam_frame', methods=['POST'])
def process_webcam_frame():
    """Process a frame from webcam"""
    try:
        # Get the image data from the request
        image_data = request.json['image']
        
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the image
        detections = nushu_system.detect_objects(image)
        annotated_image = nushu_system.draw_detections_with_custom_images(image, detections)
        
        # Convert result to base64
        result_b64 = image_to_base64(annotated_image)
        
        # Process detection results
        detection_results = []
        for class_name, confidence, box in detections:
            nushu_img = nushu_system.nushu_mapping.get_image_for_class(class_name)
            nushu_b64 = image_to_base64(nushu_img) if nushu_img else None
            
            detection_results.append({
                'class_name': class_name,
                'confidence': round(confidence, 2),
                'nushu_image': nushu_b64
            })
        
        return jsonify({
            'success': True,
            'result_image': result_b64,
            'detections': detection_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Nüshu Object Detection Web Application...")
    init_detection_system()
    print("Web server starting on http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=8000) 