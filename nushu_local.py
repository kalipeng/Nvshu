# -*- coding: utf-8 -*-
"""
Nüshu Object Detection System - Local Version
Standalone version for local image processing using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import sys

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
        self.model = YOLO('yolov8n.pt')  # Using nano version for speed
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

    def draw_detections_with_custom_images(self, image, detections):
        """
        Draw object detections on image using custom Nüshu images

        Args:
            image: OpenCV image (BGR format)
            detections: List of (class_name, confidence, box) tuples

        Returns:
            OpenCV image with detections drawn
        """
        result_image = image.copy()

        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        for class_name, confidence, box in detections:
            # Get bounding box coordinates
            height, width = image.shape[:2]
            ymin, xmin, ymax, xmax = box
            
            # Convert normalized coordinates to pixel coordinates
            x1, y1 = int(xmin * width), int(ymin * height)
            x2, y2 = int(xmax * width), int(ymax * height)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get and resize Nüshu character image
            nushu_image = self.nushu_mapping.get_image_for_class(class_name)
            if nushu_image:
                # Calculate overlay size (smaller box)
                overlay_size = min(60, (x2-x1)//2, (y2-y1)//2)
                if overlay_size > 20:  # Only overlay if box is large enough
                    # Resize Nüshu image
                    nushu_resized = nushu_image.resize((overlay_size, overlay_size), Image.Resampling.LANCZOS)
                    
                    # Position the overlay (top-left corner of bounding box)
                    overlay_x = x1 + 5
                    overlay_y = y1 + 5
                    
                    # Convert result_image to PIL for overlay
                    pil_result = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                    
                    # Paste with transparency
                    pil_result.paste(nushu_resized, (overlay_x, overlay_y), nushu_resized)
                    
                    # Convert back to OpenCV
                    result_image = cv2.cvtColor(np.array(pil_result), cv2.COLOR_RGB2BGR)
            
            # Add text label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_image

    def process_image(self, image_path):
        """
        Process a single image and return detection results

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (original_image, result_image, detections)
        """
        print(f"Processing image: {image_path}")
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect objects
        detections = self.detect_objects(image)
        print(f"Found {len(detections)} objects")
        
        # Draw detections with Nüshu characters
        result_image = self.draw_detections_with_custom_images(image, detections)
        
        return image, result_image, detections

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Nüshu Object Detection System')
    parser.add_argument('--input', '-i', type=str, help='Input image path')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam input')
    parser.add_argument('--nushu_dir', '-d', type=str, default='nushu_images',
                       help='Directory containing Nüshu character images')
    
    args = parser.parse_args()
    
    # Initialize the system
    nushu_mapping = NushuCustomImageMapping(args.nushu_dir)
    detection_system = NushuObjectDetectionSystem(nushu_mapping)
    
    if args.webcam:
        # Use webcam
        print("Starting webcam... Press 'q' to quit")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in the frame
            detections = detection_system.detect_objects(frame)
            result_frame = detection_system.draw_detections_with_custom_images(frame, detections)
            
            cv2.imshow('Nüshu Object Detection', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    elif args.input:
        # Process single image
        try:
            original, result, detections = detection_system.process_image(args.input)
            
            # Display results
            print(f"\nDetected objects:")
            for class_name, confidence, _ in detections:
                print(f"- {class_name}: {confidence:.2f}")
            
            # Show images
            cv2.imshow('Original', original)
            cv2.imshow('Nüshu Detection Result', result)
            
            # Save result
            output_path = f"result_{os.path.basename(args.input)}"
            cv2.imwrite(output_path, result)
            print(f"\nResult saved to: {output_path}")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing image: {e}")
    
    else:
        print("Please specify --input for image file or --webcam for live detection")
        print("Use --help for more options")

if __name__ == "__main__":
    main() 