import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import pandas as pd

class NushuObjectDetectionSystem:
    def __init__(self, nushu_data_path='nushu_mapping.csv'):
        """
        Initialize the Nüshu Object Detection System
        
        Args:
            nushu_data_path: Path to the CSV file containing Chinese to Nüshu mappings
        """
        # Load the Nüshu data
        self.nushu_mapping = pd.read_csv(nushu_data_path)
        
        # Load pre-trained object detection model (using MobileNetV2 for simplicity)
        self.model = MobileNetV2(weights='imagenet')
        
        # Initialize camera
        self.camera = None
        
    def start_camera(self, camera_id=0):
        """Start the camera for real-time object detection"""
        self.camera = cv2.VideoCapture(camera_id)
        
    def stop_camera(self):
        """Stop the camera"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def detect_objects(self, image):
        """
        Detect objects in the given image
        
        Args:
            image: Image to detect objects in (BGR format from OpenCV)
            
        Returns:
            List of (class_name, confidence) tuples
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to match model's expected input
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # Prepare input for the model
        image_array = np.expand_dims(image_resized, axis=0)
        image_preprocessed = preprocess_input(image_array)
        
        # Predict objects
        predictions = self.model.predict(image_preprocessed)
        
        # Decode the predictions (top 5 results)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Format results as (class_name, confidence)
        results = [(label, float(confidence)) for (_, label, confidence) in decoded_predictions]
        
        return results
    
    def translate_to_nushu(self, word):
        """
        Translate a word from Chinese/English to Nüshu
        
        Args:
            word: Word to translate
            
        Returns:
            Nüshu representation of the word if found, or the original word
        """
        # First try direct match in the mapping
        matching_rows = self.nushu_mapping[self.nushu_mapping['Chinese'] == word]
        
        if not matching_rows.empty:
            return matching_rows.iloc[0]['Nushu']
        
        # If no direct match, try to find the word in any Chinese entry
        for _, row in self.nushu_mapping.iterrows():
            if word in row['Chinese']:
                return row['Nushu']
        
        # If still no match, check if there's a translation for the word in English
        for _, row in self.nushu_mapping.iterrows():
            # This assumes there's an English column, might need modification
            if 'English' in self.nushu_mapping.columns and row['English'] == word:
                return row['Nushu']
        
        # If no translation found, return the original word
        return word
    
    def draw_nushu_labels(self, image, detections):
        """
        Draw Nüshu labels for detected objects
        
        Args:
            image: Image to draw on
            detections: List of (class_name, confidence) tuples
            
        Returns:
            Image with Nüshu labels drawn
        """
        image_with_labels = image.copy()
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Draw labels
        for i, (class_name, confidence) in enumerate(detections):
            # Translate class name to Nüshu
            nushu_text = self.translate_to_nushu(class_name)
            
            # Calculate position for the label
            y_position = 50 + i * 30
            
            # Draw a background rectangle for the text
            cv2.rectangle(image_with_labels, (10, y_position - 25), (300, y_position + 5), (0, 0, 0), -1)
            
            # Draw the text (class name and Nüshu)
            text = f"{class_name} ({confidence:.2f}): {nushu_text}"
            cv2.putText(image_with_labels, text, (15, y_position), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image_with_labels
    
    def process_frame(self, frame):
        """
        Process a single frame from the camera
        
        Args:
            frame: Frame to process
            
        Returns:
            Processed frame with Nüshu labels
        """
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Draw Nüshu labels
        frame_with_labels = self.draw_nushu_labels(frame, detections)
        
        return frame_with_labels
    
    def run_detection_loop(self):
        """Run the main detection loop using the camera"""
        if self.camera is None:
            self.start_camera()
            
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            return
        
        while True:
            # Read a frame from the camera
            ret, frame = self.camera.read()
            
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the result
            cv2.imshow('Nüshu Object Detection', processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.stop_camera()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path):
        """
        Process a single image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed image with Nüshu labels
        """
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # Process the image
        processed_image = self.process_frame(image)
        
        return processed_image

# Example usage
if __name__ == "__main__":
    # Create a mapping CSV from our data
    # This is a simplified example - in a real application, you'd need a more comprehensive mapping
    
    # Create a sample mapping
    mapping_data = {
        'Chinese': ['人', '花', '山', '水', '树', '鸟', '鱼', '书', '车', '眼'],
        'Nushu': ['𛅳', '𛈣', '𛇏', '𛅸', '𛋙', '𛊹', '𛉄', '𛈬', '𛊌', '𛋃'],
        'English': ['person', 'flower', 'mountain', 'water', 'tree', 'bird', 'fish', 'book', 'car', 'eye']
    }
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv('nushu_mapping.csv', index=False)
    
    # Initialize the system
    nushu_system = NushuObjectDetectionSystem('nushu_mapping.csv')
    
    # Run the detection loop (uncomment to use camera)
    # nushu_system.run_detection_loop()
    
    # Or process a single image
    # result = nushu_system.process_image('sample_image.jpg')
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
