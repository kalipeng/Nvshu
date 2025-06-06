# Nüshu Object Detection Web Application

A Flask-based web application that detects objects in images and displays them with traditional Nüshu characters.

## About Nüshu: The World's Only Women's Script

Nüshu, literally meaning "women's writing" in Chinese, is considered the world's only writing system created and used exclusively by women. This unique script originated in China's Jiangyong county, Hunan province, during the 19th century among rural women in the Xiao River valley.

This remarkable writing system served as an emotional outlet and source of hope for women in traditional Chinese society. Through Nüshu, women could express their feelings, share stories, and support each other in a way that was uniquely their own. 

### Acknowledgments

We extend our heartfelt gratitude to:

- **The Nushu Coder's Group on GitHub** and contributors to the [Unicode Nüshu project](https://github.com/nushu-script/unicode_nushu) for their invaluable work in digitizing and preserving this precious cultural heritage
- **All open-source developers** who have contributed to Nüshu digitization and Unicode standardization efforts
- **Traditional Nüshu keepers and researchers** who have dedicated their lives to preserving this unique women's script
- **Modern technology communities** working to bridge ancient culture with contemporary digital tools

This project honors the legacy of Nüshu by combining traditional women's script with modern AI technology, creating new ways to experience and appreciate this cultural treasure.

## Features

- **Image Upload**: Upload images for object detection with Nüshu character overlay
- **Webcam Detection**: Real-time object detection using your camera
- **Nüshu Character Gallery**: View all available Nüshu character mappings

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Nüshu Character Images**:
   - The `nushu_images/` directory contains Nüshu character images
   - Each image should be named after the object class (e.g., `person.png`, `chair.jpg`)
   - Supported formats: PNG, JPG, JPEG

3. **Run the Application**:
   ```bash
   python nushu_web_app.py
   ```

4. **Access the Web App**:
   - Open your browser and go to: `http://localhost:8000`

## Usage

### Upload Image Detection
1. Click "Upload Image" on the main page
2. Drag and drop an image or click "Choose Image" to select a file
3. Wait for processing (the AI model will detect objects)
4. View results with original image, annotated image, and Nüshu character list

### Webcam Detection
1. Click "Use Webcam" from the navigation
2. Click "Start Camera" to access your webcam
3. Click "Capture & Analyze" to detect objects in the current frame
4. View live detections with Nüshu characters in the sidebar

### Nüshu Character Gallery
1. Click "View Nüshu Characters" to see all available character mappings
2. Each character is labeled with its corresponding object class

## File Structure

```
Nvshu/
├── nushu_web_app.py          # Main Flask application
├── nvshu_local.py            # Local version (command line)
├── requirements.txt          # Python dependencies
├── templates/
│   ├── index.html           # Main page template
│   └── webcam.html          # Webcam page template
├── nushu_images/            # Nüshu character images
├── uploads/                 # Uploaded images (auto-created)
└── results/                 # Processed results (auto-created)
```

## Supported Object Classes

The system can detect and map Nüshu characters for various objects including:
- person, laptop, cell phone, microwave, book, chair, cup, bottle
- car, horse, sheep, clock, bowl, backpack, knife, refrigerator
- And many more COCO dataset classes

## Technical Details

- **Object Detection**: Uses YOLOv8 (ultralytics) for accurate and fast object detection
- **Web Framework**: Flask with modern HTML/CSS/JavaScript
- **Image Processing**: OpenCV and PIL for image manipulation
- **Real-time Processing**: WebRTC for camera access and real-time detection
- **AI Model**: YOLOv8 nano (yolov8n.pt) for optimal speed/accuracy balance

### YOLO Model Options

The system uses YOLOv8 nano by default for fast processing. You can change the model in the code for different accuracy/speed trade-offs:

- **yolov8n.pt** (nano) - Fastest, good accuracy 
- **yolov8s.pt** (small) - Balanced speed and accuracy
- **yolov8m.pt** (medium) - Higher accuracy, slower
- **yolov8l.pt** (large) - Very high accuracy, slower
- **yolov8x.pt** (extra large) - Highest accuracy, slowest 

To change the model, edit line 108 in `nushu_web_app.py`:
```python
self.model = YOLO('yolov8s.pt')  # Change to desired model
```

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

*Note: Webcam features require HTTPS in production or localhost for development*

## Troubleshooting

1. **Port 8000 already in use**: Change the port in `nushu_web_app.py` line 431
2. **Camera not working**: Ensure browser permissions are granted for camera access
3. **Slow processing**: The first detection may take longer as the AI model loads
4. **No objects detected**: Try images with clearer, well-lit objects

