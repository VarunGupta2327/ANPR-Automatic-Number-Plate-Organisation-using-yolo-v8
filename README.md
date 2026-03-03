# ANPR-Automatic-Number-Plate-Organisation-using-yolo-v8
🚗 ANPR (Automatic Number Plate Recognition) using YOLOv8

This project implements an Automatic Number Plate Recognition (ANPR) system using YOLOv8 for license plate detection and OCR (Optical Character Recognition) for extracting the plate number.

The system can process:

📷 Images

🎥 Videos

📹 Real-time webcam feed

It detects vehicle number plates and extracts text automatically.

🧠 How It Works

Vehicle Image Input

License Plate Detection using YOLOv8

Crop Detected Plate

Apply OCR to extract characters

Display / Store Plate Number

🛠️ Technologies Used

Python

YOLOv8 (Ultralytics)

OpenCV

EasyOCR / Tesseract OCR

NumPy

📦 Installation
1️⃣ Clone Repository
git clone https://github.com/VarunGupta2327/ANPR-YOLOv8.git
cd ANPR-YOLOv8
2️⃣ Install Dependencies
pip install ultralytics opencv-python easyocr numpy

If using Tesseract:

pip install pytesseract
🚀 Usage
Run on Image
python detect.py --source image.jpg
Run on Video
python detect.py --source video.mp4
Run on Webcam
python detect.py --source 0
📁 Project Structure
ANPR-YOLOv8/
│
├── models/
│   └── best.pt
├── detect.py
├── ocr.py
├── utils.py
├── requirements.txt
└── README.md
📊 Model Training (Optional)

To train custom license plate detection model:

yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
📸 Features

✅ Real-time number plate detection
✅ High accuracy using YOLOv8
✅ Automatic text extraction
✅ Works with images, videos, webcam
✅ Can store results in CSV/database

🎯 Applications

Smart Parking Systems

Toll Booth Automation

Traffic Monitoring

Security & Surveillance

Vehicle Entry Management

⚠️ Disclaimer

This project is for educational and research purposes only.
Ensure compliance with local privacy and surveillance laws before deploying in real-world applications.
