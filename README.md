# PneumoDetect-AI

PneumoDetect-AI is an AI-powered web application that detects pneumonia from chest X-ray images using deep learning and computer vision techniques.

The system provides instant AI predictions, confidence scores, Grad-CAM visual explanations, and downloadable medical reports for doctors and healthcare environments.

---

## Features

- AI-based Pneumonia Detection
- Chest X-ray Upload System
- Grad-CAM Heatmap Visualization
- Confidence Score Prediction
- PDF Diagnostic Report Generation
- Doctor Authentication System
- Admin Dashboard
- User Management System
- Activity Logs & Monitoring
- Report History Management
- Responsive UI Design

---

## Tech Stack

### Backend
- Python
- Flask
- SQLite

### AI / Machine Learning
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy

### Frontend
- HTML
- CSS
- JavaScript
- Chart.js

### Additional Libraries
- ReportLab
- tf-keras-vis

---

## AI Model Information

The model uses MobileNetV2 Transfer Learning architecture for multi-class classification.

### Classification Categories
- NORMAL
- BACTERIAL Pneumonia
- VIRAL Pneumonia

---

## System Modules

### Doctor Panel
- Login & Registration
- Upload X-ray Images
- View AI Predictions
- Download PDF Reports
- Patient Report History

### Admin Panel
- User Management
- Activity Monitoring
- System Logs
- Report Analytics
- Dashboard Statistics

---

## Dataset

Dataset used for training:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Note:
The dataset is not included in this repository due to large file size.

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/PneumoDetect-AI.git
```

### 2. Open Project Folder

```bash
cd PneumoDetect-AI
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
python app.py
```

### 5. Open Browser

```text
http://127.0.0.1:5000
```

---

## Project Structure

```text
PneumoDetect-AI/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── doctor_dashboard.html
│   ├── admin_dashboard.html
│   └── ...
│
├── static/
│   ├── CSS files
│   ├── Images
│   └── uploads/
│
├── dataset/          # Not uploaded
├── model/            # Ignored
└── database/         # Ignored
```

---

## Future Improvements

- Cloud Deployment
- Docker Support
- Multi-Disease Detection
- Real-time Camera Scanning
- Mobile Application
- Improved Model Accuracy
- Email Notifications

---

## License

MIT License

---

## Author

Jayesh Pawar
