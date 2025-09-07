# Face Expression Detection 🎭

This project uses **OpenCV** and a **Convolutional Neural Network (CNN)** model (Keras/TensorFlow) to detect human facial expressions in real-time from a webcam feed.

---

## 🚀 Features
- Detects faces using OpenCV’s Haar cascade classifier.
- Classifies emotions into:  
  **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**.
- Real-time webcam detection with bounding boxes and emotion labels.
- Includes test scripts to run predictions on single images.

---

## 📂 Project Structure
face_expression_detection/
├── main.py # Run real-time webcam emotion detection
├── test_predict.py # Test prediction on a single image
├── diagnose_model.py # Debugging script for model input/output
├── model.h5 # Trained CNN model (≈70 MB)
├── requirements.txt # Python dependencies
└── README.md # Project description & instructions

yaml
Copy code

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/wickedos/face_expression_detection.git
   cd face_expression_detection
Create and activate a virtual environment

bash
Copy code
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Ensure model.h5 is present

If you already have it, place it in the project root.

If not, download it from your own link (e.g., Google Drive) and save it as model.h5.

▶️ Run the Application
Webcam real-time detection

bash
Copy code
python main.py
Press q in the video window to quit.

Single image test

bash
Copy code
python test_predict.py path/to/image.jpg
Model diagnostics

bash
Copy code
python diagnose_model.py
📦 Dependencies
TensorFlow / Keras

OpenCV

NumPy

Pillow

(Install automatically via requirements.txt)

🙌 Acknowledgements
OpenCV for face detection.

Original dataset and CNN inspiration from FER2013.

Repository structure inspired by akmadan/Emotion_Detection_CNN.

📸 Demo
Add screenshots or GIFs of the app in action here if you’d like!

=======================
requirements.txt
tensorflow
opencv-python
numpy
Pillow

=======================
.gitignore
Ignore Python virtual environment
.venv/
pycache/
*.pyc

Ignore debug and log files
, debug_face.jpg
, run_output.txt

yaml
Copy code

---

✅ Next step after creating these three files:

```powershell
git add README.md requirements.txt .gitignore
git commit -m "Add README, requirements, and gitignore"
git push
