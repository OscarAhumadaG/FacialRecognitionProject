# Facial Recognition Project 🧠📸

Welcome to the **Facial Recognition Project** repository!  
This project showcases a simple implementation of **face detection** and **face recognition** using popular Python libraries.

---

## 🛠 Project Overview

This project demonstrates:

- **Face Detection**: Identifying faces in images or real-time video.
- **Face Recognition**: Matching detected faces against a known dataset.

It leverages **OpenCV** and **face_recognition** libraries to perform both tasks efficiently.

---

## 🔥 Technologies Used

- Python 3.10
- OpenCV
- face_recognition (based on dlib)
- Numpy

---

## 📂 Project Structure


---

---

## 🚀 Getting Started

1. **Clone the repository**:
```bash
git clone https://github.com/OscarAhumadaG/FacialRecognitionProject.git
cd FacialRecognitionProject
Install dependencies:
```
```
bash
Copy
Edit
pip install -r requirements.txt
(If there's no requirements.txt, install manually: opencv-python, face_recognition, numpy)
```

Run the scripts:

To recognize faces in an image:
```
bash
Copy
Edit
python recognize_faces_images.py
```

To recognize faces from your webcam (real-time video):
```
bash
Copy
Edit
python recognize_faces_video.py
```

🖼 How It Works
Dataset Preparation: Place images of known individuals in the dataset/ folder. Each image filename should correspond to the person's name.

Face Encoding: The scripts encode known faces.

Recognition: The model detects faces in new images or videos and matches them with the known dataset.

📈 Future Improvements
Improve recognition accuracy with better preprocessing.

Add GUI interface for easier use.

Integrate deep learning models for face detection (e.g., MTCNN).

Expand to multiple camera inputs.

📄 License
This repository is licensed under the MIT License.

✨ About Me
Oscar Darío Ahumada Gómez
LinkedIn • GitHub
