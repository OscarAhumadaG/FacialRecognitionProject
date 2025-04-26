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

## 🖼 How It Works

1. **Dataset Preparation**  
   Place images of known individuals in the `dataset/` folder. Each image filename should correspond to the person's name. This dataset is used to encode known faces for future recognition.

2. **Face Encoding**  
   The system processes the images and encodes the faces into a numerical format that can be easily compared. This encoding is stored and used for face recognition during the detection phase.

3. **Face Recognition**  
   The model detects faces in new images or video streams. Once faces are detected, the system compares the encoded features with those from the known dataset to identify the individual.

---

## 📈 Future Improvements

- **Enhanced Recognition Accuracy**  
  Improve the model's recognition accuracy by refining the preprocessing pipeline, such as better image alignment, data augmentation, or improved face encoding techniques.

- **Graphical User Interface (GUI)**  
  Develop a user-friendly GUI to simplify the usage of the facial recognition system, making it more accessible for non-technical users.

- **Deep Learning Integration**  
  Integrate advanced deep learning models for more robust face detection, such as **MTCNN** (Multi-task Cascaded Convolutional Networks), for better face alignment and detection in complex scenarios.

- **Multiple Camera Support**  
  Extend the system to support multiple camera inputs, enabling simultaneous face recognition from multiple video feeds for larger-scale applications.


📄 License
This repository is licensed under the MIT License.

