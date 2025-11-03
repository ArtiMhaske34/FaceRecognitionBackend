
# 🧠 FaceSecure: AI-Based Facial Recognition & Liveness Detection System

## 📖 Overview
**FaceSecure** is an advanced AI-driven facial recognition and liveness detection system built using **FastAPI**, **InsightFace (ArcFace)**, and **MediaPipe**. It enables secure face-based authentication by ensuring the detected face is real and live, not a static image or spoof.

The system supports face registration, recognition, liveness verification, and attendance logging through a simple REST API or web interface.

---

## ✨ Key Features
- Real-time Face Detection & Recognition
- ArcFace Embedding Extraction for High Accuracy
- Liveness Detection via Blink & Head Movement
- User Registration with Image & Embedding Storage
- JSON-based Attendance Logging
- FastAPI REST API – lightweight & scalable
- Works on both PC and Mobile (same Wi-Fi network)

---

## 🧩 Project Structure
```
FaceSecure/
│
├── app.py                    # Main FastAPI backend
├── embeddings_arcface.pkl    # Stores registered face embeddings
├── registered_faces/         # Saved user face images
├── users.json                # Registered user data
├── attendance.json           # Attendance records
├── static/
│   └── index.html            # Optional web interface
├── register_face.py          # Face registration helper script
├── requirements.txt          # Dependencies list (note: actual file is Requirements.txt.txt)
└── README.md                 # Documentation
```

---

## ⚙️ Installation & Setup

### Requirements
- Python 3.8 – 3.11
- pip (Python package manager)
- Camera access (for registration/recognition)

### Install Dependencies
```bash
pip install -r Requirements.txt.txt
```

### Sample `Requirements.txt.txt`
```txt
fastapi
uvicorn
opencv-python
insightface
mediapipe
numpy
pillow
requests
python-multipart
pydantic
```

---

## 🚀 Running the Backend

Navigate to the project folder:
```bash
cd FaceSecure
```

### For Laptop Only (Local Access)
```bash
uvicorn app:app --reload
```
Open your browser: `http://localhost:8000`

### For Mobile + Laptop (Same Wi-Fi Network)
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open your browser:
- On PC: `http://localhost:8000`
- On Mobile (same Wi-Fi): `http://<your-IPv4-address>:8000`

---

## 👤 Registering a New Face

Run the registration script to enroll a new user:
```bash
py register_face.py
```

**Steps:**
1. Webcam window opens
2. Press **SPACE** to capture your face
3. Enter your name
4. The system saves your image and updates `embeddings_arcface.pkl`

---

## 🔍 API Endpoints

Once registered, the backend automatically recognizes known users through these endpoints:

- `/detect_face` → Recognition API
- `/recognize_live` → Recognition + Liveness detection
- `/video_feed` → Live preview stream
- `/history` → View recent recognition logs

---

## 🧠 Tech Stack

| Component              | Technology                      |
| ---------------------- | ------------------------------- |
| **Backend**            | FastAPI                         |
| **Face Recognition**   | InsightFace (ArcFace model)     |
| **Liveness Detection** | MediaPipe Face Mesh             |
| **Frontend**           | Static HTML/JS (optional)       |
| **Storage**            | Pickle + JSON                   |
| **API Testing**        | Postman / Browser / Android App |

---

## 📱 Mobile Access

To run the app on your mobile browser (same Wi-Fi):

1. Find your IPv4 address using:
   ```bash
   ipconfig
   ```
   Example: `192.168.1.116`

2. Run FastAPI:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. Open on your phone:
   ```
   http://192.168.1.116:8000
   ```

---

## 🔒 Security Notes

- Face embeddings are stored locally in `embeddings_arcface.pkl`
- Registered face images are stored under `registered_faces/`
- No external cloud storage is used (fully offline support)
- Optionally integrable with SQL or Firebase in future versions

---

## 👩‍💻 Author

**Developed by:** Arti Mhaske & Team  
**Project Title:** FaceSecure – AI-Based Facial Recognition & Liveness Detection  
**Year:** 2025

---

## 🏁 Future Enhancements

- Cloud database integration (Firebase / PostgreSQL)
- Android & Web App synchronization
- Multi-face recognition in real-time video streams
- Role-based authentication (Admin / User)
- Dashboard analytics for attendance tracking

---

## 📜 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it for educational or research purposes.
