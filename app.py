import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle, os, base64, json
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import mediapipe as mp

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="Face Recognition + Liveness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Paths -----------------
EMB_PATH = "embeddings_arcface.pkl"
REGISTER_DIR = "registered_faces"
os.makedirs(REGISTER_DIR, exist_ok=True)

USERS_JSON = "users.json"
ATTEND_JSON = "attendance.json"

# ensure JSON files exist
if not os.path.exists(USERS_JSON):
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump([], f)
if not os.path.exists(ATTEND_JSON):
    with open(ATTEND_JSON, "w", encoding="utf-8") as f:
        json.dump([], f)

# ----------------- Static Files -----------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ----------------- Load or Init Embeddings -----------------
if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "rb") as f:
        known_faces = pickle.load(f)
    print(f"✅ Loaded embeddings: {list(known_faces.keys())}")
else:
    known_faces = {}
    print("⚠️ No embeddings found. Starting fresh...")

# ----------------- Initialize Face Engine -----------------
app_insight = FaceAnalysis(providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))

history_log = []

# ----------------- Utility Functions -----------------
def recognize_face(face_emb, threshold=0.6):
    min_dist = float("inf")
    identity = "Unknown"
    for name, emb in known_faces.items():
        dist = np.linalg.norm(face_emb - emb)
        if dist < min_dist:
            min_dist, identity = dist, name
    if min_dist > threshold:
        identity = "Unknown"
    return identity, min_dist

# JSON-based user store
def load_users():
    try:
        with open(USERS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_users(users_list):
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(users_list, f, ensure_ascii=False, indent=2)

def ensure_user_in_db(name: str):
    users = load_users()
    # check if exists by name
    for u in users:
        if u.get("name") == name:
            return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    users.append({"name": name, "created_at": now})
    save_users(users)

# attendance append
def append_attendance(record: dict):
    try:
        with open(ATTEND_JSON, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        arr = []
    arr.append(record)
    # keep file trimmed if you want, or keep all
    with open(ATTEND_JSON, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

# --------------- Liveness Detection (blink + movement) ---------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_pts):
    v1 = np.linalg.norm(np.array(eye_pts[1]) - np.array(eye_pts[5]))
    v2 = np.linalg.norm(np.array(eye_pts[2]) - np.array(eye_pts[4]))
    h = np.linalg.norm(np.array(eye_pts[0]) - np.array(eye_pts[3]))
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def compute_liveness(frames_bgr: list, ear_thresh: float = 0.20, movement_thresh_px: float = 10.0):
    if not frames_bgr:
        return False, 0, 0.0
    blinks = 0
    ear_below = False
    nose_positions = []
    with mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1) as mesh:
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue
            lm = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            left_eye = [(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE_IDX]
            right_eye = [(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE_IDX]
            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)
            ear = (ear_left + ear_right) / 2.0
            if ear < ear_thresh and not ear_below:
                ear_below = True
            if ear >= ear_thresh and ear_below:
                blinks += 1
                ear_below = False
            nose = lm[1]
            nose_positions.append((nose.x * w, nose.y * h))
    total_move = 0.0
    for i in range(1, len(nose_positions)):
        total_move += np.linalg.norm(np.array(nose_positions[i]) - np.array(nose_positions[i-1]))
    is_live = blinks >= 1 and total_move >= movement_thresh_px
    return is_live, blinks, float(total_move)

# =============================================================
# ✅ API 1: Register Face
# =============================================================
@app.post("/register")
async def register_face(request: Request):
    data = await request.json()
    name = data.get("name")
    image_b64 = data.get("image_base64")

    if not name or not image_b64:
        return JSONResponse({"status": "error", "message": "Name or image missing"})

    try:
        img_data = base64.b64decode(image_b64.split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Invalid image: {e}"})

    faces = app_insight.get(img)
    if not faces:
        return JSONResponse({"status": "error", "message": "No face detected"})

    known_faces[name] = faces[0].normed_embedding
    with open(EMB_PATH, "wb") as f:
        pickle.dump(known_faces, f)
    cv2.imwrite(os.path.join(REGISTER_DIR, f"{name}.jpg"), img)
    ensure_user_in_db(name)

    print(f"✅ Face registered for {name}")
    return JSONResponse({"status": "success", "message": f"Face registered for {name}"})

# =============================================================
# ✅ API 2: Detect Face (Recognition + Liveness)
# =============================================================
@app.post("/detect_face")
async def detect_face(request: Request):
    data = await request.json()
    image_b64 = data.get("image")

    if not image_b64:
        return JSONResponse({"name": "Unknown", "confidence": 0, "isLive": False})

    img_data = base64.b64decode(image_b64.split(",")[1])
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    faces = app_insight.get(img)
    if not faces:
        return JSONResponse({"name": "Unknown", "confidence": 0, "isLive": False})

    face_emb = faces[0].normed_embedding
    name, dist = recognize_face(face_emb)
    conf = max(0, 1 - dist)

    history_log.append({
        "name": name,
        "confidence": float(conf),
        "isLive": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    if len(history_log) > 50:
        history_log.pop(0)

    return JSONResponse({
        "name": name,
        "confidence": float(conf),
        "isLive": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# =============================================================
# ✅ API 2b: Recognize with Liveness (frames + location)
# =============================================================
@app.post("/recognize_live")
async def recognize_live(request: Request):
    data = await request.json()
    images_b64 = data.get("images", [])
    location = data.get("location", {})
    lat = location.get("lat") if isinstance(location, dict) else None
    lon = location.get("lon") if isinstance(location, dict) else None

    if not images_b64 or not isinstance(images_b64, list):
        return JSONResponse({"status": "error", "message": "images must be a non-empty list"})

    frames = []
    for s in images_b64:
        try:
            part = s.split(",", 1)[1] if "," in s else s
            buf = base64.b64decode(part)
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        except Exception:
            continue

    if not frames:
        return JSONResponse({"status": "error", "message": "failed to decode frames"})

    is_live, blinks, move_px = compute_liveness(frames)

    key_frame = frames[len(frames)//2]
    faces = app_insight.get(key_frame)
    if not faces:
        for fr in frames:
            faces = app_insight.get(fr)
            if faces:
                key_frame = fr
                break
    if not faces:
        return JSONResponse({"name": "Unknown", "confidence": 0.0, "isLive": is_live, "blinks": blinks, "movement": move_px})

    face_emb = faces[0].normed_embedding
    name, dist = recognize_face(face_emb)
    confidence = max(0.0, 1.0 - float(dist))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "name": name,
        "confidence": round(float(confidence), 3),
        "is_live": bool(is_live),
        "blinks": int(blinks),
        "movement_px": round(float(move_px), 2),
        "latitude": float(lat) if lat is not None else None,
        "longitude": float(lon) if lon is not None else None,
        "timestamp": now
    }
    append_attendance(record)

    if name != "Unknown":
        ensure_user_in_db(name)

    return JSONResponse({
        "name": name,
        "confidence": round(float(confidence), 3),
        "isLive": bool(is_live),
        "blinks": int(blinks),
        "movement": round(float(move_px), 2),
        "timestamp": now,
        "location": {"lat": lat, "lon": lon}
    })

# =============================================================
# ✅ API 3: Recognize Face (Deep Learning inference only)
# =============================================================
@app.post("/recognize_face")
async def recognize(request: Request):
    data = await request.json()
    image_b64 = data.get("image")

    if not image_b64:
        return JSONResponse({"status": "error", "message": "No image provided"})

    try:
        img_data = base64.b64decode(image_b64.split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        faces = app_insight.get(img)
        if not faces:
            return JSONResponse({"name": "Unknown", "confidence": 0})
        face_emb = faces[0].normed_embedding
        name, dist = recognize_face(face_emb)
        confidence = max(0, 1 - dist)
        return JSONResponse({
            "name": name,
            "confidence": round(float(confidence), 3),
            "distance": round(float(dist), 3)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# =============================================================
# ✅ API 4: Live Webcam Stream
# =============================================================
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break
        faces = app_insight.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            name, dist = recognize_face(face.normed_embedding)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({dist:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# =============================================================
# ✅ API 5: Recognition History
# =============================================================
@app.get("/history")
def get_history():
    try:
        with open(ATTEND_JSON, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        rows = []
    # return newest first
    rows_sorted = list(reversed(rows))[:50]
    return JSONResponse({"history": rows_sorted})

# =============================================================
# ✅ Root Endpoint
# =============================================================
@app.get("/")
def root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "✅ Face API Running — /video_feed for live feed. Add static/index.html for UI."}
