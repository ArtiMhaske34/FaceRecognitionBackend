import cv2
import base64
import requests
import os

# ==============================
# API endpoint (Backend)
# ==============================
API_URL = "http://127.0.0.1:8000/register"

# ==============================
# Step 1: Capture image from webcam
# ==============================
cam = cv2.VideoCapture(0)
print("📷 Press SPACE to capture your face...")

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Failed to capture camera frame.")
        break

    cv2.imshow("Register Face", frame)
    if cv2.waitKey(1) & 0xFF == 32:  # SPACE key
        break

cam.release()
cv2.destroyAllWindows()

# ==============================
# Step 2: Encode captured frame to base64
# ==============================
_, buffer = cv2.imencode('.jpg', frame)
image_b64 = base64.b64encode(buffer).decode("utf-8")
image_b64 = f"data:image/jpeg;base64,{image_b64}"

# ==============================
# Step 3: Save locally also
# ==============================
name = input("Enter your name for registration: ").strip()

if not os.path.exists("registered_faces_local"):
    os.makedirs("registered_faces_local")

save_path = os.path.join("registered_faces_local", f"{name}.jpg")
cv2.imwrite(save_path, frame)
print(f"💾 Saved local copy: {save_path}")

# ==============================
# Step 4: Send to backend
# ==============================
payload = {
    "name": name,
    "image_base64": image_b64
}

print("📤 Uploading to server...")
try:
    res = requests.post(API_URL, json=payload)
    print("✅ Server response:")
    print(res.json())
except Exception as e:
    print("❌ Error sending to server:", e)
