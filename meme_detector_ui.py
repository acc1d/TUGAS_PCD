import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math

# ================= MediaPipe =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

ORIGINAL_PATH = "dataset/source/source.jpg"

# ================= LANDMARK =================
def get_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    h, w, _ = img.shape

    # PILIH WAJAH TERBESAR (UTAMA)
    max_area = 0
    best_face = None

    for face in result.multi_face_landmarks:
        xs = [lm.x for lm in face.landmark]
        ys = [lm.y for lm in face.landmark]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))

        if area > max_area:
            max_area = area
            best_face = face

    return np.array([[lm.x * w, lm.y * h] for lm in best_face.landmark])

def normalize(points):
    points = points - np.mean(points, axis=0)
    norm = np.linalg.norm(points)
    return points if norm == 0 else points / norm

def identity_distance(lm1, lm2):
    lm1_n = normalize(lm1)
    lm2_n = normalize(lm2)
    return np.mean(np.linalg.norm(lm1_n - lm2_n, axis=1))

# ================= CONFIDENCE =================
def confidence_score(distance):
    return 1 / (1 + math.exp(-0.25 * (distance - 12))) * 100

def level(conf):
    if conf < 40:
        return "RENDAH"
    elif conf < 70:
        return "SEDANG"
    else:
        return "TINGGI"

# ================= UI LOGIC =================
def browse_image():
    global meme_path
    meme_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png")]
    )
    if meme_path:
        img = Image.open(meme_path).resize((230, 230))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

def detect():
    meme = cv2.imread(meme_path)
    original = cv2.imread(ORIGINAL_PATH)

    if meme is None or original is None:
        result_label.config(text="Gambar tidak valid", fg="black")
        return

    lm_meme = get_landmarks(meme)
    lm_orig = get_landmarks(original)

    if lm_meme is None or lm_orig is None:
        result_label.config(text="Wajah tidak terdeteksi", fg="black")
        return

    # ===== STEP 1: CEK IDENTITAS =====
    id_dist = identity_distance(lm_meme, lm_orig)

    if id_dist > 0.22:
        result_label.config(
            text="BUKAN MEME\n(Subjek Berbeda)",
            fg="blue"
        )
        return

    # ===== STEP 2: CEK MANIPULASI =====
    diff = np.mean(np.linalg.norm(lm_meme - lm_orig, axis=1))
    conf = confidence_score(diff)
    lvl = level(conf)

    if conf >= 50:
        result_label.config(
            text=f"TERDETEKSI MEME\n"
                 f"Confidence: {conf:.2f}%\n"
                 f"Tingkat Perbedaan: {lvl}",
            fg="red"
        )
    else:
        result_label.config(
            text=f"BUKAN MEME\n"
                 f"Confidence: {conf:.2f}%\n"
                 f"Tingkat Perbedaan: {lvl}",
            fg="green"
        )

# ================= UI =================
root = tk.Tk()
root.title("Meme Face Detection System")
root.geometry("560x520")

tk.Label(root, text="Meme Face Detection System",
         font=("Arial", 16, "bold")).pack(pady=10)

tk.Button(root, text="Browse Meme Image",
          command=browse_image).pack(pady=5)

img_label = tk.Label(root)
img_label.pack(pady=10)

tk.Button(root, text="Start Detection",
          bg="green", fg="white",
          font=("Arial", 11, "bold"),
          command=detect).pack(pady=10)

result_label = tk.Label(root, text="",
                        font=("Arial", 13, "bold"),
                        justify="center")
result_label.pack(pady=20)

root.mainloop()
