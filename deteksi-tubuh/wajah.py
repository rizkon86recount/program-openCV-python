import cv2
import mediapipe as mp

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # Nomor 0 mengacu pada kamera default

# Inisialisasi mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()  # Membaca frame dari kamera

        # Konversi frame ke format yang diperlukan oleh mediapipe (BGR ke RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deteksi wajah dalam frame
        results = face_detection.process(frame_rgb)

        # Jika wajah terdeteksi, gambar garis wajah dengan warna
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Tampilkan frame dari kamera
        cv2.imshow("Wajah", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Tutup kamera dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
