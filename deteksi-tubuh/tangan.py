import cv2
import mediapipe as mp

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # Nomor 0 mengacu pada kamera default

# Inisialisasi mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()  # Membaca frame dari kamera

    # Konversi frame ke format yang diperlukan oleh mediapipe (BGR ke RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan dalam frame
    results = hands.process(frame_rgb)

    # Jika tangan terdeteksi, gambar struktur tangan dengan warna
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(
                                          color=(0, 0, 255), thickness=2, circle_radius=4),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    # Tampilkan frame dari kamera
    cv2.imshow("Tangan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
