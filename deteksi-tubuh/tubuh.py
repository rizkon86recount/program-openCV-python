import cv2

# Inisialisasi video capture dari kamera (biasanya 0 untuk kamera internal)
cap = cv2.VideoCapture(0)

# Load model deteksi tubuh (contoh: HOGDescriptor)
body_cascade = cv2.HOGDescriptor()
body_cascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    if not ret:
        break

    # Deteksi tubuh dalam frame
    bodies, _ = body_cascade.detectMultiScale(frame)

    # Gambar kotak di sekitar tubuh yang terdeteksi
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tampilkan frame yang telah diproses
    cv2.imshow('Body Detection', frame)

    # Hentikan pemutaran jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup video capture dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
