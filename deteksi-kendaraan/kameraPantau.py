import cv2

# Inisialisasi video capture dari kamera (biasanya 0 untuk kamera internal)
cap = cv2.VideoCapture(0)

# Load pre-trained car detection model (contoh: Haar Cascade Classifier)
car_cascade = cv2.CascadeClassifier('cars.xml')  # Pastikan path file sesuai

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    if not ret:
        break

    # Konversi frame menjadi grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi mobil dalam frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Gambar kotak di sekitar mobil yang terdeteksi
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Tampilkan frame yang telah diproses
    cv2.imshow('Traffic Detection', frame)

    # Hentikan pemutaran jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup video capture dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
