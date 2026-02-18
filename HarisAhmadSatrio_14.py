import cv2
import mediapipe as mp

mpose = mp.solutions.pose
pose = mpose.Pose()
mdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # Konversi BGR ke RGB
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Proses deteksi pose
    hasil = pose.process(imgrgb)

    if hasil.pose_landmarks:
        print("Terdeteksi")

        mdraw.draw_landmarks(
            img,
            hasil.pose_landmarks,
            mpose.POSE_CONNECTIONS
        )

        for id, lm in enumerate(hasil.pose_landmarks.landmark):
            print(id, lm.x, lm.y)

    else:
        print("Tidak terdeteksi")

    cv2.imshow("Webcam", img)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam
cap.release()
cv2.destroyAllWindows()
