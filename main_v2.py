import cv2 
import mediapipe as mp

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        height, width, _ = frame.shape
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(rgb_image, 1.1, 4)
        result = face_mesh.process(rgb_image)

       # for x,y,w,h in faces:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(frame, (x, y), 1, (0, 0, 0), -1)


        cv2.imshow("Image", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if not ret:
            continue

cap.release()
cv2.destroyAllWindows()