import cv2
import mediapipe as mp
from math import hypot
from PIL import Image
import time
import os

path = os.getcwd()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils

low_threshold_iris_lenght = 11
high_threshold_iris_lenght = 13
blink_count = 0
image_count = 0
image_list = []
max_saved_images = 5


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(imgRGB)
        face_list = []

        height, width = frame.shape[:2]
        upper_left = (width // 3, height // 4)
        bottom_right = (width * 3 // 5, height * 3 // 4)
        standard_rect = cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), thickness=1)

        for x,y,w,h in faces:
            face_rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
            if results.multi_face_landmarks:
                for handlandmark in results.multi_face_landmarks:
                    for id,lm in enumerate(handlandmark.landmark):
                        h,w,_ = frame.shape
                        cx,cy = int(lm.x*w),int(lm.y*h)
                        face_list.append([id,cx,cy])
                    mpDraw.draw_landmarks(frame,handlandmark,mp_face_mesh.FACEMESH_CONTOURS, mpDraw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

            iou = bb_intersection_over_union([x,y,w,h], [width // 3, height // 4 ,width * 3 // 5, height * 3 // 4])
            if iou >= 0.22: capture_flag = True
            else:
              cv2.putText(frame, 'PLACE YOURSELF', (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
              cv2.putText(frame, 'INTO THE RECTANGLE', (400, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
              capture_flag = False

        if face_list != []:
            x1_left_iris, y1_left_iris = face_list[474][1], face_list[474][2]
            x2_left_iris, y2_left_iris = face_list[476][1], face_list[476][2]
            x1_right_iris, y1_right_iris = face_list[469][1], face_list[469][2]
            x2_right_iris, y2_right_iris = face_list[471][1], face_list[471][2]  
            left_iris_length = hypot(x1_left_iris-x2_left_iris, y1_left_iris-y2_left_iris)
            right_iris_length = hypot(x1_right_iris-x2_right_iris, y1_right_iris-y2_right_iris)  

            x1_left_eye, y1_left_eye = face_list[159][1], face_list[159][2]
            x2_left_eye, y2_left_eye = face_list[145][1], face_list[145][2]
            x1_right_eye, y1_right_eye = face_list[386][1], face_list[386][2]
            x2_right_eye, y2_right_eye = face_list[374][1], face_list[374][2]
            left_eye_length = hypot(x1_left_eye-x2_left_eye, y1_left_eye-y2_left_eye)
            right_eye_length = hypot(x1_right_eye-x2_right_eye, y1_right_eye-y2_right_eye)

            if right_iris_length < low_threshold_iris_lenght: 
              cv2.putText(frame, 'COME CLOSER', (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
              blink_count = 0
            elif right_iris_length > high_threshold_iris_lenght: 
              cv2.putText(frame, 'GO BACK', (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
              blink_count = 0
            else:
                cv2.putText(frame, 'JUST RIGHT', (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                if (left_eye_length <= 7.00 or left_eye_length <= 7.00) and capture_flag:
                    blink_count +=1
                    cv2.putText(frame, 'BLINK', (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if blink_count == 2:
                    cv2.putText(frame, 'ACCEPT', (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    image_count += 1
                    image_list.append(frame)
                    new_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    new_image = Image.fromarray(new_image)
                    new_image = new_image.crop((width // 3, height // 4 ,width * 3 // 5, height * 3 // 4))
                    new_image = new_image.resize((64, 64))
                    time.sleep(0.001)
                    new_image.save(path +"\\saved_images\\" +  str(image_count) + '.png')
                    print(len(image_list))
                    blink_count = 0

                    if len(image_list) == max_saved_images:
                      break
                
        cv2.imshow('Image', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()