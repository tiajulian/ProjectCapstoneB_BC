import time
import cv2 as cv
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

pose = mpPose.Pose()

cap = cv.VideoCapture('Data_Collection/tia_bicep.mp4')
pTime = 0

while True:
    success, img = cap.read()
    img = cv.resize(img, (680, 680), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)  # resizing original video
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(img, (cx, cy), 7, (255, 0, 0), cv.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.imshow('Image',img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
