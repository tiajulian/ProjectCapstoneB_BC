import time
import cv2 as cv
import PoseModule as pm


cap = cv.VideoCapture('Data_Collection/bicep_curls_1.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = cv.resize(img, (680, 680), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)  # resizing original video
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) !=0:
        print(lmList[14])
        # cv.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.imshow('Image', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
