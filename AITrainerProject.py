import cv2 as cv
import time
import numpy as np
import PoseModule as pm

#cap = cv.VideoCapture("Data_Collection/tia_bicep.mp4")
#cap = cv.VideoCapture("Data_Collection/bicepcurlA.MP4")#half rep
#cap = cv.VideoCapture("Data_Collection/bicepcurlB.MP4")#video moving inconsistent angle
#cap = cv.VideoCapture("Data_Collection/bicepcurlC.MP4") #dont count half reps
#cap = cv.VideoCapture("Data_Collection/bicepcurlD.MP4")#error
#cap = cv.VideoCapture("Data_Collection/bicepcurlE.MP4")
cap = cv.VideoCapture("Data_Collection/bicepcurlF.MP4")#inconsistence
#cap = cv.VideoCapture("Data_Collection/bicepcurlH.MP4")
#cap = cv.VideoCapture("Data_Collection/bicepcurlI.MP4")#too far
#cap = cv.VideoCapture("Data_Collection/bicepcurlJ.MP4") # bad angle
#cap = cv.VideoCapture("Data_Collection/bicepcurlK.MP4")
 #cap = cv.VideoCapture("Data_Collection/bicepcurlM.MP4")
#cap = cv.VideoCapture("Data_Collection/bicepcurlN.MP4")
#cap = cv.VideoCapture("Data_Collection/bicepcurl2.MP4")
detector = pm.poseDetector()
count = 0
dir = 0  # direction


pTime = 0

while True:
    success, img = cap.read()
    #img = cv.imread("Data_Collection/body.jpg") # read image
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        righta=detector.findAngle(img, 12, 14, 16)
        lefta= detector.findAngle(img, 11, 13, 17)
        #angle = detector.findAngle(img, 11, 13, 15)
        if lefta >180:
            per = np.interp(righta, (210, 300), (0, 100))
        elif lefta <150:
            per=np.interp(righta, (60, 140), (0, 100))
        bar = np.interp(righta, (220,310), (650,100))
        print(righta, per)

        # check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)


        # Draw Curl Count
        cv.rectangle(img,(0,450), (250,720), (0,255,0),cv.FILLED)
        cv.putText(img, str(int(count)), (45, 670), cv.FONT_HERSHEY_PLAIN, 15, (255, 0, 0, 5), 25)
        cv.putText(img, str(int(per)), (50, 100), cv.FONT_HERSHEY_PLAIN,3, (255, 0, 0, 5),5)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (50, 100), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0, 5), 5)

    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
