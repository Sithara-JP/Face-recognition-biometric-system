import cv2
import numpy as np
import face_recognition
import os

previmg = []
newPrev = "xy"
iterCount = 0
path = "D:/FaceRecognition"
imgs = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    imgs.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def fndEncod(imgs):
    encodeList=[]
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodelistKnown = fndEncod(imgs)
print('Encoding Done')
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    iterCount += 1
    flag = 0
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFr = face_recognition.face_locations(imgS)
    encodesCurrFr = face_recognition.face_encodings(imgS, facesCurrFr)

    for encodeFace, faceLoc in zip(encodesCurrFr, facesCurrFr):
        m = face_recognition.compare_faces(encodelistKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)
        matchInd = np.argmin(faceDis)
        flag = 0

        if m[matchInd]:
            name = classNames[matchInd].upper()
            if iterCount > 1:
                previmg.append(name)
            else:
                previmg.append("Done")
            newPrev = str(previmg[-1])
            if newPrev == name:
                flag = 1
            if flag == 0:
                print(name)
            else:
                name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = 4*y1, 5*x2, 5*y2, 4*x1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('webcam', img)
    cv2.waitKey(1)