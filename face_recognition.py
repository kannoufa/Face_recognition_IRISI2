import cv2
import numpy as np
import os 
import writer

camera_port = 0
       
etudiants = {
    1 : "Errouk",
    2 : "Mouzafir",
    3 : "Bentouhami",
    4 : "Tamega",
    5 : "Firoud",
    6 : "Oubenaddi",
    7 : "Aminatou",
    8 : "Marouni",
    9 : "Aguenchich",
    10 : "Oussahi",
    11 : "Kannoufa",
    12 : "Outhouna",
    13 : "Ougoud",
}

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('saved_model/s_model.yml')

faceCascade = cv2.CascadeClassifier("face_detection.xml")

cam = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)

while True:
    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)

        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if((100 - confidence)>10):
            label = etudiants[Id] + " {0:.2f}%".format(round(100 - confidence, 2))
            
            cv2.rectangle(img, (x-20,y-85), (x+w+20, y-20), (0,255,0), -1)
            cv2.putText(img, label, (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            writer.output('Presence_IRISI2_', 'IRISI 2', Id, etudiants[Id], 'yes')
        else :
            cv2.putText(img,"non reconnu" , (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
    cv2.imshow('face recognition', img) 

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()


