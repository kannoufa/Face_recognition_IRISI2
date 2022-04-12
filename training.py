import cv2
import os
import numpy as np
from PIL import Image


#vérification de l'existence du dossier à partie de son chemin
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# LBPH = Local Binary Patterns Histograms
recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier("face_detection.xml")

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        print("traitement de : " + imagePath)
        PIL_img = Image.open(imagePath).convert('L')
        
        img_numpy = np.array(PIL_img,'uint8')
        
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            ids.append(id)

    return faceSamples,ids


faces,ids = getImagesAndLabels('dataset')

# Entrainement du modèle
print("Training ...\nWAIT !")
recognizer.train(faces, np.array(ids))

# Sauvegarder le modèle dans un fichier yaml
assure_path_exists('saved_model/')
recognizer.write('saved_model/s_model.yml')
