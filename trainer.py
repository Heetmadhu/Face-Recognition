import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create();
path = 'dataSet'

def getImagesID(path):
    imgPaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imgPath in imgPaths:
        faceImg = Image.open(imgPath).convert('L');
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imgPath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs),faces

Ids,faces = getImagesID(path)
recognizer.train(faces,Ids)
recognizer.write('recognizer/trainningData.yml')
cv2.destroyAllWindows()
