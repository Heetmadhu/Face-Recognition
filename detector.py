import cv2
import sqlite3
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml');
cam = cv2.VideoCapture(0);

rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainningData.yml")


def getProfile(id):
    con = sqlite3.connect("Faces.db")
    cmd = "SELECT * FROM People WHERE ID ="+str(id)
    cursor = con.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
        
    con.close()
    return profile


id=0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret,img = cam.read();
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,xrtwer = rec.predict(gray[y:y+h,x:x+w])
        
        profile = getProfile(id)
        if(profile != None):
            cv2.putText(img,str(profile[1]),(x,y+h),font,1,(0,0,255),2);
            cv2.putText(img,str(profile[2]),(x,y+h+30),font,1,(0,0,255),2);
            cv2.putText(img,str(profile[3]),(x,y+h+60),font,1,(0,0,255),2);
            cv2.putText(img,str(profile[4]),(x,y+h+90),font,1,(0,0,255),2);
            cv2.putText(img,str(xrtwer),(x,y+h+120),font,1,(0,0,255),2);
    
    cv2.imshow("face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()
    
