import cv2
import sqlite3
import numpy as np

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml");
cam = cv2.VideoCapture(0);

def insertOrUpdate(Name):
    con=sqlite3.connect("Faces.db")
    cmd = "SELECT ID FROM People WHERE Name='"+str(Name)+"'"
    cursor = con.execute(cmd)
    isRecordExist = 0
    id=0
    for row in cursor:
        id = row[0]
        isRecordExist = 1
    if(isRecordExist == 0):
        cmd = "INSERT INTO People(Name) VALUES('"+str(Name)+"')"
        con.execute(cmd)
        con.commit()
        cmd = "SELECT ID FROM People WHERE Name='"+str(Name)+"'"
        cursor = con.execute(cmd)
        for row in cursor:
            id = row[0]
            

    
    
    con.commit()
    con.close()
    return id;

name = raw_input("enter user name:")
id = insertOrUpdate(name)
sampleNum = 0;
while(True):
    ret,img=cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum = sampleNum+1;
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100);
    img = cv2.flip(img,1)
    cv2.imshow("face",img);
    cv2.waitKey(1);
    if(sampleNum > 50):
       break;

cam.release()
cv2.destroyAllWindows()
