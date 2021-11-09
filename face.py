import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('/Users/sidhu/cprograms/programspy/cascades/data/haarcascade_frontalface_alt2.xml')

img = cv2.imread('bo2.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,scaleFactor=1.005,minNeighbors=5)
print(faces)
for(x,y,w,h)in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(268,256,0),1)
resized=cv2.resize(img,(960,540))

cv2.imshow('face recognition',resized)
cv2.waitKey(0)
cv2.destroyAllWindowa()