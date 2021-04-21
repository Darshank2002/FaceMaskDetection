
from keras.models import load_model

import cv2,os,urllib.request
import numpy as np
from django.conf import settings

face_clsfr= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



model = load_model('model-006.model')
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):


        self.video.release()

    def get_frame(self):
      while(True):
  
        ret,img = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        faces=face_clsfr.detectMultiScale(gray,1.3,5)  
        for (x,y,w,h) in faces:
           face_img=gray[y:y+w,x:x+w]
           resized=cv2.resize(face_img,(100,100))
           normalized=resized/255.0
           reshaped=np.reshape(normalized,(1,100,100,1))
           result=model.predict(reshaped)
           print(result)

           label=np.argmax(result,axis=1)[0]
          
           cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
           cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
           cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
    





        
