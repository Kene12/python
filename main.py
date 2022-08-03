import cv2
from datetime import datetime
now = datetime.today()
from cvzone.FaceDetectionModule import FaceDetector
detector = cv2.FaceDetector()


def draw(img,classifier,scaleFactor,minNieghbors,color,text,date):
         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
         features=classifier.detectMultiScale(gray,scaleFactor,minNieghbors)
         coords=[]
         for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,text,(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
                coords= [x,y,w,h]
         cv2.putText(img,date,(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.9,( 255, 255, 255 ),3)
         
         return  img,coords
def detect(img,face_cascade):
          currentTime = datetime.now()
          timestampMessage = currentTime.strftime("%Y/%m/%d   %H: %M : %S")
          img,coords=draw(img,face_cascade,1.1,10,(0,0,250),"Face",timestampMessage)
          return img
cap = cv2.VideoCapture(0)
while (True):
        ret,frame = cap.read()
        frame = detect(frame,detector)
        cv2.imshow('frame',frame)
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
        elif (cv2.waitKey(1) & 0xFF == ord('s')):
            cv2.imwrite('face.jpg',frame)
cap.release()
cv2.destroyAllWindows()
