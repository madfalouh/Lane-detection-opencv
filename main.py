import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny=cv2.Canny(blur , 50 , 150)
    return  canny

def region(image) :
    height = image.shape[0]
    triangle=np.array([[(400 , height) , (1000 , height )  , (800 , 500)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask , triangle , 255)
    masked = cv2.bitwise_and(image , mask)
    return  masked

def display(image , lines) :
   line_image=np.zeros_like(image)
   if lines is not None :
       for line in lines :
           x1,y1,x2,y2=line.reshape(4)
           cv2.line(line_image , (x1,y1) , (x2,y2) , (255,0,0) , 10)

           return line_image




cap = cv2.VideoCapture("videoplayback.mp4")




while  (cap.isOpened()) :
     _, frame= cap.read()
     anny = canny(frame)
     crim = region(anny)
     lines = cv2.HoughLinesP(crim, 2, np.pi / 180, 80, np.array([]), minLineLength=40, maxLineGap=5)
     lineimg = display(frame, lines)
     combo = cv2.addWeighted(frame, 0.8, lineimg, 1, 1)
     cv2.imshow("res",combo)
     cv2.waitKey(1)
     #plt.imshow(combo)
     #plt.show()


