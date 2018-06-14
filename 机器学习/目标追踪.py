import cv2
import numpy as np
v = cv2.VideoCapture(r"./video/目标追踪玄凤“涨财”原视频.mp4")
w = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('./video/xuanfeng.avi',fourcc,24,(w,h))
n = 0
while True:
    try:
        if(n==1):
            pass
        img,frame = v.read()
        img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        img_gauss = cv2.GaussianBlur(img_gray,(15,15),0)
        if(n==0):
            bg = img_gauss
            n+=1
        diff = cv2.absdiff(bg,img_gauss)
        diff = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)[1]
        es = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
        diff = cv2.dilate(diff,es,iterations = 2)
        a,b,c = cv2.findContours(diff,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in b:
            if(cv2.contourArea(i)<4500):
                continue
            #将i转为矩形的坐标
            x,y,w,h = cv2.boundingRect(i)
            if(w>h):
                r = w//2+w//5
            else:
                r = h//2+h//5
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #2为粗细
            cv2.circle(frame,((2*x+w)//2,(2*y+h)//2),r,(0,0,255),2)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame,r'hi,i am comming',(x,y),font,1,(0,0,255),1,False)
        cv2.imshow("diff",diff)
        cv2.imshow("frame",frame)
        videoWriter.write(frame)
        cv2.waitKey(1)
    except Exception as err:
        print(err)
        pass
v.release()
videoWriter.release()
cv2.destroyAllWindows()
