import cv2
pathmodel = "haarcascade_frontalface_alt2.xml"
face = cv2.CascadeClassifier(pathmodel)

#读取图片并进行灰度处理
imgPath = "./img/pexels-photo-235462.jpeg"
img = cv2.imread(imgPath)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("W",img_gray)

#检测人脸
rst = face.detectMultiScale(
    img_gray,
    scaleFactor = 1.02,
    minNeighbors = 3,
    minSize = (50,50),
    flags = cv2.CASCADE_SCALE_IMAGE
    )

#标注人脸位置
for x,y,w,h in rst:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
cv2.imshow("F",img)
cv2.imwrite("./Result img/faceTest.jpg",img)
