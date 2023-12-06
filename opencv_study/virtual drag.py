import cvzone
import numpy as np
import math

import cv2
from cvzone.HandTrackingModule import HandDetector#检测手

#webcm
cap=cv2.VideoCapture(0)
cap.set(3,1200)
cap.set(propId=4,value=720)
dector=HandDetector(detectionCon=0.8,maxHands=1)
coloR=(255,0,255)
# cx,cy,w,h=100,100,200,200#动态更改矩形坐标的位置
class DragRec():

    def __init__(self,posicenter,size=[200,200]):
        self.position=posicenter
        self.size=size
    def update(self):
        cx,cy=self.position
        w,h=self.size
        global coloR
        #判断手指是否在其中
        if cx - w // 2 < fiinboxx[0] < cx + w // 2 and cy - h // 2 < fiinboxx[1] < cy + h // 2:

            self.position= fiinboxx[0], fiinboxx[1]
        elif cx - w // 2 < fiinboxx[0] + fiinboxx[2] < cx + w // 2 and cy - h // 2 < fiinboxx[1] < cy + h // 2:

            self.position= fiinboxx[0] + fiinboxx[2], fiinboxx[1]


rectlist=[]
for x in range(5):
    rectlist.append(DragRec([x*250+150,150]))

while True:
    success,img=cap.read()
    img=cv2.flip(img,1)#镜像的缘由
    # hands,img=dector.findHands(img)#这个能将所有的位置和骨架显示出来，如果不想
    hands=dector.findHands(img)

    if hands[0]:
        """这个和视频只中的不一样,应该是更新的版本不一样"""
        p1= hands[0][0]['lmList'][8][:2]
        p2 = hands[0][0]['lmList'][12][:2]

        dis,m,z=dector.findDistance(img=img,p1=p1,p2=p2)

        if dis<50:


            fiinboxx=hands[0][0]['bbox']
            for rect in rectlist:
                rect.update()
    print(len(rectlist))
    for rect in rectlist:
        cx,cy=rect.position
        w,h=rect.size

        cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),coloR,cv2.FILLED)
        """cvzone.cornerRect函数可以用来在图像上绘制有定角度的矩形框,它主要有以下用途:"""
        cvzone.cornerRect(img,(cx-w//2,cy-h//2,w,h),20,rt=0)

    cv2.imshow('image',img)
    if  cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
单个调试
while True:
    success,img=cap.read()
    img=cv2.flip(img,1)#镜像的缘由
    # hands,img=dector.findHands(img)#这个能将所有的位置和骨架显示出来，如果不想
    hands=dector.findHands(img)

    if hands[0]:
        '这个和视频只中的不一样,应该是更新的版本不一样'
        p1= hands[0][0]['lmList'][8][:2]
        p2 = hands[0][0]['lmList'][12][:2]
""""""
        dis,m,z=dector.findDistance(img=img,p1=p1,p2=p2)
        # print(dis)
        if dis<50:


            fiinboxx=hands[0][0]['bbox']
            if cx-w//2<fiinboxx[0]<cx+w//2 and cy-h//2<fiinboxx[1]<cy+h//2  :
                coloR=(0,255,0)
                cx,cy=fiinboxx[0],fiinboxx[1]
            elif cx-w//2<fiinboxx[0]+fiinboxx[2]<cx+w//2 and cy-h//2<fiinboxx[1]<cy+h//2:
                coloR = (0, 255, 0)
                cx, cy = fiinboxx[0]+fiinboxx[2], fiinboxx[1]

            else:
                coloR=(255,0,255)


    cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),coloR,cv2.FILLED)

    cv2.imshow('image',img)
    if  cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""