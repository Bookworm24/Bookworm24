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
#用拟合函数来寻找距离最为适合的点，通过x和x的平方和y的平方
x=[300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff=np.polyfit(x,y,2)#相当于y=Ax2+bx+c


while True:
    success,img=cap.read()
    # hands,img=dector.findHands(img)#这个能将所有的位置和骨架显示出来，如果不想
    hands=dector.findHands(img,draw=False)
    if hands:#判段有没有手
        print(hands[0][0])
        lmlist=hands[0][0]['lmList']#更换一下，会报错



        """得到数据
        [[431, 406, 0], [502, 401, -23], [564, 353, -25], [600, 305, -24], [630, 272, -22], [542, 260, 1], [578, 213, -4], [600, 182, -14], [617, 155, -22], [503, 234, 4], [528, 167, 3], [544, 129, -2], [557, 100, -7], [461, 224, 3], [479, 159, -2], [488, 125, -9], [496, 96, -14], [417, 230, -1], [418, 177, -9], [418, 148, -12], [420, 124, -13]]

        """
        # bbox=hands[0]['bbox']#得到手的四个坐标的位置，或者
        # x,y,w,h=hands[0]['bbox']
        # #一共有21个数据是表示手的数据，那么我们具体要那几个
        # x1,y1,_=lmlist[5]
        # x2,y2,_=lmlist[17]
        # #在我们使用旋转的时候发现x,的值会发生变化，但事实上不变的，你们我们就通过y不变来检测旋转
        # distance=int(math.sqrt((y2-y1)**2-(x2-x1)**2))
        # A,B,C=coff#返回的数据
        # distancecm=A*distance**2+B*distance+C
        #
        # #如果你要显示边框的话
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
        # cvzone.putTextRect(img,f'int{distancecm}cm',(x+5,y-10))
        # print(abs(x2-x1))
    cv2.imshow('image',img)
    if  cv2.waitKey(1) & 0xFF == ord('q'):
        break

