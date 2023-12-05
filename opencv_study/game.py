import random
import time

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
#game variables.the center of the circle
timestare=time.time()
cx,cy=250,250
counter=0
#circle color
color=(255,0,255)#一开始的颜色
score=0
totaltime=30
while True:
    success,img=cap.read()
    img=cv2.flip(img,1)#进行水平翻转
    if time.time()-timestare<totaltime:#满足
        # hands,img=dector.findHands(img)#这个能将所有的位置和骨架显示出来，如果不想
        hands=dector.findHands(img,draw=False)
        if hands:#判段有没有手
            try:
                lmlist=hands[0][0]['lmList']#手没在时候会报错
                bbox = hands[0][0]['bbox']  # 得到手的四个坐标的位置，或者
                x, y, w, h = hands[0][0]['bbox']




                #一共有21个数据是表示手的数据，那么我们具体要那几个
                x1,y1,_=lmlist[5]
                x2,y2,_=lmlist[17]
                #在我们使用旋转的时候发现x,的值会发生变化，但事实上不变的，你们我们就通过y不变来检测旋转
                distance=int(math.sqrt((y2-y1)**2+(x2-x1)**2))
                A,B,C=coff#返回的数据
                distancecm=A*distance**2+B*distance+C
                if distancecm<40:

                    #改变颜色,并且判断圆是否是在手掌之中
                    if x<cx<x+w and y<cy<y+h:
                        counter=1
                        print('手在中间，counter为1')
                #         color=(0,255,0)

                # else:
                #         color=(255,0,255)到这里，我们需要考虑到counter的影响

                #如果你要显示边框的话
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
                cvzone.putTextRect(img,f'int{distancecm}cm',(x+5,y-10))
            except:
                pass

        #draw the button
        if counter:#改变颜色，手在的话为真
            counter+=1#现在counter为3时
            color=(0,255,0)#把颜色color变为绿色
            print('color改变')
            if counter==3:
                #制作小圆点的随机出现，小圆点的圆心的x,y坐标
                cx=random.randint(100,1100)
                cy= random.randint(100, 600)
                color=(255,0,255)
                score+=1#统计分数
                counter=0

        cv2.circle(img,(cx,cy),30,color,cv2.FILLED)
        cv2.circle(img,(cx,cy),10,(255,255,255),cv2.FILLED)
        cv2.circle(img,(cx,cy),20,(255,255,255),2)
        cv2.circle(img,(cx,cy),30,(50,50,50),2)
        cvzone.putTextRect(img,f'time:{int(totaltime-(time.time()-timestare))}',(1000,75),scale=3,offset=20)
        cvzone.putTextRect(img,f'Score:{str(score).zfill(2)}',(60,75),scale=3,offset=20)#zfill2是实现保持2位小如01,02
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        cvzone.putTextRect(img, 'gameover', (400, 400), scale=5, offset=30,thickness=7)
        cvzone.putTextRect(img, f'score{score}', (425, 500), scale=3, offset=20, thickness=7)
        cvzone.putTextRect(img, f'press r to rese', (460, 575), scale=2, offset=10)
    cv2.imshow('image',img)
    if  cv2.waitKey(1) & 0xFF == ord('r'):
        #重新开始玩
        timestare=time.time()
        score=0

