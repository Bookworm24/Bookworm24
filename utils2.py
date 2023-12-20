import cv2
import numpy as np

def getContours(img,cThr=[100,100],showCanny=False,minArea=1000,filter=3,draw=True):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#让img装换为gray是为了运用高斯模糊
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)#用（5,5）作为卷积
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])#用canny算法，边缘检测
    kernel = np.ones((5,5))#用卷积
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)#用腐蚀
    imgThre = cv2.erode(imgDial,kernel,iterations=2)#用扩充来实现对轮廓的细化和粗化

    if showCanny:cv2.imshow('Canny',imgThre)#判断是否允许以实现允许的切换
    contours,hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#找到轮廓的像素坐标，逼近轮廓和关系
    finalCountours = []#实现对轮廓坐标的存储和找到最大坐标，即存储到银行卡的坐标
    #len contours为50.在第一时候有50个在contour里面。
    # 第二次只有5个,因为选了面积最大的,两张卡在面积最大的里面
    x=1
    for i in contours:#第一次找到的为50个，第二次为有5个则有轮廓，还有层级关系
        x+=1
        print(x)

        area = cv2.contourArea(i)#计算每个轮廓的面积，以便找到银行卡的面积

        y=0
        if area > minArea:#定义最小面积，用来排除面积
            y+=1#在所有轮廓中，第一次有一个面积大于1000的即最大的那个
            # 在第二次中有两个，即两个银卡
            print(y)
            peri = cv2.arcLength(i,True)#曲线中相邻的两个像素的之间的连线的长度
            approx = cv2.approxPolyDP(i,0.02*peri,True)#根据输入的轮廓得到的最佳的逼近多边形
            # print(len(approx))#每个逼近有4个点，为矩形
            bbox = cv2.boundingRect(approx)#用矩形逼近
            if filter > 0:

                if len(approx) == filter:#这个是查看是否为正方形，三角形等,不是就不会执行了
                    finalCountours.append([len(approx),area,approx,bbox,i])#
                    print('fdgewrfwe')
            else:

                finalCountours.append([len(approx),area,approx,bbox,i])
                print()

    finalCountours = sorted(finalCountours,key = lambda x:x[1] ,reverse= True)#这里只找到了一个轮廓是最大轮廓，在银行卡之间的最大矩形
    # print(len(finalCountours))#长度为1，排序的话是为了找到面积最大值
    if draw:
        for con in finalCountours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)

    return img, finalCountours

def reorder(myPoints):
    # print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)#和这个行列都相同零矩阵
    myPoints = myPoints.reshape((4,2))#将其shape转化为4行2列
    add = myPoints.sum(1)#将其以行相加() 函数被用于沿着第二个轴（axis=1）计算 mypoint 中每行的和，结果存储在变量 add 中。


    # print(add)
    myPointsNew[0] = myPoints[np.argmin(add)]#这行代码的作用是将 mypoint 中和 add 最小值对应的行赋值给 mypointnew 的第一行。
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)#np.diff 是 NumPy 库中的一个函数，用于计算数组中相邻元素之间的差值。
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print(myPointsNew)
    return myPointsNew

def warpImg (img,points,w,h,pad=20):
    # print(points)
    points =reorder(points)#面积最大的轮廓点
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

