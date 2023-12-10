import cv2
import math

path = '1.png'
img = cv2.imread(path)
pointsList = []

def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:#鼠标事件
        size = len(pointsList)
        """
        检查pointsList列表的长度，以确定是否已经存在足够的点来绘制线
        """
        if size != 0 and size % 3 != 0:#三给角构成
            cv2.line(img,tuple(pointsList[round((size-1)/3)*3]),(x,y),(0,0,255),2)
        cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])


"""
您提供的代码定义了一个名为gradient的函数，用于计算两点之间的斜率（梯度）。该函数接受两个点pt1和pt2作为输入，并返回斜率的值。
"""
def distance(pt1,pt2):
    # print((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    # return (pt2[1]-pt1[1]),(pt2[0]-pt1[0])

def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    # print(pt1,pt2,pt3)
    # print(pt1,pt2,pt3)
    """ 直接操作坐标"""
    a = distance(pt2, pt3)
    b = distance(pt1, pt3)
    c = distance(pt1, pt2)
    A = round(math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))))

    print(A)

    # m1 = gradient(pt1,pt2)#p1是圆点然后一个点和另一个点
    # m2 = gradient(pt1,pt3)

    # angR = math.atan((m2-m1)/(1+(m2*m1)))#反正切角度查找
    # # angD = abs(round(math.degrees(angR)))
    # angD = round(math.degrees(angR))
    # print(m2)
    # print(m1)


    cv2.putText(img,str(A),(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX,
                    1.5,(0,0,255),2)


while True:
    if len(pointsList) % 3 == 0 and len(pointsList) !=0:
        getAngle(pointsList)


    cv2.imshow('Image',img)
    cv2.setMouseCallback('Image',mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList = []
        img = cv2.imread(path)
        #重新清除返回