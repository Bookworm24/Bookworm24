import cv2
from object import utils2 as utlis

###################################
webcam = True
path = '../image/object.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale = 3
wP = 210 *scale
hP= 297 *scale
###################################


    # if webcam:success,img = cap.read()
    # else:
img = cv2.imread(path)

imgContours , conts = utlis.getContours(img,minArea=50000,filter=4)
if len(conts) != 0:#长度为1
    biggest = conts[0][2]#因为安装排序后为面积最大
    #print(biggest)
    imgWarp = utlis.warpImg(img, biggest, wP,hP)
    imgContours2, conts2 = utlis.getContours(imgWarp,#两次来实现先从银行卡的白纸中再用银行卡
                                             minArea=2000, filter=4,
                                             cThr=[50,50],draw =True)

    if len(conts) != 0:
        for obj in conts2:
            cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)#用这个进行画画
            nPoints = utlis.reorder(obj[2])
            nW = round((utlis.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
            nH = round((utlis.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
            cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = obj[3]
            cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.imshow('Original', imgContours2)
    # if len(conts) != 0:
    #     for obj in conts:
    #         cv2.polylines(imgContours,[obj[2]],True,(0,255,0),2)
    #         nPoints = utlis.reorder(obj[2])
    #         nW = round((utlis.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
    #         nH = round((utlis.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
    #         cv2.arrowedLine(imgContours, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
    #                         (255, 0, 255), 3, 8, 0, 0.05)
    #         cv2.arrowedLine(imgContours, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
    #                         (255, 0, 255), 3, 8, 0, 0.05)
    #         x, y, w, h = obj[3]
    #         cv2.putText(imgContours, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
    #                     (255, 0, 255), 2)
    #         cv2.putText(imgContours, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
    #                     (255, 0, 255), 2)
    # cv2.imshow('A4', imgContours2)

img = cv2.resize(img,(0,0),None,0.5,0.5)

cv2.waitKey(0)
cv2.destroyAllWindows()