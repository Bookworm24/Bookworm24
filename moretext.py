import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time


pytesseract.pytesseract.tesseract_cmd =r"D:\Cache_py\Python\Python38\site-packages\pytesseract\terssocr\tesseract.exe"
img = cv2.imread('../image/text.PNG')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##############################################
##### Image to String   ######
##############################################
# print(pytesseract.image_to_string(img))pytesseract.image_to_string()函数从图像中提取中文文本，您需要确保已经正确配置了Tesseract OCR引擎，并且安装了支持中文识别的语言数据文件。以下是一个示例代码：
# 提取文本

#############################################
#### Detecting Characters  ######
#############################################
# hImg, wImg,_ = img.shape
# boxes = pytesseract.image_to_boxes(img)#觉中是一种对象检测和定位的基本概念。它通过一个最小的矩形框来表示对象的位置和范围。
# for b in boxes.splitlines():#将包含一个字符串,其每个行表示一个字符的边界框信息
#
#     b = b.split(' ')
#     print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])#定位上下左右相减
#     cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)
#     cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)


##############################################
##### Detecting Words  ######
##############################################
# #[   0          1           2           3           4          5         6       7       8        9        10       11 ]
# #['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text']
# boxes = pytesseract.image_to_data(img)
# for a,b in enumerate(boxes.splitlines()):
#         print(b)
#         print(a,'a')#a为个数
#         if a!=0:
#
#             b = b.split()
#             if len(b)==12:
#                 x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
#                 cv2.putText(img,b[11],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
#                 cv2.rectangle(img, (x,y), (x+w, y+h), (50, 50, 255), 2)


##############################################
##### Detecting ONLY Digits  ######
##############################################
# hImg, wImg,_ = img.shape
# conf = r'--oem 3 --psm 6 outputbase digits'
"""
让我总结一下conf = '--oem 3 --psm 6'这两个参数的含义:

--oem参数设置OCR引擎模式,这里的值3对应默认的LSTM神经网络引擎,这是一种深度学习技术,识别效果较好。

--psm参数设定图像结构解析模式,值6表示假设图像包含单个文字块,会根据此假设进行文字识别。
"""
# boxes = pytesseract.image_to_boxes(img,config=conf)
# print(boxes)
# for b in boxes.splitlines():
#
#
#     b = b.split(' ')
#
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)
#     cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)



##############################################
##### Webcam and Screen Capture Example ######
##############################################
# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
def captureScreen(bbox=(300,300,1500,1000)):
    capScr = np.array(ImageGrab.grab(bbox))#您提供的代码片段使用PIL(Python图像库)中的ImageGrab模块来截取屏幕截图。bbox参数指定要捕获的区域的边界框坐标。
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr

timer = cv2.getTickCount()#计算运行的时间
# _,img = cap.read()
img = captureScreen()#计算截屏时间
#DETECTING CHARACTERES
hImg, wImg,_ = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    #print(b)
    b = b.split(' ')
    #print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)
    cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);#计算时间
cv2.putText(img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,230,20), 2);
cv2.imshow("Result",img)
cv2.waitKey(1)
#
#

cv2.imshow('img', img)

cv2.waitKey(0)