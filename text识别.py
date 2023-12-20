import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time


# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('../image/21.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#转换为灰度图

#############################################
#### Detecting Characters  ######
#############################################
"""
boxes = pytesseract.image_to_boxes(img) 这行代码使用Pytesseract库对图像进行光学字符识别（OCR），并获取检测到的字符的边界框。

具体的工作流程如下：

pytesseract.image_to_boxes(img) 是Pytesseract库提供的一个函数。它接受一个图像（img）作为输入，并对其进行OCR处理。

OCR过程分析图像并识别其中的字符。

image_to_boxes() 函数返回一个字符串表示，其中包含检测到的字符及其对应的边界框。字符串中的每一行表示一个字符及其边界框的坐标。

通过将 pytesseract.image_to_boxes(img) 的结果赋值给变量 boxes，你可以在代码中访问和处理检测到的字符及其边界框。

例如，你可以遍历 boxes 字符串中的每一行，提取单个字符及其坐标。这对于文本提取、字符分割或文本标注等任务非常有用。

请注意，在使用 image_to_boxes() 函数之前，你需要确保已经安装了必要的依赖项，包括Tesseract OCR和Pytesseract库本身。
"""
pytesseract.pytesseract.tesseract_cmd =r"D:\Cache_py\Python\Python38\site-packages\pytesseract\terssocr\tesseract.exe"

hImg, wImg,_ = img.shape#宽和高
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():#遍历每个字符
    print(b)
    b = b.split(' ')
    print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])#化为整数
    # cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)#矩形坐标加数字
    cv2.putText(img,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)