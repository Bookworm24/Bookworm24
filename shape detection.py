import cv2
import numpy as np
from  object import stack_all


"""
`cv2.GaussianBlur()`是OpenCV中用于对图像进行高斯模糊的函数。高斯模糊是一种常用的图像滤波方法，它可以减少图像中的噪声，并平滑图像的细节。

以下是使用`cv2.GaussianBlur()`函数对图像进行高斯模糊的示例代码：

```python
import cv2

# 读取图像
image = cv2.imread('image.jpg')

# 对图像进行高斯模糊
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 显示原始图像和模糊后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

在这个示例中，我们首先使用`cv2.imread()`函数读取名为'image.jpg'的图像。然后，我们使用`cv2.GaussianBlur()`函数对图像进行高斯模糊。该函数接受三个参数：输入图像、高斯核的大小和标准差。在这里，我们将高斯核的大小设置为(5, 5)，标准差设置为0。

最后，我们使用`cv2.imshow()`函数显示原始图像和模糊后的图像，并使用`cv2.waitKey(0)`等待用户按下键盘上的任意键。当用户按下键盘上的任意键时，窗口将被关闭。

您可以根据需要调整高斯核的大小和标准差，以实现不同程度的模糊效果。较大的高斯核将导致更强的模糊效果，而较小的高斯核将导致较轻的模糊效果。标准差控制高斯分布的形状，较大的标准差将导致更广泛的模糊效果，而较小的标准差将导致更局部的模糊效果。

cv.resize
cv2.resize()函数具有以下参数：

src：输入图像，可以是NumPy数组或图像文件的路径。
dsize：目标图像的大小，可以是元组 (width, height) 或整数值。如果是元组，则表示目标图像的宽度和高度；如果是整数值，则表示目标图像的缩放比例。例如，dsize=(500, 300) 表示目标图像的宽度为500像素，高度为300像素；dsize=0.5 表示将图像缩放为原始图像的一半大小。
fx：水平方向的缩放比例。如果同时指定了 fx 和 fy，则 dsize 参数将被忽略。
fy：垂直方向的缩放比例。如果同时指定了 fx 和 fy，则 dsize 参数将被忽略。
interpolation：插值方法，用于调整图像大小。可以是以下几种方法之一：
cv2.INTER_NEAREST：最近邻插值法。
cv2.INTER_LINEAR：双线性插值法（默认值）。
cv2.INTER_CUBIC：双三次插值法。
cv2.INTER_LANCZOS4：Lanczos插值法。
"""






def getcontours(img,imgconotours):#该函数用于轮廓检测主要用于检测图像中的轮廓信息以及各个轮廓之间的结构信息，并将检测结果通过值返回
    contours,hierachy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#contours是检测到的轮廓，每个轮廓中存放的坐标
    """cv2.findContours()函数返回两个值：轮廓列表和层次结构。我们将轮廓存储在contours变量中。

然后，我们创建一个与原始图像大小相同的空白图像作为绘制轮廓的画布。最后，我们使用cv2.drawContours()函数将轮廓绘制在画布上。

cv2.drawContours()函数的参数包括画布图像、轮廓列表、要绘制的轮廓索引（-1表示绘制所有轮廓）、颜色和线宽度。"""
    """cv2.contourArea()是OpenCV库中用于计算轮廓的面积的函数。轮廓是图像中连续的边界线，而轮廓的面积表示轮廓所包围的区域的大小。"""
    for cnt in contours:
        area=cv2.contourArea(cnt)#寻找面积
        areamin=cv2.getTrackbarPos('area','param')
        if areamin >1000:
            cv2.drawContours(imgconotours, contours, -1, (0, 0, 255), 7)
            """cv2.arcLength()是OpenCV库中用于计算轮廓的周长或弧长的函数然后，我们使用cv2.arcLength()函数计算第一个轮廓的周长或弧长。cv2.arcLength()函数的参数包括轮廓和一个布尔值，用于指定轮廓是否闭合。"""
            per=cv2.arcLength(cnt,True)
            #找到轮廓点的坐标cv2.approxPolyDP()是OpenCV库中的一个函数，用于对轮廓进行多边形逼近。它可以将复杂的轮廓近似为更简单的多边形，从而减少轮廓的点数。
            approx=cv2.approxPolyDP(cnt,0.02*per,True)
            #用于计算轮廓的边界矩形。边界矩形是一个矩形框，完全包围了给定轮廓。
            x_,y_,w,h=cv2.boundingRect(approx)
            cv2.rectangle(imgconotours,(x_,y_),(x_+w,y_+h),(0,255,0),5)
            cv2.putText(imgconotours,"points"+str(len(approx)),(x_+w+20,y_+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
            cv2.putText(imgconotours,"AREA"+str(int(area)),(x_+w+20,y_+45),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)


            #print(len(approx))#长度为4，代表只有4个点




def empty(x):
    pass
framwidth=1920/4
framheight=1080/4
cap=cv2.VideoCapture('../image/test2.mp4')
cap.set(3,framwidth)
cap.set(4,framheight)

# img=cv2.imread('../image/104.jpg')
# imgblur=cv2.GaussianBlur(img,(7,7),1)
# imggray=cv2.cvtColor(imgblur,cv2.COLOR_BGR2GRAY)
# imgstack=stack_img(0.8,[img,imgblur,imggray])
# cv2.namedWindow('q',cv2.WINDOW_NORMAL)
# cv2.imshow('d',img)
# cv2.imshow('s',imgblur)
# cv2.imshow('w',imggray)
# cv2.imshow('q',imgstack)
#接下来使用canny边缘检测
cv2.namedWindow('param')
cv2.resizeWindow('param',640,240)
cv2.createTrackbar("threshold1","param", 155, 255,empty )
cv2.createTrackbar("threshold2","param", 255, 255,empty )
cv2.createTrackbar("area","param", 5000,30000,empty )
stack=stack_all
while True:

    _,img=cap.read()
    imgconotours =img.copy()
    # cv2.imshow('orign',img)
    imgblur=cv2.GaussianBlur(img,(7,7),1)
    imggray=cv2.cvtColor(imgblur,cv2.COLOR_BGR2GRAY)

    # print([img,imgblur,imggray][0])
    threshold1=cv2.getTrackbarPos('threshold1','param')
    threshold2=cv2.getTrackbarPos('threshold2','param')
    imgcanny=cv2.Canny(imggray,threshold1,threshold2)
    # imgstack = stack_img(0.8, [img, imgblur, imgcanny])


    kernel=np.ones((5,5))
    #对图像进行膨胀操作，可以扩大图像中的亮区域或增加物体的大小。膨胀操作基于图像中的结构元素
    #iterations参数，用于指定膨胀操作的迭代次数。默认情况下，迭代次数为1。
    imgdel=cv2.dilate(imgcanny,kernel,iterations=1)
    # getcontours(imgdel,imgconotours)
    # imgstack =stack.stack_img(0.8,( [img, imggray, imgcanny],[imgdel,imgconotours,imgconotours]))
    imgstack =stack.stack_img(0.8,[img, imggray, imgcanny],)

    cv2.namedWindow('stack',cv2.WINDOW_NORMAL)

    cv2.imshow('stack',imgstack)


    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()