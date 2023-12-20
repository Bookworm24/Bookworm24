import cv2 as cv
import numpy as np
"""
四个点158,59
106，134
163,176
218，99
"""
img=cv.imread('../image/prespective.png')


plts=np.float32([[158,60],[218,99],[163,176],[106,134],])
height=110
weight=81
plts2=np.float32([[0,height],[weight,height],[weight,0],[0,0],])
# print(plts)
# for i in range(4):
# cv.circle(img,(plts[2][0],plts[2][1]),5,(0,0,255),cv.FILLED)#绘制填充的圆
# #计算透视变换矩阵
reotation=cv.getPerspectiveTransform(plts,plts2)
#进行透视变换投影
iimg_warp=cv.warpPerspective(img,reotation,(weight,height))
dst=cv.flip(iimg_warp,1)
cv.imshow('d',img)
cv.namedWindow('dd', cv.WINDOW_NORMAL)
cv.namedWindow('kl', cv.WINDOW_NORMAL)
cv.imshow('dd',iimg_warp)
cv.imshow('kl',dst)
cv.waitKey(0)
cv.destroyAllWindows()