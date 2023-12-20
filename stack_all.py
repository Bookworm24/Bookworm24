import cv2
import numpy as np
def stack_img(scale,imgarry):
    rows=len(imgarry)
    # print(imgarry[0].shape)
    # print(imgarry[1].shape)
    # print(len(imgarry[0]))
    # print('rows',rows)
    cols=len(imgarry[0])
    # print('cols',cols)
    rowsavialiable=isinstance(imgarry[0],list)#判断是单张图片还是多张图片
    # print(rowsavialiable)
    width=imgarry[0][0].shape[1]#128个，3列
    # print(imgarry[0][0])#取出一行
    #1280print(len(imgarry[0][0]))

    #print(imgarry[0].shape)720,1280
    height=imgarry[0][0].shape[0]
    if rowsavialiable:
        for x in range(0,rows):#取每个图像
            for y in range(0,cols):#原图像一共有720行高度，
                if imgarry[x][y].shape[:2]==imgarry[0][0].shape[:2]:#如果第1,2,3个图像的元素的第n个高度真的等于第一个图像高度
                    imgarry[x][y]=cv2.resize(imgarry[x][y],(0,0),None,scale,scale)#调整大小，在设置了x,y比例后长宽就不限制
                else:#如果高度不等
                    imgarry[x][y]=cv2.resize(imgarry[x][y],(imgarry[0][0].shape[1],imgarry[0][0].shape[0]),None,scale,scale)
                if len(imgarry[x][y].shape)==2:imgarry[x][y]=cv2.cvtColor(imgarry[x][y],cv2.COLOR_GRAY2BGR)#灰度图像
        imgblank=np.zeros((height,width,3),np.uint8)#生成零矩阵
        hor=[imgblank]*rows#生成3个
        hor_con=[imgblank]*rows
        for x in range(0,rows):
            hor[x]=np.hstack(imgarry[x])#将3个空白与之叠加
        ver=np.vstack(hor)
        #for-else语句是一种特殊的语法结构，用于在循环结束后执行一些操作
    else:
        for x in range(0,rows):
            if imgarry[x].shape[:2]==imgarry[0].shape[:2]:
                imgarry[x]=cv2.resize(imgarry[x],(0,0),None,scale,scale)
            else:
                imgarry[x]=cv2.resize(imgarry[x],(imgarry[0].shape[1],imgarry[0].shape[0]),None,scale,scale)
            if len(imgarry[x].shape) == 2: imgarry[x] = cv2.cvtColor(imgarry[x], cv2.COLOR_GRAY2BGR)
        hor=np.hstack(imgarry)

        ver=hor

    return ver

