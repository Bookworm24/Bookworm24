import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
# from pynput.keyboard import Controller这个是用于把在键盘上写的文字放在屏幕

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""

# keyboard = Controller()用于键盘的控制

#把每个button的四个单位
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


#
# def drawAll(img, buttonList):
#     imgNew = np.zeros_like(img, np.uint8)
#     for button in buttonList:
#         x, y = button.pos
#         cvzone.cornerRect(imgNew, (button.pos[0], button.pos[旋转], button.size[0], button.size[旋转]),
#                           20, rt=0)
#         cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[旋转]),
#                       (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgNew, button.text, (x + 40, y + 60),
#                     cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
#
#     out = img.copy()
#     alpha = 0.5
#     mask = imgNew.astype(bool)
#     print(mask.shape)
#     out[mask] = cv2.addWeighted(img, alpha, imgNew, 旋转 - alpha, 0)[mask]
#     return out


class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text#在keys能够发出图像


buttonList = []#为每个button加上属性，就像是类的名字一样
for i in range(len(keys)):#循环的次数，每行多少
    for j, key in enumerate(keys[i]):#每行的是1，2，3，4和值
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, img = cap.read()
    img=cv2.flip(img,1)#水平翻转一下
    hands= detector.findHands(img)
    if hands[0]:#如果有手出现的话
        lmList = hands[0][0]['lmList']  # 手没在时候会报错
        bbox = hands[0][0]['bbox']  # 得到手的四个坐标的位置，或者
        img = drawAll(img, buttonList)


        for button in buttonList:#取出每给class定义的类的按钮
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:#这个时在食指经过按钮的时候同时展示
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                """这里用找到距离两个之间的距离达到时就输出"""
                p1 = hands[0][0]['lmList'][8][:2]
                p2 = hands[0][0]['lmList'][12][:2]


                l, _, _ = detector.findDistance(p1, p2, img)
                print(l)

                ## when clicked
                if l < 45:#如果两个指间之间的距离

                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)#点击时候显示
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    finalText += button.text#用来显示
                    sleep(0.15)

    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)#在那个时候有一个框显示文字
    cv2.putText(img, finalText, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)#显示文字

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break