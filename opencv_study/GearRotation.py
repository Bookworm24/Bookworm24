import cvzone
import cv2
import numpy as np

angle = 0


def empty(a):
    pass
def rotateImage(imgInput, angle, scale=1, keepSize=False):

    h, w = imgInput.shape[:2]

    # Calculate the center of the original image
    center = (w // 2, h // 2)

    # Calculate the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)

    # Perform the actual rotation and return the image
    imgOutput = cv2.warpAffine(src=imgInput, M=rotate_matrix, dsize=(w, h))

    return imgOutput



cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 100)
cv2.createTrackbar("Speed", "Parameters", 1, 25, empty)

while True:
    imgBack = np.ones((500, 800, 3), np.uint8) * 255
    imgG1 = cv2.imread("Resources/gear.png", cv2.IMREAD_UNCHANGED)
    imgG2 = imgG1.copy()

    val = cv2.getTrackbarPos("Speed", "Parameters")
    imgG1 = rotateImage(imgG1, angle + 23)
    imgG2 = rotateImage(imgG2, -angle)
    angle += val

    imgResult = cvzone.overlayPNG(imgBack, imgG1, [125, 100])
    imgResult = cvzone.overlayPNG(imgResult, imgG2, [400, 100])


    cv2.imshow("Image", imgResult)
    cv2.waitKey(1)
