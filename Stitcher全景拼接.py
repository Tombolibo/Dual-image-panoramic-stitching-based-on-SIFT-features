import cv2
import numpy as np

#原始图像
imgLeft = cv2.imread(r'p1.jpg')
imgMid = cv2.imread('p3.jpg', cv2.IMREAD_COLOR)
imgRight = cv2.imread(r'p2.jpg')

cv2.namedWindow('left', cv2.WINDOW_NORMAL)
cv2.namedWindow('mid', cv2.WINDOW_NORMAL)
cv2.namedWindow('right', cv2.WINDOW_NORMAL)
cv2.imshow('left', imgLeft)
cv2.imshow('mid', imgMid)
cv2.imshow('right', imgRight)
cv2.waitKey(0)

stitcher = cv2.Stitcher.create()

# stitcher.setPanoConfidenceThresh(0.5125)    # 降低匹配置信度阈值（默认0.8）
# stitcher.setRegistrationResol(0.3)       # 降低配准分辨率（默认0.6，加快计算）

result = stitcher.stitch([imgLeft,imgMid, imgRight])
# result = stitcher.stitch([result[1], imgRight])
print(result[0])

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test', result[1])
cv2.waitKey(0)