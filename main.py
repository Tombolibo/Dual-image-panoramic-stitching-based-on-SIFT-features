import time

import cv2
import numpy as np


# 实现拼接两张图的全景拼接（基于SIFT）
class PanoramaStitcher(object):
    def __init__(self, levels = 5):
        self._imgLeft = None  # 左图
        self._imgRight = None  # 右图
        self._imgRightH = None  # 右图的Height？
        self._imgBlended = None
        self._size = None
        self._sift = cv2.SIFT.create()  # SIFT特征提取模型
        self._bf = cv2.BFMatcher()  # 暴力匹配模型
        self._blender = cv2.detail.MultiBandBlender(0, levels)  # 多频段融合模型

    '''
    #直方图匹配，防止拼接亮度色彩对比度不均匀（弃用）
    #原图，模板，
    def hist_match(self, source, reference):
        src_hist = cv2.calcHist([source], [0], None, [256], [0,256])
        ref_hist = cv2.calcHist([reference], [0], None, [256], [0,256])
        lut = np.interp(np.cumsum(src_hist), np.cumsum(ref_hist), np.arange(256))
        return cv2.LUT(source, lut.astype('uint8'))

    '''

    # 读取图像
    def read_img(self, leftimgPath, rightimgPath, scale=1):
        self._imgLeft = cv2.imread(leftimgPath)  # 读取左图像
        self._imgRight = cv2.imread(rightimgPath)  # 读取右图像
        if scale != 1:
            # 缩放一下
            self._imgLeft = cv2.resize(self._imgLeft, None, None, scale, scale)
            self._imgRight = cv2.resize(self._imgRight, None, None, scale, scale)
        if self._size is None:
            # 让右图跟左图一样大
            self._size = (self._imgLeft.shape[1], self._imgLeft.shape[0])
            self._imgRight = cv2.resize(self._imgRight, self._size)
        else:
            # 让左右图一样大
            self._imgLeft = cv2.resize(self._imgLeft, self._size)
            self._imgRight = cv2.resize(self._imgRight, self._size)
        # print('img left shape: ', self._imgLeft.shape)
        # print('img right shape: ', self._imgRight.shape)


    #设置大小，并调整图像大小if图像不为空的话
    def set_size(self, size):
        self._size = size
        if self._imgLeft is not None:
            self._imgLeft = cv2.resize(self._imgLeft, self._size)
        if self._imgRight is not None:
            self._imgRight = cv2.resize(self._imgRight, self._size)


    #进行特征提取并拼接（特征点匹配误差，单应矩阵距离误差，融合比例
    def stich(self, distanceApprox = 0.3, distanceError = 5.0, fusionRatio = 0.1):
        #创建sift特征提取器
        kpLeft, desLeft = self._sift.detectAndCompute(cv2.cvtColor(self._imgLeft, cv2.COLOR_BGR2GRAY), None)
        kpRight, desRight = self._sift.detectAndCompute(cv2.cvtColor(self._imgRight, cv2.COLOR_BGR2GRAY), None)
        #暴力匹配特征，创建暴力匹配器
        matches = self._bf.knnMatch(desLeft, desRight, k=2)
        #提取相同特征点并定位left与right坐标
        goodMatches = []
        for m,n in matches:
            if m.distance < distanceApprox * n.distance:
                goodMatches.append(m)
        #至少要有4个点才能计算单应矩阵
        if len(goodMatches)<4:
            print('匹配点不足：{}'.format(len(goodMatches)))
            return
        else:
            #通过匹配对提取left中的点和与其在right中对应的点坐标
            print('匹配点个数：', len(goodMatches))
            leftGoodPoints = np.float32([kpLeft[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            rightGoodPodints = np.float32([kpRight[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)


            #计算单应矩阵
            H, mask = cv2.findHomography(rightGoodPodints ,leftGoodPoints , cv2.RANSAC, distanceError)
            #进行透视变换
            self._imgRightH = cv2.warpPerspective(self._imgRight, H, (self._size[0]*2, self._size[1]))
            #进行多频段融合
            maskLeft = np.ones(self._imgLeft.shape[:2], dtype=np.uint8)*255
            maskRight = np.zeros(self._imgRightH.shape[:2], dtype=np.uint8)
            maskRight[:,int(self._imgRightH.shape[1]//2*(1-fusionRatio)):] = 255
            # cv2.imshow('mask left', maskLeft)
            # cv2.imshow('mask right', maskRight)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()
            #准备参数
            self._blender.prepare([0,0,self._imgRightH.shape[1], self._imgRightH.shape[0]])
            #喂食
            self._blender.feed(self._imgLeft, maskLeft, (0,0))
            self._blender.feed(self._imgRightH, maskRight, (0,0))
            #频段融合
            self._imgBlended, maskBlended = self._blender.blend(None, None)
            self._imgBlended = cv2.convertScaleAbs(self._imgBlended)
            # # 进行融合后边缘黑边处理
            # imgBlendedCanny = cv2.Canny(self._imgBlended, 50,100)
            # # imgBlendedThresh = cv2.medianBlur(imgBlendedThresh, 9)
            # contours, hirearchy = cv2.findContours(imgBlendedCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print('轮廓个数：', len(contours))
            #
            # # 挑出最大轮廓
            # maxContourID = np.argmax([cv2.contourArea(contour) for contour in contours])
            # # 多边形近似
            # poly = cv2.approxPolyDP(contours[maxContourID], 0.05*cv2.arcLength(contours[maxContourID], True), True)
            # # 计算最大轮廓的外接矩形
            # rect = cv2.boundingRect(poly)
            # print('rect: ', rect)
            #
            # cv2.drawContours(imgBlendedCanny, [poly], -1, 127, 10)
            #
            # print(maxContourID)
            #
            # cv2.imshow('imgBlended', imgBlendedCanny)
            # cv2.imshow('self._imgBlended', self._imgBlended)
            # cv2.waitKey(0)
            #
            #
            # print('imgBlended.shape: ', self._imgBlended.shape)

if __name__ == '__main__':
    myPanoramaStitcher = PanoramaStitcher()
    myPanoramaStitcher.read_img(r'./carp1.jpg', r'./carp2.jpg', 0.5)

    t1 = time.time()
    myPanoramaStitcher.stich(0.7, 5, 0.1)
    print('合成时间：', 1000*(time.time()-t1), 'ms')


    #展示融合效果
    cv2.imshow('imgLeft', myPanoramaStitcher._imgLeft)
    cv2.imshow('imgRight', myPanoramaStitcher._imgRight)
    cv2.imshow('imgStitch', myPanoramaStitcher._imgBlended)
    print('imgleft.shape: ', myPanoramaStitcher._imgLeft.shape)
    print('imgRight.shape: ', myPanoramaStitcher._imgRight.shape)
    print('imgBlended.shape: ', myPanoramaStitcher._imgBlended.shape)
    cv2.waitKey(0)

