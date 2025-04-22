# Dual image panoramic stitching based on SIFT features
## 基于SIFT特征的双图全景拼接

使用SIFT尺度不变换特征对两张图片进行全景拼接。提取SIFT特征，并进行特征点匹配，根据匹配情况计算单应矩阵最后进行透视变换，以实现全景拼接。对接缝处的色彩亮度不一情况使用了多频段融合方式以消除。<br/>
main.py为自实现，可以进行调参、观察中间效果、更换匹配器、调整多频段融合效果等。<br/>
Stitcher全景拼接.py为opencv提供全景拼接器，可实现多图全景拼接。<br/>

## 为了保证图片信息完整性未去除黑边
