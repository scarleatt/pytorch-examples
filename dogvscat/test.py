#-*- coding: utf-8 -*-
import cv2
import numpy as np
 
def Process(img):

    # 高斯平滑
    gaussian = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    #cv2.GaussianBlur(src,ksize,sigmaX[,sigmaxY[,borderType]]])高斯滤波函数
    #src: 输入图像
    #ksize: 高斯内核大小,元组类型
    #sigmaX: 高斯核函数在X方向上的标准偏差
    #sigmaY: 高斯核函数在Y方向上的标准偏差，如果sigmaY是0，则函数会自动将sigmaY的值设置为与sigmaX相同的值，如果sigmaX和sigmaY都是0，这两个值将由ksize[0]和ksize[1]计算而来。建议将size、sigmaX和sigmaY都指定出来。
    #borderType: 推断图像外部像素的某种便捷模式，有默认值cv2.BORDER_DEFAULT，如果没有特殊需要不用更改，具体可以参考borderInterpolate()函数。
    
    # 中值滤波
    median = cv2.medianBlur(gaussian, 5)
    #cv2.medianBlur(src,ksize)
    #src: 输入图像
    #ksize: 高斯内核大小,元组类型

    # Sobel算子
    # 梯度方向: x
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    # 利用Sobel方法可以进行sobel边缘检测
    # img表示源图像，即进行边缘检测的图像
    # 图像深度是cv2.CV_8U、cv2.CV_16U、cv2.CV_16S、cv2.CV_32F以及cv2.CV_64F其中的某一个
    # 第三和第四个参数分别是对X和Y方向的导数（即dx,dy），对于图像来说就是差分，这里1表示对X求偏导（差分），0表示不对Y求导（差分）。其中，X还可以求2次导。
    # 注意：对X求导就是检测X方向上是否有边缘。
    # 第五个参数ksize是指核的大小。
    # 这里说明一下，这个参数的前四个参数都没有给谁赋值，而ksize则是被赋值的对象
    # 实际上，这是可省略的参数，而前四个是不可省的参数。注意其中的不同点值化
    
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    #灰度值小于175的点置0，灰度值大于175的点置255
    #最后的参数是阈值类型，对应一个公式，一般用这个就可以
    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    #第一个参数定义结构元素，如椭圆（MORPH_ELLIPSE）、交叉形结构（MORPH_CROSS）和矩形（MORPH_RECT）
    #第二个参数就是指和函数的size，9×1
  
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细小杂点
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓更明显
    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    
    #存储中间图片 
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)
    cv2.imwrite("erosion.png", erosion)
    cv2.imwrite("dilation2.png", dilation2)
    
    return dilation2
def GetRegion(img):
    regions = []
    # 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #三个输入参数：输入图像（二值图像），轮廓检索方式，轮廓近似方法
    #轮廓检索方式：cv2.RETR_EXTERNAL   只检测外轮廓。cv2.RETR_LIST    检测的轮廓不建立等级关系。cv2.RETR_CCOMP 建立两个等级的轮廓，上面一层为外边界，里面一层为内孔的边界信息。cv2.RETR_TREE   建立一个等级树结构的轮廓
    #轮廓近似方法：cv2.CHAIN_APPROX_NONE   存储所有边界点。cv2.CHAIN_APPROX_SIMPLE 压缩垂直、水平、对角方向，只保留端点。cv2.CHAIN_APPROX_TX89_L1 使用teh-Chini近似算法。cv2.CHAIN_APPROX_TC89_KCOS  使用teh-Chini近似算法
    #三个返回值：图像，轮廓，轮廓的层析结构

    #筛选面积小的
    for contour in contours:
        #计算该轮廓的面积
        area = cv2.contourArea(contour)
        #面积小的都筛选掉
        if (area < 2000):
            continue
        #轮廓近似，作用很小
        epslion = 1e-3 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epslion, True)
        #epsilon，是从轮廓到近似轮廓的最大距离。是一个准确率参数，好的epsilon的选择可以得到正确的输出。True决定曲线是否闭合。
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(contour)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ## 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        #车牌正常情况下长高比在2-5之间
        ratio =float(width) / float(height)
        if (ratio < 5 and ratio > 2):
            regions.append(box)
    return regions
    
def detect(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 预处理及形态学处理，得到可以查找矩形的图片
    prc = Process(gray)
    #得到车牌轮廓
    regions = GetRegion(prc)
    print('[INFO]:Detect %d license plates' % len(regions))
    #用绿线画出这些找到的轮廓
    for box in regions:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        #五个输入参数：原始图像，轮廓，轮廓的索引（当设置为-1时，绘制所有轮廓），画笔颜色，画笔大小
        #一个返回值：返回绘制了轮廓的图像
    cv2.imshow('Result', img)
    #保存结果文件名
    cv2.imwrite('result.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    #输入的参数为图片的路径
    img = cv2.imread('1.jpg')
    detect(img)
