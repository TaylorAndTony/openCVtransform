from cv2 import cv2
import numpy as np
from time import sleep
from pynput import keyboard
from threading import Thread


# 固定尺寸
def resizeImg(image, height=900):
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img


# 边缘检测
def getCanny(image):
    """ 边缘检测 """
    # 高斯模糊
    binary = cv2.GaussianBlur(image, (3, 3), 2, 2)
    # 边缘检测
    binary = cv2.Canny(binary, 60, 240, apertureSize=3)
    # 膨胀操作，尽量使边缘闭合
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def findMaxContour(image):
    """ 寻找边缘 """
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 计算面积
    max_area = 0.0
    max_contour = []
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > max_area:
            max_area = currentArea
            max_contour = contour
    return max_contour, max_area


def getBoxPoint(contour):
    """ 多边形拟合凸包 """
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx


def draw_boxes_on_img(image):
    """ 在传入的图片上画出边角
    返回：画图后的图片，坐标，是否能画图
    """
    # 边缘检测
    img = image.copy()
    try:
        binary_img = getCanny(img)
        # 找边缘
        max_contour, max_area = findMaxContour(binary_img)
        # 拟合矩形
        boxes = getBoxPoint(max_contour)
        # 画矩形
        cv2.drawContours(img, max_contour, -1, (0, 255, 0), 9)
        # print(boxes)
        # [ 369  355]
        #  [2152  465]
        #  [2364 2776]
        #  [ 400 2985]
        # 画圆圈
        for box in boxes:
            cv2.circle(img, tuple(box), 30, (0, 255, 255), 10)
        return img, boxes, True
    except:
        print('无法检测')
        boxes = None
        return image, boxes, False


def fix_perspective(img, points):
    """ 对图像进行自动透视修正 """
    # 左上，左下，右下，右上
    # 原始四点坐标
    img_copy = img.copy()
    try:
        n = [points[0], points[3], points[2], points[1]]
    except:
        print('无法构建透视四边形')
        return
    p1 = np.array(n, dtype=np.float32)
    # 对应四点坐标，左上，左下，右下，右上
    # p2 = np.array([(0, 0), (0, 1500), (1000, 1500), (1000, 0)],
    #               dtype=np.float32)
    # 自动计算 p2:
    LT = points[0]
    LB = points[3]
    RB = points[2]
    # 高度 = 开方（（左上纵坐标-左下纵坐标）^2 + （左上横坐标-左下横坐标））
    height = np.sqrt((LB[1] - LT[1])**2 + (LB[0] - LT[0])**2)
    # 左下和右下点的距离
    width = np.sqrt((LB[1] - RB[1])**2 + (LB[0] - RB[0])**2)
    # 对应四点坐标，左上，左下，右下，右上
    p2 = np.array([(0, 0), (0, height), (width, height), (width, 0)],
                  dtype=np.float32)
    # print('目标点：', p2)
    try:
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
    except cv2.error:
        print('无法矫正透视')
        return
    # 使用透视变换
    result = cv2.warpPerspective(img_copy, M, (0, 0))
    # 重新截取
    result = result[:int(p2[1][1] + 1), :int(p2[2][0] + 1)]
    # cv2.imwrite(f'./output/perspective_fix.png', result)
    return result


class ImageProcessor:
    def __init__(self, filename):
        self.filename = filename
        self.save_now = False
        # 启动键盘监听
        self.keyboard_response()

    def keyboard_control(self, key):
        """ callback func to handel keystokes """
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        if k == 's':
            self.save_now = True

    def keyboard_response(self):
        """ use threading to begin keyboard listenning """
        def __t():
            with keyboard.Listener(on_release=self.keyboard_control) as listener:
                listener.join()
        t = Thread(target=__t)
        t.setDaemon(True)
        t.start()

    def manual(self, input_image):
        """ 自动打开图片，识别透视并保存 """
        path = input_image
        img = cv2.imread(path)
        print(type(img))
        # img = resizeImg(img)
        box, points, can_do = draw_boxes_on_img(img)
        cv2.imwrite(f'./output/{self.filename}-boxes.png', box)
        # cv2.imwrite(outpath, box)
        if can_do:
            img = fix_perspective(img, points)
            if img is not None:
                cv2.imwrite(f'./output/{self.filename}.png', img)
                print('已保存')
            else:
                print('无法保存')

    def cam_loop(self):
        """ 调用摄像头自动识别边角 """
        cap = cv2.VideoCapture(0)
        while True:
            _, original = cap.read()
            box, points, can_do = draw_boxes_on_img(original)
            # 显示实时画图
            cv2.imshow('realtime', box)
            # save
            if self.save_now:
                self.save_now = False
                # 检测有没有边缘
                if can_do:
                    fixed = fix_perspective(original, points)
                    # 检测能不能修复
                    if fixed is not None:
                        cv2.imwrite(f'./output/{self.filename}-fixed.png', fixed)
                        print('已保存')
                    else:
                        print('无法保存')
                else:
                    print('无法完成透视修正')
            # debug
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            sleep(0.02)


if __name__ == '__main__':
    processer = ImageProcessor('camera')
    # processer.manual('paper.jpg')
    processer.cam_loop()
