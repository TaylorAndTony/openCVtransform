from threading import Thread
from time import sleep

import numpy as np
from cv2 import cv2 as cv
from pynput import keyboard


def contour_demo(img):
    """ 预处理，高斯滤波（用处不大），4次开操作 """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 1)
    ref, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=4)
    contours, hierachy = cv.findContours(thresh, cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    return contours


def fix_perspective(img, save=True):
    """ 对图像进行自动透视修正 """
    contours = contour_demo(img)
    # 轮廓唯一，以后可以扩展
    # print(contours)
    contour = contours[0]
    # 求周长，可在后面的转换中使用周长和比例
    # print('周长：', cv.arcLength(contour, True))
    img_copy = img.copy()
    # 使用approxPolyDP，将轮廓转换为直线，22为精度（越高越低），TRUE为闭合
    approx = cv.approxPolyDP(contour, 22, True)
    n = []
    # 生产四个角的坐标点
    for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
        n.append((x, y))
    # 左上，左下，右下，右上
    # p1: [[ 370.  356.], [ 402. 2981.], [2360. 2775.], [2151.  466.]]
    # <class 'numpy.ndarray'>
    # 原始四点坐标
    p1 = np.array(n, dtype=np.float32)
    # 对应四点坐标，左上，左下，右下，右上
    # p2 = np.array([(0, 0), (0, 1500), (1000, 1500), (1000, 0)],
    #               dtype=np.float32)
    # 自动计算 p2:
    LT = p1[0]
    LB = p1[1]
    RB = p1[2]
    # 高度 = 开方（（左上纵坐标-左下纵坐标）^2 + （左上横坐标-左下横坐标））
    height = np.sqrt((LB[1] - LT[1])**2 + (LB[0] - LT[0])**2)
    # 左下和右下点的距离
    width = np.sqrt((LB[1] - RB[1])**2 + (LB[0] - RB[0])**2)
    # 对应四点坐标，左上，左下，右下，右上
    p2 = np.array([(0, 0), (0, height), (width, height), (width, 0)],
                  dtype=np.float32)
    # print('目标点：', p2)
    M = cv.getPerspectiveTransform(p1, p2)  # 变换矩阵
    # 使用透视变换
    result = cv.warpPerspective(img_copy, M, (0, 0))
    # 重新截取
    result = result[:int(p2[1][1] + 1), :int(p2[2][0] + 1)]
    if save:
        cv.imwrite('fixed.png', result)
    return result


def single_image():
    src = cv.imread('page.jpg')
    fix_perspective(src)


def keyboard_control(key):
    """ callback func to handel keypress """
    global RUNNING
    global ACTIVE
    try:
        k = key.char
    except AttributeError:
        k = str(key)
    if k == 's':
        RUNNING = False
    elif k == 'a':
        ACTIVE = True


def keyboard_response():
    """ use threading to begin keyboard listenning """
    def __t():
        with keyboard.Listener(on_release=keyboard_control) as listener:
            listener.join()

    t = Thread(target=__t)
    t.setDaemon(True)
    t.start()


def cam_loop():
    """ 摄像头采集 """
    global ACTIVE
    keyboard_response()
    cam = cv.VideoCapture(0)
    while RUNNING:
        _, original = cam.read()
        cv.imshow('result', original)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if ACTIVE:
            ACTIVE = False
            print('triggered')
            cv.imwrite('cap.png', original)
            print('Done')
            sleep(0.1)
    cv.destroyAllWindows()


def stack_loop(nums=5, delay=0.2):
    """
    拍摄堆栈图像
    :param nums: 堆栈次数
    :param delay: 每次拍摄的延时
    """
    global ACTIVE
    # 计数器和图片列表
    current = 0
    images = []
    keyboard_response()
    cam = cv.VideoCapture(0)
    while RUNNING:
        _, original = cam.read()
        # 展示窗口
        cv.imshow('result', original)
        # 退出程序判断
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # 激活程序判断
        if ACTIVE:
            # 连拍控制
            current += 1
            if current >= nums:
                ACTIVE = False
                current = 0
            # 载入列表
            images.append(original)
            print(f'progress: {current} / {nums}')
            # 等待
            sleep(delay)
        # 完成连拍
        if len(images) == nums:
            print('finished sampling')
            # 叠加堆栈
            # 第一张图像
            stacked = images[0]
            for image in images[1:]:
                cv.addWeighted(stacked, 0.5, image, 0.5, 0, stacked)
            images = []
            cv.imwrite('stacked.png', stacked)
            print('stack operation finished')
    cv.destroyAllWindows()


if __name__ == '__main__':
    RUNNING = True
    ACTIVE = False
    stack_loop(10)
