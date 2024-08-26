import cv2
import numpy
import math
import time
import serial

ser = serial.Serial("/dev/ttyAMA2", 115200)
if not ser.isOpen():
    print("open failed")
else:
    print("open success: ")
    print(ser)

# 640x480
cap = cv2.VideoCapture(0)

while cap.isOpened():
    start_time = time.time()

    qz_data_lb = numpy.zeros((3, 3, 10), numpy.int8)
    for e in range(10):
        qz_data = numpy.zeros((3, 3), numpy.int8)
        ret, frame = cap.read()
        img_qp = frame.copy()
        img_qz = frame.copy()
        # 识别棋盘
        img_qp = img_qp[:, :, 0]
        # 大津法
        img_qp = cv2.threshold(img_qp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        k1 = numpy.ones((3, 3), numpy.uint8)
        img_qp = cv2.morphologyEx(img_qp, cv2.MORPH_CLOSE, k1)

        # # 拟合棋盘边框
        # contours, hierarchy = cv2.findContours(img_qp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # if contours is not None:
        #     max_cnt = contours[0]
        #     max_len = cv2.arcLength(max_cnt, True)
        # for contour in contours:
        #     cnt_len = cv2.arcLength(contour, True)
        #     if cnt_len > max_len and cnt_len > 900 and cnt_len < 2000:
        #         cnt = cv2.approxPolyDP(contour, 0.01 * cnt_len, True)
        #         if len(cnt) == 4:
        #             max_cnt = cnt
        #             cv2.drawContours(frame, [max_cnt], -1, (255, 255, 0), 3)
        max_cnt = [[[124 ,115]],[[391 ,117]],[[395, 375]],[[130, 383]]]
        # if len(max_cnt) == 4:
        #     epoch = 4
        #     while (max_cnt[0][0][0] > 320 or max_cnt[0][0][1] > 240) and epoch:
        #         xy_cz = max_cnt.copy()
        #         epoch = epoch - 1
        #         max_cnt[0] = xy_cz[1]
        #         max_cnt[1] = xy_cz[2]
        #         max_cnt[2] = xy_cz[3]
        #         max_cnt[3] = xy_cz[0]
        #     if max_cnt[1][0][0] < 320 :
        #         xy_cz = max_cnt.copy()
        #         max_cnt[1] = xy_cz[3]
        #         max_cnt[3] = xy_cz[1]
        #
        #     if max_cnt[2][0][0] != max_cnt[3][0][0]:
        #         k = (max_cnt[2][0][1] - max_cnt[3][0][1]) / (max_cnt[2][0][0]-max_cnt[3][0][0])
        #         degree = int(math.degrees(math.atan(-k)))
        #print(max_cnt)
        width = int((max_cnt[1][0][0] - max_cnt[0][0][0])/3)
        height = int((max_cnt[3][0][1] - max_cnt[0][0][1])/3)


        img_qp = cv2.GaussianBlur(img_qp, (5, 5), 2)
        cv2.imshow('img_qp', img_qp)
        # 使用霍夫圆变换检测圆形
        circles_b = cv2.HoughCircles(
            img_qp,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=37,
            param1=50,
            param2=37,
            minRadius=10,
            maxRadius=40
        )

        # 黑色棋子
        if circles_b is not None:
            circles_b = numpy.round(circles_b[0, :]).astype("int")
            for (x, y, r) in circles_b:
                for i in range(3):
                    if max_cnt[0][0][0] + i*width < x and x < max_cnt[0][0][0] + (i+1)*width:
                        for j in range(3):
                            if max_cnt[0][0][1] + j * height < y and y < max_cnt[0][0][1] + (j + 1) * height:
                                qz_data[j][i] = 1

                # 绘制圆形边界
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)

            img_qz = cv2.cvtColor(img_qz, cv2.COLOR_BGR2GRAY)
            k1 = numpy.ones((3, 3), numpy.uint8)
            img_qz = cv2.morphologyEx(img_qz, cv2.MORPH_CLOSE, k1)

            # 预处理图像
            img_qz = cv2.GaussianBlur(img_qz, (9, 9), 2)
            #cv2.imshow('frame', img_qz)
            # 使用霍夫圆变换检测圆形
            cv2.imshow('img_qz', img_qz)
            circles_w = cv2.HoughCircles(
                img_qz,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=37,
                param1=50,
                param2=37,
                minRadius=10,
                maxRadius=40
            )
            # 如果找到圆形，则进行绘制
            if circles_w is not None:
                circles_w = numpy.round(circles_w[0, :]).astype("int")
                for (x, y, r) in circles_w:
                    for i in range(3):
                        if (max_cnt[0][0][0] + i * width < x and x < max_cnt[0][0][0] + (i + 1) * width):
                            for j in range(3):
                                if (max_cnt[0][0][1] + j * height < y and y < max_cnt[0][0][1] + (j + 1) * height) and qz_data[j][i] != 1:
                                    qz_data[j][i] = 2
                                    cv2.circle(frame, (x, y), r, (255, 0, 0), 4)
        #print(qz_data)
        cv2.imshow('frame', frame)
        for r in range(3):
            for c in range(3):
                qz_data_lb[r][c][e] = qz_data[r][c]
                #print(qz_data_lb)

    for r in range(3):
        for c in range(3):
            qz_data[r][c] = round(sum(qz_data_lb[r][c])/len(qz_data_lb[r][c]))
    ser.write('@'.encode('utf-8'))
    ser.write(f'{qz_data[0][0]}'.encode('utf-8'))
    ser.write(f'{qz_data[0][1]}'.encode('utf-8'))
    ser.write(f'{qz_data[0][2]}'.encode('utf-8'))
    ser.write(f'{qz_data[1][0]}'.encode('utf-8'))
    ser.write(f'{qz_data[1][1]}'.encode('utf-8'))
    ser.write(f'{qz_data[1][2]}'.encode('utf-8'))
    ser.write(f'{qz_data[2][0]}'.encode('utf-8'))
    ser.write(f'{qz_data[2][1]}'.encode('utf-8'))
    ser.write(f'{qz_data[2][2]}'.encode('utf-8'))
    ser.write('#'.encode('utf-8'))
    ser.write('*'.encode('utf-8'))
    print(qz_data)
    end_time = time.time()
    execution_time = end_time - start_time
    print(1/execution_time)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        break
cv2.destroyAllWindows()
cap.release()
