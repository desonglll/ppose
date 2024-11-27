import json

from django.shortcuts import render

import cv2
from django.http import StreamingHttpResponse

from .algos.pose import PoseDetector

cap = cv2.VideoCapture(1)
detector = PoseDetector()


def get_frame(param):
    while True:
        ret, img = cap.read()
        if not ret:
            break

        angle = 0  # 初始化角度

        if param == "1":
            img = detector.find_pose(img)
        elif param == "2":
            img, landmarks = detector.find_position(img, convert_to_x_y_pixel=True, draw=True)
        elif param == "3":
            img, angle = detector.find_angle(img, 16, 14, 12)
        elif param == "4":
            img, angle = detector.find_angle_with_horizontal(img, 5, 10, draw=True)
        elif param == "5":
            img, angle = detector.find_angle_with_horizontal_mean(img, 2, 5, 9, 10, draw=True)
        elif param == "6":
            img, angle = detector.find_angle_with_horizontal_mean_all(img, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, draw=True)

        _, buffer = cv2.imencode('.jpg', img)

        # 将图像数据和角度信息组合成 JSON
        data = {
            "angle": angle,
            "image": buffer.tobytes().decode("latin1")  # 将二进制数据转换为可传输字符串
        }
        yield (
            b'--frame\r\n'
            b'Content-Type: application/json\r\n\r\n' +
            json.dumps(data).encode('utf-8') +
            b'\r\n'
        )

def video_feed(request):
    param = request.GET.get('param', '1')
    return StreamingHttpResponse(get_frame(param), content_type='multipart/x-mixed-replace; boundary=frame')


def video_page(request):
    return render(request, 'testcv2/video_page.html')
