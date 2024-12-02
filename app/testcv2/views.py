import json
import base64  # 用于 Base64 编码图像数据
from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from .algos.pose import PoseDetector

cap = cv2.VideoCapture(0)
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
            if landmarks[7].x > landmarks[0].x:
                print("Camera on LEFT")
            else:
                print("Camera on RIGHT")

        elif param == "3":
            img, angle = detector.find_angle(img, 16, 14, 12)
        elif param == "4":
            img, angle = detector.find_angle_with_horizontal(img, 5, 10, draw=True)
        elif param == "5":
            img, landmarks = detector.find_position(img, convert_to_x_y_pixel=True, draw=False)
            if landmarks:

                if landmarks[7].x > landmarks[0].x:  # 判断摄像头所处的左右
                    img, angle = detector.find_angle_with_horizontal_mean(img, 2, 5, 9, 10, side="left", draw=True,
                                                                          fix=80)
                else:
                    img, angle = detector.find_angle_with_horizontal_mean(img, 2, 5, 9, 10, side="right", draw=True)
        elif param == "6":
            img, landmarks = detector.find_position(img, convert_to_x_y_pixel=True, draw=False)
            if landmarks[7].x > landmarks[0].x:  # 判断摄像头所处的左右
                img, angle = detector.find_angle_with_horizontal_mean_all(img, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, draw=True,
                                                                          side="left")
            else:
                img, angle = detector.find_angle_with_horizontal_mean_all(img, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, draw=True,
                                                                          side="right")

        _, buffer = cv2.imencode('.jpg', img)

        # 将图像数据编码为 Base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        if not angle:
            angle = 0

        # 将图像数据和角度信息组合成 JSON
        data = {
            "angle": angle - 10,
            "image": image_base64  # Base64 编码后的图像
        }
        yield f"data: {json.dumps(data)}\n\n"


def video_feed(request):
    param = request.GET.get('param', '1')
    return StreamingHttpResponse(
        get_frame(param),
        content_type='text/event-stream',  # 使用 SSE
    )


def single_video_page(request):
    return render(request, 'testcv2/single_video_page.html')


def index_page(request):
    return render(request, 'testcv2/index.html')


def double_video_page(request):
    return render(request, 'testcv2/double_video_page.html')
