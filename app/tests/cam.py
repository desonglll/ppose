import cv2

# 尝试不同的摄像头序号
for i in range(10):  # 假设最多有5个摄像头
    cap = cv2.VideoCapture(i)
    if cap.isOpened():  # 如果摄像头成功打开
        print(f"摄像头序号 {i} 可用")
        cap.release()  # 关闭摄像头
    else:
        print(f"摄像头序号 {i} 不可用")
