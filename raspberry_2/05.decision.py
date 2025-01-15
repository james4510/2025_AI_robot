import time
import cv2
from picamera2 import Picamera2
import numpy as np

def make_black(image, threshold=140):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_image = cv2.inRange(gray_image, threshold, 255)
    return black_image, gray_image

def path_decision(image, limit=150):
    height, width = image.shape
    image = image[height-limit:height-10, :]
    height = limit - 1
    width = width - 1
    image = np.flipud(image)
    
    mask = image != 0
    white_distance = np.where(mask.any(axis=0), mask.argmax(axis=0), height)
    left = 0
    right = width
    center = int((left + right) / 2)
    left_sum = np.sum(white_distance[left:center-60])
    right_sum = np.sum(white_distance[center+60:right])
    forward_sum = np.sum(white_distance[center-60:center+60])
    print(left_sum, right_sum, forward_sum)

    # 방향 결정
    if forward_sum > 12000:
        decision = 'f'
    elif left_sum > right_sum:
        decision = 'l'
    elif left_sum <= right_sum:
        decision = 'r'
    elif forward_sum < 500:
        decision = 'b'
    else:
        decision = 'except'
    return decision

# Picamera2 초기화
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

time.sleep(0.05)  # 카메라 초기화 대기

while True:
    # 프레임 캡처
    frame = picam2.capture_array()
    # 이미지 표시
    cv2.imshow("image", frame)

    # 이진화 및 경로 결정
    black, gray = make_black(frame)
    decision = path_decision(black)
    print(decision)

    # 텍스트와 사각형 표시
    cv2.rectangle(frame, (0, 10), (320, 90), (0, 255, 0), 3)
    cv2.putText(frame, decision, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
    
    # 이미지 출력
    cv2.imshow("image", frame)
    cv2.imshow("black", black)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()