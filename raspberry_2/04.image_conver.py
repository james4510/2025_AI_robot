import time
import cv2
from picamera2 import Picamera2                                                                 # type: ignore
import numpy as np

def make_black(image, threshold=140):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_image = cv2.inRange(gray_image, threshold, 255)
    return black_image, gray_image

# Picamera2 초기화
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

time.sleep(0.05)  # 카메라 초기화 시간 대기

try:
    while True:
        # 프레임 캡처
        frame = picam2.capture_array()

        # 컬러 공간 변환 (RGB에서 BGR로 변환)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 화면 Flip (상하좌우 반전)
        frame_bgr_flipped = cv2.flip(frame_bgr, -1)  # flipCode=-1은 상하좌우 반전

        # 그레이스케일 및 임계값 적용
        black, gray = make_black(frame_bgr_flipped)
        
        # 이미지 표시
        cv2.imshow("color", frame_bgr_flipped)
        cv2.imshow("gray", gray)
        cv2.imshow("black", black)
        
        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
