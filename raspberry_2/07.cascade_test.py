import time
import cv2
from picamera2 import Picamera2

def make_black(image, threshold = 140):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_image=cv2.inRange(gray_image, threshold, 255)
    return black_image, gray_image

def signal(image, path): #cascade
    signal_cascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = signal_cascade.detectMultiScale(gray, 1.3,5, minSize=(35,35))
    return faces

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
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 이진화 및 경로 결정
        black, gray = make_black(image)
        faces=signal(image,'./cascade.xml')
        if faces is ():
            cv2.putText(image, "no stop", (20, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        else:
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        
        cv2.imshow("image", image)
        cv2.imshow("black",black)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
