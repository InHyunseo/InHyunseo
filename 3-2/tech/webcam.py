import cv2

# CSI 카메라는 이 함수를 통해서 열어야 화질이 좋고 안 끊깁니다.
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# 사용법: 숫자 대신 함수를 호출해서 넣습니다.
print(gstreamer_pipeline(sensor_id=0)) # 파이프라인 문자열 확인용
cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("CSI 카메라가 잘 열렸습니다!")
else:
    print("카메라를 열 수 없습니다. 선이 잘 꼽혔는지 확인하세요.")