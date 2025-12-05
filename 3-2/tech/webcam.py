import cv2

# 가장 기초적인 파이프라인 (해상도, FPS 자동)
pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

print(f"GStreamer 지원 여부: {cv2.getBuildInformation().find('GStreamer: YES') != -1}")
print("카메라 여는 중...")

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("✅ 카메라 성공! (창이 뜹니다)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC로 종료
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("❌ 실패... (Daemon 재시작 하셨나요?)")