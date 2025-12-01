import cv2
import threading
import time
import random
import queue                                    # [NEW]
import pyttsx3                                  # [NEW]
from ultralytics import YOLO

# --- 설정값 ---
CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# 라벨 오버레이 상태
LAST_LABEL_MSG = None
LAST_LABEL_TIME = 0.0
LABEL_MSG_DURATION = 1.5  # 초

# [NEW] TTS 설정
TTS_RATE = 180
TTS_COOLDOWN = 0.8  # 같은 문구 연속 재생 방지(초)
tts_queue = queue.Queue()
last_spoken = {"msg": None, "t": 0.0}

# [NEW] 일부 COCO 라벨 한글 매핑(원하면 추가)
KO_LABELS = {
    "person":"사람","car":"자동차","bus":"버스","truck":"트럭","bicycle":"자전거",
    "motorcycle":"오토바이","dog":"강아지","cat":"고양이","chair":"의자",
    "bottle":"병","cup":"컵","cell phone":"휴대폰","laptop":"노트북","book":"책"
}
def to_korean(name: str) -> str:
    return KO_LABELS.get(name, name)

def label_at_gaze(yolo_results, gx, gy, class_names):
    """
    시선점(gx, gy)이 포함된 박스 중 '신뢰도'가 가장 높은 라벨을 반환.
    없으면 (None, None).
    """
    if yolo_results is None:
        return None, None

    best = None
    best_conf = -1.0

    for data in yolo_results.boxes.data.tolist():  # [xmin,ymin,xmax,ymax,conf,cls]
        xmin, ymin, xmax, ymax = map(int, data[:4])
        conf = float(data[4]); cls_id = int(data[5])
        if (xmin <= gx <= xmax) and (ymin <= gy <= ymax):
            if conf > best_conf:
                best_conf = conf
                best = class_names[cls_id]

    return (best, best_conf) if best is not None else (None, None)

# --- 공유 변수 및 Lock 설정 ---
latest_frame = None
latest_yolo_results = None
latest_eog_gaze_data = {'x': 0, 'y': 0}

frame_lock = threading.Lock()
yolo_results_lock = threading.Lock()
eog_gaze_lock = threading.Lock()

is_running = True

# --- [NEW] TTS 전용 스레드 ---
def tts_thread_func():
    engine = pyttsx3.init()
    # 한국어 보이스 자동 선택(있으면)
    try:
        voices = engine.getProperty('voices')
        ko_id = None
        for v in voices:
            name = getattr(v, 'name', '').lower()
            langs = [str(l).lower() for l in getattr(v, 'languages', [])] if hasattr(v,'languages') else []
            if 'korean' in name or 'ko' in name or any('ko' in l for l in langs) or '한국' in name:
                ko_id = v.id; break
        if ko_id: engine.setProperty('voice', ko_id)
    except Exception as e:
        print("[TTS] 보이스 선택 오류:", e)
    engine.setProperty('rate', TTS_RATE)

    while is_running or not tts_queue.empty():
        try:
            msg = tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            engine.say(msg)
            engine.runAndWait()
        except Exception as e:
            print("[TTS] 재생 오류:", e)

# --- 1. YOLO 객체 감지 스레드 ---
def yolo_thread_func(yolo_model):
    global latest_frame, latest_yolo_results, is_running
    print("YOLO 스레드 시작.")

    while is_running:
        current_frame = None
        with frame_lock:
            if latest_frame is not None:
                current_frame = latest_frame.copy()

        if current_frame is not None:
            results = yolo_model(current_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            with yolo_results_lock:
                latest_yolo_results = results[0]
        
        time.sleep(0.01)
    print("YOLO 스레드 종료")

# --- 2. 가상 EOG 데이터 생성 스레드 ---
def eog_data_thread_func():
    global latest_eog_gaze_data, is_running
    gaze_x, gaze_y = 320, 240
    while is_running:
        gaze_x = max(0, min(639, gaze_x + random.randint(-10, 10)))
        gaze_y = max(0, min(479, gaze_y + random.randint(-10, 10)))
        with eog_gaze_lock:
            latest_eog_gaze_data = {'x': gaze_x, 'y': gaze_y}
        time.sleep(0.02)
    print("EOG 데이터 스레드 종료")

# --- 메인 스레드 (카메라, UI, 융합) ---
if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    print("YOLOv8n 모델 로드 완료.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 창 포커스 편의(선택)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)

    # 스레드 시작
    yolo_t = threading.Thread(target=yolo_thread_func, args=(model,))
    eog_t  = threading.Thread(target=eog_data_thread_func)
    tts_t  = threading.Thread(target=tts_thread_func, daemon=True)   # [NEW] 데몬 스레드

    yolo_t.start(); eog_t.start(); tts_t.start()
    print("메인 루프 시작. 'q' 종료, 'g' 시선 라벨 TTS.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            with frame_lock:
                latest_frame = frame.copy()

            # YOLO 결과 스냅샷
            current_yolo_results = None
            with yolo_results_lock:
                if latest_yolo_results is not None:
                    current_yolo_results = latest_yolo_results

            # 박스 렌더링
            if current_yolo_results:
                for data in current_yolo_results.boxes.data.tolist():
                    confidence = float(data[4])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    xmin, ymin, xmax, ymax = map(int, data[:4])
                    cls_id = int(data[5])
                    class_name = model.names[cls_id]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}",
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

            # 시선 렌더링
            with eog_gaze_lock:
                gaze_x, gaze_y = latest_eog_gaze_data['x'], latest_eog_gaze_data['y']
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)

            # 최근 라벨 메시지 오버레이
            if LAST_LABEL_MSG and (time.time() - LAST_LABEL_TIME) < LABEL_MSG_DURATION:
                cv2.putText(frame, LAST_LABEL_MSG, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('frame', frame)

            # 키 처리
            key = cv2.waitKeyEx(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                name, conf = label_at_gaze(current_yolo_results, gaze_x, gaze_y, model.names)
                if name is not None:
                    say = to_korean(name)
                    msg = f"[GAZE] {say} {conf:.2f}"
                else:
                    say = "대상 없음"
                    msg = "[GAZE] (no object)"

                # 화면 표시
                LAST_LABEL_MSG = msg
                LAST_LABEL_TIME = time.time()
                print(msg)

                # [NEW] TTS 큐로 전송 (중복/쿨다운 방지)
                now = time.time()
                if say != last_spoken["msg"] or (now - last_spoken["t"]) > TTS_COOLDOWN:
                    tts_queue.put(say)
                    last_spoken["msg"] = say
                    last_spoken["t"] = now

    except KeyboardInterrupt:
        print("종료 신호 감지...")
    finally:
        is_running = False
        yolo_t.join(); eog_t.join()
        cap.release()
        cv2.destroyAllWindows()
        print("모든 리소스 정리 및 프로그램 종료.")
