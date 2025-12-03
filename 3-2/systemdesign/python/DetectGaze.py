import cv2
import threading
import time
import random
import queue
import socket
import os
from ultralytics import YOLO

# [NEW] gTTS 및 오디오 재생용 라이브러리
try:
    from gtts import gTTS
    import pygame
except ImportError:
    print("❌ 필수 라이브러리가 없습니다. 설치해주세요: pip install gTTS pygame")

# --- [설정값] 사용자 환경에 맞춰 조절하세요 ---
HOST = '0.0.0.0'
PORT = 5000

# 카메라 해상도
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ★ [중요] 각도 -> 픽셀 변환 감도 (Gain)
PIXELS_PER_DEGREE_X = 18.0 
PIXELS_PER_DEGREE_Y = 18.0

# YOLO 설정
CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)

# 라벨 오버레이 상태
LAST_LABEL_MSG = None
LAST_LABEL_TIME = 0.0
LABEL_MSG_DURATION = 1.5

# TTS 설정
# gTTS는 인터넷 통신을 하므로 쿨다운을 좀 더 넉넉히 주는 게 좋습니다.
TTS_COOLDOWN = 3.0  
tts_queue = queue.Queue()
last_spoken = {"msg": None, "t": 0.0}

# 한글 라벨 매핑
KO_LABELS = {
    "person":"사람", "car":"자동차", "bus":"버스", "truck":"트럭", "bicycle":"자전거",
    "motorcycle":"오토바이", "dog":"강아지", "cat":"고양이", "chair":"의자",
    "bottle":"병", "cup":"컵", "cell phone":"휴대폰", "laptop":"노트북", "book":"책",
    "keyboard":"키보드", "mouse":"마우스", "monitor":"모니터", "tv":"텔레비전"
}

def to_korean(name: str) -> str:
    return KO_LABELS.get(name, name)

def label_at_gaze(yolo_results, gx, gy, class_names):
    """ 시선점(gx, gy)이 포함된 박스 중 신뢰도가 가장 높은 라벨 반환 """
    if yolo_results is None:
        return None, None

    best = None
    best_conf = -1.0

    for data in yolo_results.boxes.data.tolist():
        xmin, ymin, xmax, ymax = map(int, data[:4])
        conf = float(data[4]); cls_id = int(data[5])
        
        margin = 10
        if (xmin - margin <= gx <= xmax + margin) and (ymin - margin <= gy <= ymax + margin):
            if conf > best_conf:
                best_conf = conf
                best = class_names[cls_id]

    return (best, best_conf) if best is not None else (None, None)

# --- 공유 변수 및 Lock ---
latest_frame = None
latest_yolo_results = None
latest_eog_gaze_data = {'x': CAM_WIDTH // 2, 'y': CAM_HEIGHT // 2}

frame_lock = threading.Lock()
yolo_results_lock = threading.Lock()
eog_gaze_lock = threading.Lock()

is_running = True

# --- 1. [수정됨] gTTS 스레드 ---
def tts_thread_func():
    # pygame 믹서 초기화 (오디오 재생용)
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"[오디오 초기화 실패] {e}")
        return

    print("🎤 Google TTS 시스템 준비 완료")

    while is_running:
        try:
            # 큐에서 메시지 대기
            msg = tts_queue.get(timeout=0.1)
            
            if msg:
                print(f"🔊 생성 중... : {msg}")
                
                # 1. 구글 서버에서 음성 파일 생성 (lang='ko' 한국어)
                tts = gTTS(text=msg, lang='ko', slow=False)
                
                # 2. 임시 파일로 저장
                filename = "temp_voice.mp3"
                tts.save(filename)
                
                # 3. 재생
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                
                # 4. 재생이 끝날 때까지 대기
                while pygame.mixer.music.get_busy() and is_running:
                    time.sleep(0.1)
                
                # 5. 파일 연결 해제 (삭제를 위해)
                pygame.mixer.music.unload()
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS 오류] {e}")
            # 인터넷 연결 문제일 수 있음

# --- 2. YOLO 스레드 ---
def yolo_thread_func(yolo_model):
    global latest_frame, latest_yolo_results, is_running
    print("YOLO 모델 가동 중...")

    while is_running:
        current_frame = None
        with frame_lock:
            if latest_frame is not None:
                current_frame = latest_frame.copy()

        if current_frame is not None:
            results = yolo_model(current_frame, imgsz=320, conf=CONFIDENCE_THRESHOLD, verbose=False)
            with yolo_results_lock:
                latest_yolo_results = results[0]
        
        time.sleep(0.01)

# --- 3. TCP 시선 데이터 수신 스레드 ---
def tcp_eog_thread_func():
    global latest_eog_gaze_data, is_running
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"🚀 시선 데이터 서버 대기중... ({HOST}:{PORT})")
        
        client_socket, addr = server_socket.accept()
        print(f"✅ C# 프로그램 연결됨: {addr}")
        
        buffer = ""
        
        while is_running:
            try:
                data = client_socket.recv(1024)
                if not data: break
                
                buffer += data.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line: continue
                    
                    if ',' in line:
                        try:
                            parts = line.split(',')
                            angle_x = float(parts[0])
                            angle_y = float(parts[1])
                            
                            px = int(CAM_WIDTH / 2 + (angle_x * PIXELS_PER_DEGREE_X))
                            py = int(CAM_HEIGHT / 2 - (angle_y * PIXELS_PER_DEGREE_Y))
                            
                            px = max(0, min(CAM_WIDTH, px))
                            py = max(0, min(CAM_HEIGHT, py))
                            
                            with eog_gaze_lock:
                                latest_eog_gaze_data = {'x': px, 'y': py}
                                
                        except ValueError:
                            pass
            except Exception as e:
                print(f"데이터 수신 중 오류: {e}")
                break
                
    except Exception as e:
        print(f"TCP 서버 오류: {e}")
    finally:
        try:
            server_socket.close()
        except: pass
        print("TCP 서버 종료")

# --- 메인 스레드 ---
if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    t1 = threading.Thread(target=yolo_thread_func, args=(model,))
    t2 = threading.Thread(target=tcp_eog_thread_func)
    t3 = threading.Thread(target=tts_thread_func, daemon=True)

    t1.start(); t2.start(); t3.start()
    
    print("\n" + "="*40)
    print(" [시스템 시작] C# 연결 후 캘리브레이션을 진행하세요.")
    print(" - 'g' 키: 현재 시선에 있는 물체 읽기 (gTTS)")
    print(" - 'q' 키: 프로그램 종료")
    print("="*40 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            with frame_lock:
                latest_frame = frame.copy()

            # 1. YOLO 결과
            current_results = None
            with yolo_results_lock:
                if latest_yolo_results:
                    current_results = latest_yolo_results

            if current_results:
                for data in current_results.boxes.data.tolist():
                    conf = float(data[4])
                    if conf < CONFIDENCE_THRESHOLD: continue
                    
                    xmin, ymin, xmax, ymax = map(int, data[:4])
                    cls_id = int(data[5])
                    label = to_korean(model.names[cls_id])
                    
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (xmin, ymin-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

            # 2. 시선(EOG) 그리기
            with eog_gaze_lock:
                gx, gy = latest_eog_gaze_data['x'], latest_eog_gaze_data['y']
            
            cv2.line(frame, (gx-10, gy), (gx+10, gy), RED, 2)
            cv2.line(frame, (gx, gy-10), (gx, gy+10), RED, 2)
            cv2.circle(frame, (gx, gy), 8, RED, 2)

            # 3. 상태 메시지
            if LAST_LABEL_MSG and (time.time() - LAST_LABEL_TIME) < LABEL_MSG_DURATION:
                cv2.putText(frame, LAST_LABEL_MSG, (20, 50), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow('Eye Tracking AI', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                name, conf = label_at_gaze(current_results, gx, gy, model.names)
                if name:
                    k_name = to_korean(name)
                    msg = f"시선 감지: {k_name}"
                    
                    now = time.time()
                    if k_name != last_spoken["msg"] or (now - last_spoken["t"]) > TTS_COOLDOWN:
                        tts_queue.put(k_name) 
                        last_spoken = {"msg": k_name, "t": now}
                else:
                    msg = "감지된 물체 없음"
                
                print(f"👀 {msg}")
                LAST_LABEL_MSG = msg
                LAST_LABEL_TIME = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        is_running = False
        cap.release()
        cv2.destroyAllWindows()
        # 임시 파일 삭제 시도
        try:
            if os.path.exists("temp_voice.mp3"):
                os.remove("temp_voice.mp3")
        except: pass
        print("프로그램 종료.")