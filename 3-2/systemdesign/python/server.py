import socket

# 1. 서버 설정 (C#과 똑같은 주소와 포트여야 합니다)
HOST = '127.0.0.1'  # 내 컴퓨터 (Localhost)
PORT = 5000         # 포트 번호

def start_server():
    # 소켓 생성 (IPv4, TCP)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 주소 재사용 설정 (끄고 켤 때 에러 방지)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 바인딩 및 대기
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1) # 최대 1명 접속 허용
        print(f"🚀 Python 서버 대기중... ({HOST}:{PORT})")
        print("이제 C# 프로그램을 실행해주세요!")
        
        # 연결 수락 (여기서 C#이 접속할 때까지 멈춰있음)
        client_socket, addr = server_socket.accept()
        print(f"✅ 연결 성공! C# 주소: {addr}")
        
        while True:
            # 데이터 수신 (최대 1024바이트)
            data = client_socket.recv(1024)
            
            # 데이터가 없으면 연결 끊김
            if not data:
                print("❌ C# 프로그램과 연결이 끊어졌습니다.")
                break
                
            # 디코딩 (byte -> string)
            # 예: "10.5, -5.2\n"
            text = data.decode('utf-8').strip()
            
            if text:
                try:
                    # 콤마로 분리해서 좌표 확인
                    x_str, y_str = text.split(',')
                    x = float(x_str)
                    y = float(y_str)
                    
                    print(f"👀 시선 좌표 수신 -> X: {x:.2f}, Y: {y:.2f}")
                    
                    # [나중에 할 일] 여기서 마우스 커서를 움직이거나 로봇에게 명령을 보냄
                    
                except ValueError:
                    # 데이터가 깨져서 올 경우 무시
                    pass
                    
    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_server()