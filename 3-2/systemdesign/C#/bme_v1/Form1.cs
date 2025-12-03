using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO.Ports; //C#에서 시리얼 통신 위해 반드시 추가해야 하는 네임스페이스
using System.Net.Sockets; // TCP 통신을 위해 필수!

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        SerialPort sPort;
        int[] data_buff = new int[200];
        static int buffsize = 2000;
        double[] input_Data_1 = new double[buffsize];
        double[] input_Data_2 = new double[buffsize];
        double[] input_Data_3 = new double[buffsize];
        public double[] input_Draw_1 = new double[buffsize];
        public double[] input_Draw_2 = new double[buffsize];
        public double[] input_Draw_3 = new double[buffsize];

        int start_flag = 0;
        int data_count = 0;
        int Data_1, Data_2, Data_3;

        // === [2. 캘리브레이션 관련 변수 (수정됨)] ===
        // 현재 캘리브레이션이 어떤 상태인지 저장 (None이면 평상시)
        CalibState currentCalibState = CalibState.None;

        // 0.5초 동안 데이터를 잠시 모아둘 리스트
        List<double> tempCalibBuffer = new List<double>();

        // 각 지점별 측정된 평균값 저장
        double val_Center, val_Right, val_Left, val_Up, val_Down;

        // 최종 계산될 각도 변환 상수 (K)
        double K_Horz = 0; // 수평 각도 상수
        double K_Vert = 0; // 수직 각도 상수


        private EOGprocess eogFilter = new EOGprocess(); //EOG 필터링 객체 불러오기
        private EOGprocess eogFilter2 = new EOGprocess();

        string thisdate = DateTime.Now.ToString("yyMMdd");

        // [TCP 통신용 변수 추가]
        private TcpClient client;
        private NetworkStream stream;
        private bool isConnected = false;

        // Gaze 계산 객체
        private GazeCalculator gazeCalc = new GazeCalculator();


        public Form1()
        {
            InitializeComponent();
        }

        private void maskedTextBox1_MaskInputRejected(object sender, MaskInputRejectedEventArgs e)
        {

        }

        private void groupBox1_Enter(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            if (null != sPort)
            {
                if (sPort.IsOpen)
                {
                    sPort.Close();
                    sPort.Dispose();
                    sPort = null;
                }
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            btnOpen.Enabled = true;
            btnClose.Enabled = false;

            cboPortName.BeginUpdate();
            foreach (string comport in SerialPort.GetPortNames())
            {
                cboPortName.Items.Add(comport);
            }
            cboPortName.EndUpdate();

            //포트 자동 선택(없으면 기본값)
            cboPortName.SelectedItem = "COM4";
            txtBaudRate.Text = "115200";
            CheckForIllegalCrossThreadCalls = false;
            txtDate.Text = thisdate;

            ConnectToPythonServer(); // 파이썬 서버 연결 시도
        }

        private void BtnOpen_Click(object sender, EventArgs e)
        {
            try
            {
                if (null == sPort)
                {
                    sPort = new SerialPort();
                    sPort.DataReceived += new SerialDataReceivedEventHandler(SPort_DataReceived);

                    sPort.PortName = cboPortName.SelectedItem.ToString();
                    sPort.BaudRate = Convert.ToInt32(txtBaudRate.Text);
                    sPort.DataBits = (int)8;
                    sPort.Parity = Parity.None;
                    sPort.StopBits = StopBits.One;
                    sPort.Open();
                }

                if (sPort.IsOpen)
                {
                    // 1. 캘리브레이션 변수 초기화
                    K_Horz = 0; K_Vert = 0;
                    currentCalibState = CalibState.None;
                    val_Center = 0; val_Right = 0; val_Left = 0; val_Up = 0; val_Down = 0;

                    // 2. 캘리브레이션 창 생성
                    calibration calibForm = new calibration();

                    // [이벤트 연결 1] 지휘자(calib)가 "측정 시작해!"라고 할 때
                    calibForm.OnMeasureStart += (state) =>
                    {
                        if (state == CalibState.Finish)
                        {
                            // 모든 과정 종료 -> 상수 계산
                            CalculateConstants();
                        }
                        else
                        {
                            // 상태 변경 및 버퍼 비우기 (데이터 수집 준비)
                            currentCalibState = state;
                            tempCalibBuffer.Clear();
                        }
                    };

                    // [이벤트 연결 2] 지휘자(calib)가 "측정 멈춰!"라고 할 때
                    calibForm.OnMeasureStop += () =>
                    {
                        if (tempCalibBuffer.Count > 0)
                        {
                            double avg = tempCalibBuffer.Average(); // 모인 데이터 평균

                            // 현재 상태에 맞는 변수에 저장
                            switch (currentCalibState)
                            {
                                case CalibState.Center: val_Center = avg; break;
                                case CalibState.Right: val_Right = avg; break;
                                case CalibState.Left: val_Left = avg; break;
                                case CalibState.Up: val_Up = avg; break;
                                case CalibState.Down: val_Down = avg; break;
                            }
                        }
                        currentCalibState = CalibState.None; // 측정 대기 상태로 복귀
                    };

                    // 3. 캘리브레이션 창 띄우기 (21초 시퀀스 시작)
                    // ShowDialog는 창이 닫힐 때까지 코드 실행을 여기서 멈춥니다.
                    calibForm.ShowDialog();

                    // 4. 완료 후 버튼 상태 변경
                    btnOpen.Enabled = false;
                    btnClose.Enabled = true;
                }
                else
                {
                    btnOpen.Enabled = true;
                    btnClose.Enabled = false;
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void BtnClose_Click(object sender, EventArgs e)
        {
            if (null != sPort)
            {
                if (sPort.IsOpen)
                {
                    sPort.Close();
                    sPort.Dispose();
                    sPort = null;
                }
            }
            btnOpen.Enabled = true;
            btnClose.Enabled = false;
        }

        private void scope1_Click(object sender, EventArgs e)
        {

        }

        private void SPort_DataReceived(object sender, System.IO.Ports.SerialDataReceivedEventArgs e)
        {
            while (sPort.BytesToRead > 0)
            {
                if (!sPort.IsOpen) return; // 포트가 닫혔으면 아무것도 하지 않음

                int currentByte = sPort.ReadByte();
                if (currentByte < 0) break; // 읽기 실패 시 중단

                // 1. start_flag가 0이면 시작 바이트(0x81)를 기다립니다.
                if (start_flag == 0)
                {
                    if (currentByte == 0x81)
                    {
                        start_flag = 1; // 시작 바이트를 찾았음
                        data_count = 0; // 데이터 카운터 초기화
                    }
                }
                // 2. 시작 바이트를 찾았다면 (start_flag == 1)
                else
                {
                    data_buff[data_count] = currentByte; // 버퍼에 데이터 저장
                    data_count++;

                    // MSP430은 0x81 뒤에 총 6바이트 (Data_Hi, Data_Lo, 0, 0, 0, 0)를 보냅니다.
                    // 6바이트가 모두 수신되었는지 확인합니다.
                    if (data_count == 6)
                    {
                        // MSP430이 보낸 2바이트를 조합하여 Data_1을 만듭니다.
                        // data_buff[0] = Packet[1] = (adc1 >> 7) & 0x7F
                        // data_buff[1] = Packet[2] = adc1 & 0x7F
                        Data_1 = ((data_buff[0] & 0x7F) << 7) + (data_buff[1] & 0x7F);
                        Data_2 = ((data_buff[2] & 0x7F) << 7) + (data_buff[3] & 0x7F);
                        Data_3 = 0;

                        double filteredData = eogFilter.ProcessSample(Data_1);
                        double filteredData2 = eogFilter2.ProcessSample(Data_2);

                        // --- [분기점] 캘리브레이션 중 vs 평상시 ---

                        if (currentCalibState != CalibState.None)
                        {
                            // [캘리브레이션 모드] 그래프 안 그림, 버퍼에 저장
                            if (currentCalibState == CalibState.Right || currentCalibState == CalibState.Left)
                            {
                                tempCalibBuffer.Add(filteredData); // 수평 데이터 수집
                            }
                            else if (currentCalibState == CalibState.Up || currentCalibState == CalibState.Down)
                            {
                                tempCalibBuffer.Add(filteredData2); // 수직 데이터 수집
                            }
                            else if (currentCalibState == CalibState.Center)
                            {
                                // 중앙일 때는 수평값 기준으로 수집 (필요 시 수직도 별도 처리 가능)
                                tempCalibBuffer.Add(filteredData);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < buffsize - 1; i++)
                            {
                                input_Data_1[i] = input_Data_1[i + 1];
                                input_Data_2[i] = input_Data_2[i + 1];
                            }
                            input_Data_1[buffsize - 1] = filteredData;
                            input_Data_2[buffsize - 1] = filteredData2;

                            input_Draw_1 = input_Data_1; // 차트에 그릴 데이터 업데이트
                            input_Draw_2 = input_Data_2;

                            //---TCP 코드 추가됨---
                            // 1. 연결되어 있고, 캘리브레이션이 완료된 상태라면?
                            if (isConnected && stream != null && gazeCalc.IsCalibrated)
                            {
                                try
                                {
                                    // 2. GazeCalculator를 통해 각도 계산 (특허 알고리즘 적용됨)
                                    double angleH, angleV;
                                    gazeCalc.CalculateGaze(filteredData, filteredData2, out angleH, out angleV);

                                    // [선택 A] 각도(Angle)를 그대로 보내고 싶을 때
                                    string msg = $"{angleH:F2},{angleV:F2}\n";

                                    // [선택 B] 화면 픽셀 좌표(Pixel)로 변환해서 보내고 싶을 때 (추천)
                                    // (파이썬 OpenCV 창 크기가 640x480이라고 가정)
                                    // System.Drawing.Point pixel = gazeCalc.GetCameraCoordinates(angleH, angleV, 640, 480);
                                    // string msg = $"{pixel.X},{pixel.Y}\n";

                                    // 3. TCP 전송
                                    byte[] dataToSend = Encoding.UTF8.GetBytes(msg);
                                    stream.Write(dataToSend, 0, dataToSend.Length);
                                }
                                catch
                                {
                                    isConnected = false; // 에러나면 연결 끊김 처리
                                }
                            }
                        }
                        // 다음 패킷을 받기 위해 플래그와 카운터 초기화
                        start_flag = 0;
                        data_count = 0;
                    }
                }
            }
        }


        private void On_timer1(object sender, EventArgs e)
        {
            scope1.Channels[0].Data.SetYData(input_Data_1);
            if (scope1.Channels.Count > 1)
            {
                scope1.Channels[1].Data.SetYData(input_Data_2);
            }
        }
        // === 파이썬 서버 연결 함수 ===
        private void ConnectToPythonServer()
        {
            try
            {
                // 1. 아까 만든 변수(client)에 실제 연결 객체를 생성해서 저장(할당)합니다.
                // "내 컴퓨터(127.0.0.1)의 5000번 포트로 연결해라"
                client = new TcpClient("192.168.165.70", 5000);

                // 2. 아까 만든 변수(stream)에 데이터 통로를 저장합니다.
                stream = client.GetStream();

                // 3. 연결 성공 깃발을 듭니다.
                isConnected = true;

                MessageBox.Show("Python 서버와 연결 성공!");
            }
            catch (Exception ex)
            {
                // 실패하면 변수들은 null 상태거나 연결 안 됨 상태로 남습니다.
                isConnected = false;
                // 파이썬이 안 켜져 있으면 에러가 나므로, 테스트할 땐 파이썬을 먼저 켜야 합니다.
                // MessageBox.Show("연결 실패: " + ex.Message); 
            }
        }

        // === [6. 캘리브레이션 결과 계산 함수 (추가됨)] ===
        private void CalculateConstants()
        {
            // 1. 계산기에 측정된 5점 데이터 입력 (이때 내부적으로 상수 K, 진폭 등이 계산됨)
            // 파라미터 순서: Center, Right, Left, Center(Vertical), Up, Down
            // (GazeCalculator의 SetCalibrationData 함수 정의에 맞춰서 넣습니다)
            // 보통 수평/수직 중앙값이 같으므로 val_Center를 두 번 넣습니다.
            gazeCalc.SetCalibrationData(val_Center, val_Right, val_Left, val_Center, val_Up, val_Down);

            // 2. 메시지 박스는 확인용으로 띄우되, 이게 뜨면 코드가 잠시 멈출 수 있으니 
            // 실제 사용 시에는 주석 처리하는 게 좋습니다.
            // MessageBox.Show("캘리브레이션 완료! 이제 시선 추적을 시작합니다.");
            
            // (중요) 콘솔이나 디버그 창에만 살짝 남기기
            System.Diagnostics.Debug.WriteLine("Calibration Finished & Set to GazeCalculator");
        }
    }
}
