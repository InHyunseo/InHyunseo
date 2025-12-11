using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO.Ports;
using System.Net.Sockets;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        bool isCalibrated = false;

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

        // === [1. 워밍업 관련 변수 (초기 튀는 현상 방지)] ===
        int stable_count = 0;
        const int WARMUP_SAMPLES = 200; // 처음 200개 데이터는 버림

        // === [2. 캘리브레이션 관련 변수] ===
        CalibState currentCalibState = CalibState.None;

        // ★★★ [수정 핵심] 버퍼와 중앙값을 가로/세로 분리! ★★★
        List<double> tempCalibBuffer_H = new List<double>();
        List<double> tempCalibBuffer_V = new List<double>();

        double val_Center_H, val_Center_V; // 중앙값도 H/V 따로 저장
        double val_Right, val_Left, val_Up, val_Down;

        // 최종 계산될 각도 변환 상수 (K)
        double K_Horz = 0;
        double K_Vert = 0;

        private EOGprocess eogFilter = new EOGprocess();
        private EOGprocess eogFilter2 = new EOGprocess();

        string thisdate = DateTime.Now.ToString("yyMMdd");

        // [TCP 통신용 변수]
        private TcpClient client;
        private NetworkStream stream;
        private bool isConnected = false;

        // =========================================================
        // [새로 추가] EOG 지능형 알고리즘 변수 (Delay & Floating Center)
        // =========================================================
        
        // 1. 버퍼 (0.4초 미래 데이터 확보용)
        const int BUFF_SIZE = 25; 
        List<double> smartBuff_H = new List<double>();
        List<double> smartBuff_V = new List<double>();

        // 2. 화면 기준점 (Floating Center)
        // 기존에는 320, 240 고정이었으나, 이제는 눈이 머무는 곳으로 계속 바뀝니다.
        double screen_center_H = 320.0;
        double screen_center_V = 240.0;

        // 3. 튜닝 파라미터 (블링크 및 응시 판단 기준)
        const double THRESH_LEVEL = 40.0;        // 유의미한 이동 감지 레벨
        const double THRESH_BLINK_SLOPE = -3.5;  // 블링크 급하강 기울기
        const double THRESH_FIXATION_SLOPE = 1.5;// 응시(드리프트) 완만 기울기

        // 4. 상태 관리 변수
        int ignoreTimer = 0;      // 블링크 무시 타이머
        double holdPixel_H = 320; // 고정된 좌표 저장용
        double holdPixel_V = 240;
        // =========================================================



        public Form1()
        {
            InitializeComponent();
            eogFilter.SetGain(1.0);  
            eogFilter2.SetGain(5.0);
        }

        private void maskedTextBox1_MaskInputRejected(object sender, MaskInputRejectedEventArgs e) { }
        private void groupBox1_Enter(object sender, EventArgs e) { }
        private void label1_Click(object sender, EventArgs e) { }

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

            cboPortName.SelectedItem = "COM4";
            txtBaudRate.Text = "115200";
            CheckForIllegalCrossThreadCalls = false;
            txtDate.Text = thisdate;

            ConnectToPythonServer();
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
                    // 1. 변수 초기화
                    K_Horz = 0; K_Vert = 0;
                    currentCalibState = CalibState.None;

                    // 값 초기화 (H, V 분리됨)
                    val_Center_H = 0; val_Center_V = 0;
                    val_Right = 0; val_Left = 0; val_Up = 0; val_Down = 0;
                    stable_count = 0; // 워밍업 카운트 리셋

                    // 2. 캘리브레이션 창 생성
                    calibration calibForm = new calibration();

                    calibForm.OnMeasureStart += (state) =>
                    {
                        if (state == CalibState.Finish)
                        {
                            CalculateConstants();
                        }
                        else
                        {
                            currentCalibState = state;
                            // 버퍼 2개 모두 비우기
                            tempCalibBuffer_H.Clear();
                            tempCalibBuffer_V.Clear();

                            // [추가] Center 상태가 시작될 때 로그 출력
                            if (state == CalibState.Center)
                            {
                                System.Diagnostics.Debug.WriteLine("--- Calib: Center Start ---");
                            }
                        }
                    };

                    calibForm.OnMeasureStop += () =>
                    {
                        // 데이터가 충분히 모였는지 확인
                        if (tempCalibBuffer_H.Count > 0 && tempCalibBuffer_V.Count > 0)
                        {
                            switch (currentCalibState)
                            {
                                case CalibState.Center:
                                    // 1. Center: 수집된 데이터의 평균(Average)을 저장
                                    val_Center_H = tempCalibBuffer_H.Average();
                                    val_Center_V = tempCalibBuffer_V.Average();
                                    break;

                                case CalibState.Up:
                                    // 위로 갈 때: 수직 신호의 최대값(Peak) 저장
                                    val_Up = tempCalibBuffer_V.Max();
                                    break;

                                case CalibState.Down:
                                    // 아래로 갈 때: 수직 신호의 최소값(Valley) 저장
                                    val_Down = tempCalibBuffer_V.Min();
                                    break;

                                case CalibState.Right:
                                    // 오른쪽 갈 때: 수평 신호의 최대값 저장
                                    val_Right = tempCalibBuffer_H.Max();
                                    break;

                                case CalibState.Left:
                                    // 왼쪽 갈 때: 수평 신호의 최소값 저장
                                    val_Left = tempCalibBuffer_H.Min();
                                    break;
                            }
                        }
                        currentCalibState = CalibState.None;
                    };
                   
                    calibForm.ShowDialog();

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

        private void scope2_Click(object sender, EventArgs e)
        { }

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

        private void scope1_Click(object sender, EventArgs e) { }

        private void SPort_DataReceived(object sender, System.IO.Ports.SerialDataReceivedEventArgs e)
        {
            while (sPort.BytesToRead > 0)
            {
                if (!sPort.IsOpen) return;

                int currentByte = sPort.ReadByte();
                if (currentByte < 0) break;

                if (start_flag == 0)
                {
                    if (currentByte == 0x81)
                    {
                        start_flag = 1;
                        data_count = 0;
                    }
                }
                else
                {
                    data_buff[data_count] = currentByte;
                    data_count++;

                    if (data_count == 6)
                    {
                        Data_1 = ((data_buff[0] & 0x7F) << 7) + (data_buff[1] & 0x7F);
                        Data_2 = ((data_buff[2] & 0x7F) << 7) + (data_buff[3] & 0x7F);
                        Data_3 = 0;

                        double filteredData = eogFilter.ProcessSample(Data_1);
                        double filteredData2 = eogFilter2.ProcessSample(Data_2);

                        // === [워밍업] 초반 데이터 200개 버리기 (그래프 튀는 현상 방지) ===
                        if (stable_count < WARMUP_SAMPLES)
                        {
                            stable_count++;
                            start_flag = 0; data_count = 0;
                            continue; // 여기서 함수 끝냄 (그래프 안 그림)
                        }

                        
                        // ★★★ [수정] 그래프 그리기를 조건문 밖으로 뺐습니다! ★★★
                        for (int i = 0; i < buffsize - 1; i++)
                        {
                            input_Data_1[i] = input_Data_1[i + 1];
                            input_Data_2[i] = input_Data_2[i + 1];
                        }
                        input_Data_1[buffsize - 1] = filteredData;
                        input_Data_2[buffsize - 1] = filteredData2;

                        input_Draw_1 = input_Data_1;
                        input_Draw_2 = input_Data_2;
                        

                        // --- [분기점] 캘리브레이션 중 vs 평상시 ---
                        if (currentCalibState != CalibState.None)
                        {
                            // [캘리브레이션 모드]
                            // ★ 어떤 상태이든 H, V 데이터 둘 다 모아야 나중에 계산 가능
                            tempCalibBuffer_H.Add(filteredData);
                            tempCalibBuffer_V.Add(filteredData2);
                        }
                        else
                        {
                            // [평상시 모드] TCP 전송 및 좌표 변환 로직
                            if (isConnected && stream != null)
                            {
                                try
                                {
                                    // 1. 각도 계산 (기존 로직)
                                    double angleX = (filteredData - val_Center_H) * K_Horz;
                                    double angleY = (filteredData2 - val_Center_V) * K_Vert;

                                    // 2. 드리프트 자동 보정 (중요!)
                                    // 시선이 한쪽으로 쏠려 있으면, 중앙점(Center)을 아주 천천히 그쪽으로 이동시킵니다.
                                    // EOG 신호가 시간이 지나면 흘러내리는 현상을 막아줍니다.
                                    val_Center_H += (filteredData - val_Center_H) * 0.001;
                                    val_Center_V += (filteredData2 - val_Center_V) * 0.001;

                                    // 3. 각도 -> 픽셀 변환 (감도 조절)
                                    // sensitivity 값을 키우면 눈을 조금만 움직여도 화면 끝까지 갑니다.
                                    double sensitivity_H = 25.0;
                                    double sensitivity_V = 100.0;// (기존보다 더 높임: 18.0 -> 25.0)

                                    // 화면 중앙(320, 240)을 기준으로 더하고 뺍니다.
                                    int pixelX = 320 + (int)(angleX * sensitivity_H);
                                    int pixelY = 240 - (int)(angleY * sensitivity_V); // Y축은 위가 (-)라서 뺌

                                    // 4. 화면 밖으로 나가지 않게 가두기 (Clamping)
                                    pixelX = Math.Max(0, Math.Min(640, pixelX));
                                    pixelY = Math.Max(0, Math.Min(480, pixelY));

                                    // 5. 전송 (좌표값)
                                    string msg = $"{pixelX},{pixelY}\n";
                                    byte[] dataToSend = Encoding.UTF8.GetBytes(msg);
                                    stream.Write(dataToSend, 0, dataToSend.Length);

                                    // [디버깅] 값이 잘 나오는지 출력창에서 확인해보세요
                                    System.Diagnostics.Debug.WriteLine($"Send: {pixelX}, {pixelY}");
                                }
                                catch
                                {
                                    isConnected = false;
                                }
                            }
                        }

                        start_flag = 0;
                        data_count = 0;
                    }
                }
            }
        }

        private void On_timer1(object sender, EventArgs e)
        {
            if (scope1.Channels.Count >0)
            {
                scope1.Channels[0].Data.SetYData(input_Data_1);
            }
            
            if (scope2.Channels.Count > 0)
            {
                scope2.Channels[0].Data.SetYData(input_Data_2);
            }
        }

        private void ConnectToPythonServer()
        {
            try
            {
                client = new TcpClient("xx", 5000); // IP 확인 필요
                stream = client.GetStream();
                isConnected = true;
                MessageBox.Show("Python 서버와 연결 성공!");
            }
            catch (Exception)
            {
                isConnected = false;
            }
        }

        // === [6. 캘리브레이션 결과 계산 함수] ===
        private void CalculateConstants()
        {
            // [1] 수평(H) 계산: 양 끝값(Right, Left)만 사용
            double swingH = Math.Abs(val_Right - val_Left);
            double estCenterH = (val_Right + val_Left) / 2.0; // 중앙값 추정
            // 40도(우+20 ~ 좌-20) 움직였을 때 전압차(swingH)
            // K값 계산: (전체 스윙 / 2)가 20도에 해당함
            double halfSwingH = swingH / 2.0;
            if (halfSwingH > 0.001) K_Horz = 20.0 / halfSwingH;


            double deltaU = Math.Abs(val_Up - val_Center_V);
            double deltaD = Math.Abs(val_Down - val_Center_V);
            double halfSwingV = (deltaU + deltaD) / 2.0;
            double targetAngleV = 10.0;
            if (halfSwingV > 0.001) K_Vert = targetAngleV / halfSwingV;

            // [4] Form1 변수 업데이트 (그래프 0점 보정용)
            val_Center_H = estCenterH;

            isCalibrated = true; // 이제부터 그래프 0점 보정 시작

            MessageBox.Show("Peak-to-Peak 캘리브레이션 완료!\n\n" +
                            $"[수평] 범위: {swingH:F0}, 중앙추정: {estCenterH:F0}\n" +
                            $"[수직] 범위:{halfSwingV:F0}, 중앙추정: {val_Center_V:F0}\n\n" +
                            $"K_H: {K_Horz:F3}, K_V: {K_Vert:F3}");
        }

        // =========================================================
        // [핵심 함수] 3-Case 알고리즘 (Blink, Fixation, Free Move)
        // =========================================================
        private void ProcessSmartGaze(double currH, double currV)
        {
            // [1] 버퍼링
            smartBuff_H.Add(currH);
            smartBuff_V.Add(currV);

            if (smartBuff_H.Count < BUFF_SIZE) return;

            // [2] 데이터 준비
            // delayed: 화면 출력용 (과거 - 상승 전 데이터)
            double delayedH = smartBuff_H[0];
            double delayedV = smartBuff_V[0];

            // recent: 판단용 (최신 - 피크 데이터)
            double recentH = smartBuff_H[BUFF_SIZE - 1];
            double recentV = smartBuff_V[BUFF_SIZE - 1];

            // 기울기 계산 (최근 5프레임 변화량)
            double prevV = smartBuff_V[BUFF_SIZE - 6];
            double slopeV = (recentV - prevV) / 5.0;

            double prevH = smartBuff_H[BUFF_SIZE - 6];
            double slopeH = (recentH - prevH) / 5.0;

            // 현재 전압이 기준점(val_Center) 대비 얼마나 움직였는가?
            double diffV = recentV - val_Center_V;
            double diffH = recentH - val_Center_H;

            // 감도 설정 (기존 코드 값 유지)
            double sensitivity_H = 25.0; 
            double sensitivity_V = 100.0; // 사용자가 설정한 값

            // ==========================================================
            // [판단 로직]
            // ==========================================================

            // [A] 블링크 무시 타이머 작동 중 (Case 1 처리 후)
            if (ignoreTimer > 0)
            {
                ignoreTimer--;
                
                // 타이머가 끝나면, 현재 안정화된 전압을 새로운 0점으로 잡고
                // 화면 기준점은 고정했던 그 자리로 설정 (Floating 완료)
                if (ignoreTimer == 0)
                {
                    screen_center_H = holdPixel_H;
                    screen_center_V = holdPixel_V;
                    
                    // 전압 기준점도 현재 값(안정화된 값)으로 갱신
                    val_Center_H = smartBuff_H[BUFF_SIZE / 2];
                    val_Center_V = smartBuff_V[BUFF_SIZE / 2];
                    
                    // 버퍼 비워서 새 출발
                    smartBuff_H.Clear(); smartBuff_V.Clear();
                    return;
                }

                // 타이머 도중에는 고정된 좌표만 전송
                SendViaTCP((int)holdPixel_H, (int)holdPixel_V);
                
                // 데이터 소비
                smartBuff_H.RemoveAt(0); smartBuff_V.RemoveAt(0);
                return;
            }

            // [B] 이벤트 감지 (H 또는 V가 유의미하게 움직였을 때)
            if (Math.Abs(diffV) > THRESH_LEVEL || Math.Abs(diffH) > THRESH_LEVEL)
            {
                // ------------------------------------------------------
                // Case 1: 블링크 (수직 급하강) -> 상승 전 좌표 고정
                // ------------------------------------------------------
                if (slopeV < THRESH_BLINK_SLOPE)
                {
                    // "상승 전 좌표" 계산 (delayed 데이터 + 현재 screen_center 사용)
                    // 블링크 튀기 전의 평온했던 위치를 계산
                    double preBlinkAngleX = (delayedH - val_Center_H) * K_Horz;
                    double preBlinkAngleY = (delayedV - val_Center_V) * K_Vert;

                    holdPixel_H = screen_center_H + (preBlinkAngleX * sensitivity_H);
                    holdPixel_V = screen_center_V - (preBlinkAngleY * sensitivity_V);

                    // 타이머 설정 (약 0.3~0.4초 무시)
                    ignoreTimer = 25; 
                    
                    SendViaTCP((int)holdPixel_H, (int)holdPixel_V);
                }
                // ------------------------------------------------------
                // Case 2: 응시/드리프트 (완만함) -> 피크 좌표 고정 & 0점 이동
                // ------------------------------------------------------
                else if (Math.Abs(slopeV) < THRESH_FIXATION_SLOPE &&
                         Math.Abs(slopeH) < THRESH_FIXATION_SLOPE)
                {
                    // "피크 좌표" 계산 (recent 데이터 사용)
                    // 지금 눈이 가 있는 그 위치
                    double peakAngleX = (recentH - val_Center_H) * K_Horz;
                    double peakAngleY = (recentV - val_Center_V) * K_Vert;

                    double currentPeakPixel_H = screen_center_H + (peakAngleX * sensitivity_H);
                    double currentPeakPixel_V = screen_center_V - (peakAngleY * sensitivity_V);

                    // ★ 핵심: Floating Center 적용
                    // 1. 화면의 기준점을 지금 보고 있는 '피크 위치'로 옮김
                    screen_center_H = currentPeakPixel_H;
                    screen_center_V = currentPeakPixel_V;

                    // 2. 전압의 기준점을 지금 떠 있는 '높은 전압'으로 옮김
                    val_Center_H = recentH;
                    val_Center_V = recentV;

                    // 3. 화면에는 옮겨진 기준점을 전송 (고정 효과)
                    SendViaTCP((int)screen_center_H, (int)screen_center_V);

                    // 4. 버퍼 초기화 (과거 데이터 삭제로 즉시 반응)
                    smartBuff_H.Clear(); smartBuff_V.Clear();
                    return;
                }
                // ------------------------------------------------------
                // Case 3: 자유 이동 (Scan) -> 지연된 데이터 추적
                // ------------------------------------------------------
                else
                {
                    double moveAngleX = (delayedH - val_Center_H) * K_Horz;
                    double moveAngleY = (delayedV - val_Center_V) * K_Vert;

                    double movePixelX = screen_center_H + (moveAngleX * sensitivity_H);
                    double movePixelY = screen_center_V - (moveAngleY * sensitivity_V);

                    SendViaTCP((int)movePixelX, (int)movePixelY);
                }
            }
            // [C] 평상시 (중앙 부근)
            else
            {
                double idleAngleX = (delayedH - val_Center_H) * K_Horz;
                double idleAngleY = (delayedV - val_Center_V) * K_Vert;

                double idlePixelX = screen_center_H + (idleAngleX * sensitivity_H);
                double idlePixelY = screen_center_V - (idleAngleY * sensitivity_V);

                SendViaTCP((int)idlePixelX, (int)idlePixelY);
            }

            // 데이터 소비
            smartBuff_H.RemoveAt(0);
            smartBuff_V.RemoveAt(0);
        }

        // [보조] TCP 전송 및 좌표 가두기(Clamping) 함수
        private void SendViaTCP(int x, int y)
        {
            // 화면 밖으로 나가지 않게 가두기
            x = Math.Max(0, Math.Min(640, x));
            y = Math.Max(0, Math.Min(480, y));

            if (isConnected && stream != null)
            {
                try
                {
                    string msg = $"{x},{y}\n";
                    byte[] dataToSend = Encoding.UTF8.GetBytes(msg);
                    stream.Write(dataToSend, 0, dataToSend.Length);
                }
                catch
                {
                    isConnected = false;
                }
            }
        }
    }
}