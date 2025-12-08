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

        // 수직 게인 (필요시 조정)
        private const double V_GAIN = 1.0;

        public Form1()
        {
            InitializeComponent();
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

                    // [이벤트 연결 1] 측정 시작
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
                        }
                    };

                    // [이벤트 연결 2] 측정 멈춤 (값 저장 로직)
                    calibForm.OnMeasureStop += () =>
                    {
                        // 데이터가 있다면 평균 계산
                        if (tempCalibBuffer_H.Count > 0 && tempCalibBuffer_V.Count > 0)
                        {
                            double avgH = tempCalibBuffer_H.Average();
                            double avgV = tempCalibBuffer_V.Average();

                            switch (currentCalibState)
                            {
                                case CalibState.Center:
                                    // ★★★ 핵심: Center일 때 가로/세로 기준을 각각 저장 ★★★
                                    val_Center_H = avgH;
                                    val_Center_V = avgV;
                                    break;
                                case CalibState.Right: val_Right = avgH; break;
                                case CalibState.Left: val_Left = avgH; break;
                                case CalibState.Up: val_Up = avgV; break;
                                case CalibState.Down: val_Down = avgV; break;
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

                        // ★ [디버깅용] 진짜 들어오는 값 확인 (필요시 주석 해제)
                        // System.Diagnostics.Debug.WriteLine($"RAW -> H: {Data_1}, V: {Data_2}");

                        double filteredData = eogFilter.ProcessSample(Data_1);
                        double filteredData2 = eogFilter2.ProcessSample(Data_2);

                        // === [워밍업] 초반 데이터 200개 버리기 (그래프 튀는 현상 방지) ===
                        if (stable_count < WARMUP_SAMPLES)
                        {
                            stable_count++;
                            start_flag = 0; data_count = 0;
                            continue; // 여기서 함수 끝냄 (그래프 안 그림)
                        }

                        // 수직 게인 적용 (필요시)
                        filteredData2 = filteredData2 * V_GAIN;

                        // ==========================================================
                        // ★★★ [수정] 그래프 그리기를 조건문 밖으로 뺐습니다! ★★★
                        // 이제 캘리브레이션 중에도 그래프가 멈추지 않고 움직입니다.
                        // ==========================================================
                        for (int i = 0; i < buffsize - 1; i++)
                        {
                            input_Data_1[i] = input_Data_1[i + 1];
                            input_Data_2[i] = input_Data_2[i + 1];
                        }
                        input_Data_1[buffsize - 1] = filteredData;
                        input_Data_2[buffsize - 1] = filteredData2;

                        input_Draw_1 = input_Data_1;
                        input_Draw_2 = input_Data_2;
                        // ==========================================================

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
                            // [SPort_DataReceived 함수 내부 맨 아래쪽]

                            // [평상시 모드] TCP 전송 및 좌표 변환 로직
                            if (isConnected && stream != null)
                            {
                                try
                                {
                                    // 1. 각도 계산 (기존 로직)
                                    double angleX = (filteredData - val_Center_H) * K_Horz; 
                                    double angleY = (filteredData2 - val_Center_V) * K_Vert; 

                                    // 2. [추가된 기능] 드리프트 자동 보정 (중요!)
                                    // 시선이 한쪽으로 쏠려 있으면, 중앙점(Center)을 아주 천천히 그쪽으로 이동시킵니다.
                                    // EOG 신호가 시간이 지나면 흘러내리는 현상을 막아줍니다.
                                    val_Center_H += (filteredData - val_Center_H) * 0.001; 
                                    val_Center_V += (filteredData2 - val_Center_V) * 0.001;

                                    // 3. 각도 -> 픽셀 변환 (감도 조절)
                                    // sensitivity 값을 키우면 눈을 조금만 움직여도 화면 끝까지 갑니다.
                                    double sensitivity = 25.0; // (기존보다 더 높임: 18.0 -> 25.0)

                                    // 화면 중앙(320, 240)을 기준으로 더하고 뺍니다.
                                    int pixelX = 320 + (int)(angleX * sensitivity);
                                    int pixelY = 240 - (int)(angleY * sensitivity); // Y축은 위가 (-)라서 뺌

                                    // 4. 화면 밖으로 나가지 않게 가두기 (Clamping)
                                    pixelX = Math.Max(0, Math.Min(640, pixelX));
                                    pixelY = Math.Max(0, Math.Min(480, pixelY));

                                    // 5. 전송 (좌표값)
                                    string msg = $"{pixelX},{pixelY}\n";
                                    byte[] dataToSend = Encoding.UTF8.GetBytes(msg);
                                    stream.Write(dataToSend, 0, dataToSend.Length);
                                    
                                    // [디버깅] 값이 잘 나오는지 출력창에서 확인해보세요
                                    // System.Diagnostics.Debug.WriteLine($"Send: {pixelX}, {pixelY}");
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
            scope1.Channels[0].Data.SetYData(input_Data_1);
            if (scope1.Channels.Count > 1)
            {
                scope1.Channels[1].Data.SetYData(input_Data_2);
            }
        }

        private void ConnectToPythonServer()
        {
            try
            {
                client = new TcpClient("10.146.99.70", 5000); // IP 확인 필요
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
            // 수평(H) 계산: val_Center_H 사용
            double diffRight = Math.Abs(val_Right - val_Center_H);
            double diffLeft = Math.Abs(val_Left - val_Center_H);
            double avgH = (diffRight + diffLeft) / 2.0;

            if (avgH > 0.001) K_Horz = 20.0 / avgH;

            // 수직(V) 계산: val_Center_V 사용 (이제 정확해짐!)
            double diffUp = Math.Abs(val_Up - val_Center_V);
            double diffDown = Math.Abs(val_Down - val_Center_V);
            double avgV = (diffUp + diffDown) / 2.0;

            if (avgV > 0.001) K_Vert = 20.0 / avgV;

            MessageBox.Show($"캘리브레이션 완료!\n\n" +
                            $"[측정값 평균]\n중앙(H,V): {val_Center_H:F0}, {val_Center_V:F0}\n" +
                            $"우측: {val_Right:F0}, 좌측: {val_Left:F0}\n" +
                            $"위: {val_Up:F0}, 아래: {val_Down:F0}\n\n" +
                            $"[계산된 상수 K]\n수평(Horz): {K_Horz:F3}\n수직(Vert): {K_Vert:F3}");
        }
    }
}