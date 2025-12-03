using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO.Ports; // 시리얼 통신
using System.Net.Sockets; // TCP 통신

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

        // === [캘리브레이션 관련 변수] ===
        CalibState currentCalibState = CalibState.None;
        List<double> tempCalibBuffer = new List<double>();
        double val_Center, val_Right, val_Left, val_Up, val_Down;

        // === [수직 신호 증폭 배수 (Gain)] ===
        // 1.5배 유지 (노이즈와 신호 크기 균형)
        private const double V_GAIN = 1.5;

        // === [수정됨 1] 노이즈 제거 필터 설정 (반응 속도 개선) ===
        // 100은 너무 둔해서(0.2초 지연), 40(0.08초 지연)으로 줄였습니다.
        // 이제 그래프가 훨씬 빠릿빠릿하게 따라올 겁니다.
        private const int FILTER_SIZE = 40; 
        private Queue<double> qFilter_H = new Queue<double>(); // 수평 노이즈 제거용
        private Queue<double> qFilter_V = new Queue<double>(); // 수직 노이즈 제거용

        // === [수정됨 2] 데드존 미세 조정 ===
        // 반응이 빨라진 만큼 데드존을 살짝 조여서 떨림을 잡습니다.
        private const double DEADZONE_H = 3.0; // 수평 유지
        private const double DEADZONE_V = 5.0; // 수직은 8.0 -> 5.0으로 조금 더 민감하게 변경

        // 필터링 및 계산 객체
        private EOGprocess eogFilter = new EOGprocess(); 
        private EOGprocess eogFilter2 = new EOGprocess();
        private GazeCalculator gazeCalc = new GazeCalculator(); 

        string thisdate = DateTime.Now.ToString("yyMMdd");

        // [TCP 통신용 변수]
        private TcpClient client;
        private NetworkStream stream;
        private bool isConnected = false;

        public Form1()
        {
            InitializeComponent();
        }

        // [이동 평균 필터 함수]
        private double GetFilteredData(Queue<double> q, double rawData)
        {
            q.Enqueue(rawData);
            if (q.Count > FILTER_SIZE) q.Dequeue();
            return q.Average();
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

            // [그래프 축 고정]
            try
            {
                scope1.Channels[0].Axis.AutoScaling = false; 
                scope1.Channels[0].Axis.Min = -2000; 
                scope1.Channels[0].Axis.Max = 2000;
                
                if (scope1.Channels.Count > 1)
                {
                    scope1.Channels[1].Axis.AutoScaling = false;
                    scope1.Channels[1].Axis.Min = -2000;
                    scope1.Channels[1].Axis.Max = 2000;
                }
            }
            catch { /* Scope 속성이 다를 경우 무시 */ }

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
                    // 변수 초기화
                    currentCalibState = CalibState.None;
                    val_Center = 0; val_Right = 0; val_Left = 0; val_Up = 0; val_Down = 0;
                    
                    qFilter_H.Clear();
                    qFilter_V.Clear();

                    // 캘리브레이션 창 생성
                    calibration calibForm = new calibration();

                    calibForm.OnMeasureStart += (state) =>
                    {
                        if (state == CalibState.Finish) CalculateConstants();
                        else
                        {
                            currentCalibState = state;
                            tempCalibBuffer.Clear();
                        }
                    };

                    calibForm.OnMeasureStop += () =>
                    {
                        if (tempCalibBuffer.Count > 0)
                        {
                            double avg = tempCalibBuffer.Average();
                            switch (currentCalibState)
                            {
                                case CalibState.Center: val_Center = avg; break;
                                case CalibState.Right: val_Right = avg; break;
                                case CalibState.Left: val_Left = avg; break;
                                case CalibState.Up: val_Up = avg; break;
                                case CalibState.Down: val_Down = avg; break;
                            }
                        }
                        currentCalibState = CalibState.None;
                    };

                    calibForm.ShowDialog(); 

                    btnOpen.Enabled = false;
                    btnClose.Enabled = true;
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

                        // 1. 하드웨어/1차 필터
                        double rawDataH = eogFilter.ProcessSample(Data_1);
                        double rawDataV = eogFilter2.ProcessSample(Data_2);

                        // 2. 소프트웨어 이동 평균 필터 (노이즈 제거)
                        double cleanH = GetFilteredData(qFilter_H, rawDataH);
                        double cleanV = GetFilteredData(qFilter_V, rawDataV);

                        // =========================================================
                        // [신호 반전 및 수직 증폭]
                        // =========================================================
                        cleanH = -cleanH;
                        cleanV = -cleanV * V_GAIN; 

                        // --- [분기점] 캘리브레이션 중 vs 평상시 ---
                        if (currentCalibState != CalibState.None)
                        {
                            if (currentCalibState == CalibState.Right || currentCalibState == CalibState.Left)
                                tempCalibBuffer.Add(cleanH); 
                            else if (currentCalibState == CalibState.Up || currentCalibState == CalibState.Down)
                                tempCalibBuffer.Add(cleanV); 
                            else if (currentCalibState == CalibState.Center)
                                tempCalibBuffer.Add(cleanH);
                        }
                        else
                        {
                            // [평상시 모드]
                            Array.Copy(input_Data_1, 1, input_Data_1, 0, buffsize - 1);
                            Array.Copy(input_Data_2, 1, input_Data_2, 0, buffsize - 1);
                            
                            input_Data_1[buffsize - 1] = cleanH;
                            input_Data_2[buffsize - 1] = cleanV;

                            input_Draw_1 = input_Data_1;
                            input_Draw_2 = input_Data_2;

                            // TCP 전송 및 좌표 변환
                            if (isConnected && stream != null && gazeCalc.IsCalibrated)
                            {
                                try
                                {
                                    double angleH, angleV;
                                    
                                    // 각도 계산
                                    gazeCalc.CalculateGaze(cleanH, cleanV, out angleH, out angleV);

                                    // 데드존 적용
                                    if (Math.Abs(angleH) < DEADZONE_H) angleH = 0;
                                    if (Math.Abs(angleV) < DEADZONE_V) angleV = 0;

                                    // 좌표 변환 (640x480)
                                    System.Drawing.Point pixel = gazeCalc.GetCameraCoordinates(angleH, angleV, 640, 480);

                                    // 전송
                                    string msg = $"{pixel.X},{pixel.Y}\n";
                                    byte[] dataToSend = Encoding.UTF8.GetBytes(msg);
                                    stream.Write(dataToSend, 0, dataToSend.Length);
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
            // 차트 그리기
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
                client = new TcpClient("192.168.165.70", 5000);
                stream = client.GetStream();
                isConnected = true;
                MessageBox.Show("Python 서버와 연결 성공!");
            }
            catch (Exception)
            {
                isConnected = false;
            }
        }

        private void CalculateConstants()
        {
            // 수집된 clean값(증폭된 값)을 계산기에 설정
            gazeCalc.SetCalibrationData(val_Center, val_Right, val_Left, val_Center, val_Up, val_Down);
            System.Diagnostics.Debug.WriteLine("Calibration Finished & Set to GazeCalculator");
        }

        // --- 기타 이벤트 핸들러 ---
        private void Form1_FormClosed(object sender, FormClosedEventArgs e) {
            if (sPort != null && sPort.IsOpen) { sPort.Close(); sPort.Dispose(); sPort = null; }
        }
        private void maskedTextBox1_MaskInputRejected(object sender, MaskInputRejectedEventArgs e) { }
        private void groupBox1_Enter(object sender, EventArgs e) { }
        private void label1_Click(object sender, EventArgs e) { }
        private void scope1_Click(object sender, EventArgs e) { }
    }
}