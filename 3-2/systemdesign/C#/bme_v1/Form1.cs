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
    // 캘리브레이션 상태 (프로젝트에 맞게 필요하면 항목 추가/수정)
    public enum CalibState
    {
        None = 0,
        Center,
        Up,
        Down,
        Left,
        Right,
        Finish
    }

    public partial class Form1 : Form
    {
        private object calibLock = new object(); // 스레드 충돌 방지
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

        int debug_hz_count = 0;
        DateTime last_debug_time = DateTime.Now;

        // === [1. 워밍업 관련 변수 (초기 튀는 현상 방지)] ===
        int stable_count = 0;
        const int WARMUP_SAMPLES = 200; // 처음 200개 데이터는 버림

        // === [2. 캘리브레이션 관련 변수] ===
        bool isCalibrated = false;
        CalibState currentCalibState = CalibState.None;

        List<double> tempCalibBuffer_H = new List<double>();
        List<double> tempCalibBuffer_V = new List<double>();

        double val_Center_H, val_Center_V;
        double val_Right, val_Left, val_Up, val_Down;

        double K_Horz_Right = 0.1; // 오른쪽으로 갈 때의 민감도
        double K_Horz_Left = 0.1;  // 왼쪽으로 갈 때의 민감도
        double K_Vert_Up = 0.1;
        double K_Vert_down = 0.1;

        private EOGprocess eogFilter = new EOGprocess();
        private EOGprocess eogFilter2 = new EOGprocess();

        string thisdate = DateTime.Now.ToString("yyMMdd");

        // [TCP 통신용 변수]
        private TcpClient client;
        private NetworkStream stream;
        private bool isConnected = false;

        // =========================================================
        // [EOG 지능형 알고리즘 변수 (Delay & Floating Center)]
        // =========================================================

        // 500Hz 기준 0.4초 지연 = 200샘플
        const int BUFF_SIZE = 100;

        List<double> smartBuff_H = new List<double>();
        List<double> smartBuff_V = new List<double>();

        // 화면 기준점 (Floating Center)
        double screen_center_H = 320.0;
        double screen_center_V = 240.0;

        // 튜닝 파라미터
        const double THRESH_LEVEL = 100.0;
        const double THRESH_BLINK_SLOPE = -10;
        const double THRESH_FIXATION_SLOPE = 5;

        // 화면 크기/타겟 각도 기반 감도
        const int SCREEN_W = 640;
        const int SCREEN_H = 480;
        const double TARGET_ANGLE_H = 20.0; // 캘리브레이션에서 targetAngleH와 동일
        const double TARGET_ANGLE_V = 10.0; // 캘리브레이션에서 targetAngleV와 동일

        // 상태 변수
        int ignoreTimer = 0;      // 블링크 무시 타이머(샘플 단위)
        double holdPixel_H = 320; // 고정된 좌표 저장용
        double holdPixel_V = 240;

        const double THRESH_BLINK_LEVEL = 350.0; // 블링크로 보는 V편차 임계값(튜닝: 250~500)
        const double OUT_ALPHA = 0.25;
        double outX = 320.0;
        double outY = 240.0;
        bool outInit = false;

        // ---------------------------------------------------------

        private static double Clamp(double v, double lo, double hi)
            => Math.Max(lo, Math.Min(hi, v));

        private double GetMedian(List<double> source)
        {
            if (source == null || source.Count == 0) return 0;
            var sorted = source.OrderBy(x => x).ToList();
            int mid = sorted.Count / 2;
            return (sorted.Count % 2 != 0) ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
        }

        public Form1()
        {
            InitializeComponent();
            eogFilter.SetGain(2.0);
            eogFilter2.SetGain(2.4);
        }

        private void maskedTextBox1_MaskInputRejected(object sender, MaskInputRejectedEventArgs e) { }
        private void groupBox1_Enter(object sender, EventArgs e) { }
        private void label1_Click(object sender, EventArgs e) { }
        private void scope2_Click(object sender, EventArgs e) { }
        private void scope1_Click(object sender, EventArgs e) { }

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

            cboPortName.SelectedItem = "COM5";
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
                    sPort.DataBits = 8;
                    sPort.Parity = Parity.None;
                    sPort.StopBits = StopBits.One;
                    sPort.Open();
                }

                if (sPort.IsOpen)
                {
                    // 1) 변수 초기화
                    K_Horz_Right = 0; K_Horz_Left = 0; K_Vert_Up = 0; K_Vert_down = 0;
                    currentCalibState = CalibState.None;

                    val_Center_H = 0; val_Center_V = 0;
                    val_Right = 0; val_Left = 0; val_Up = 0; val_Down = 0;

                    stable_count = 0;

                    // 2) 캘리브레이션 실행
                    RunCalibration();
                }

                btnOpen.Enabled = false;
                btnClose.Enabled = true;
            }
            catch (System.Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void RunCalibration()
        {
            calibration calibForm = new calibration();

            calibForm.OnMeasureStart += (state) =>
            {
                if (state == CalibState.Finish) CalculateConstants();
                else
                {
                    currentCalibState = state;
                    lock (calibLock)
                    {
                        tempCalibBuffer_H.Clear();
                        tempCalibBuffer_V.Clear();
                    }
                }
            };

            calibForm.OnMeasureStop += () =>
            {
                lock (calibLock)
                {
                    if (tempCalibBuffer_H.Count > 0 && tempCalibBuffer_V.Count > 0)
                    {
                        double medH = GetMedian(tempCalibBuffer_H);
                        double medV = GetMedian(tempCalibBuffer_V);

                        switch (currentCalibState)
                        {
                            case CalibState.Center:
                                val_Center_H = medH;
                                val_Center_V = medV;
                                break;

                            case CalibState.Up:
                                val_Up = medV;
                                break;

                            case CalibState.Down:
                                val_Down = medV;
                                break;

                            case CalibState.Right:
                                val_Right = medH;
                                break;

                            case CalibState.Left:
                                val_Left = medH;
                                break;
                        }
                    }
                }

                currentCalibState = CalibState.None;
            };

            calibForm.ShowDialog();
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

        private void SPort_DataReceived(object sender, SerialDataReceivedEventArgs e)
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
                        /*
                        debug_hz_count++;
                        TimeSpan span = DateTime.Now - last_debug_time;
                        if (span.TotalSeconds >= 1.0)
                        {
                            System.Diagnostics.Debug.WriteLine($"현재 샘플 레이트: {debug_hz_count} Hz");
                            debug_hz_count = 0;
                            last_debug_time = DateTime.Now;
                        }
                        */

                        Data_1 = ((data_buff[0] & 0x7F) << 7) + (data_buff[1] & 0x7F);
                        Data_2 = ((data_buff[2] & 0x7F) << 7) + (data_buff[3] & 0x7F);
                        Data_3 = 0;

                        double filteredData = eogFilter.ProcessSample(Data_1);
                        double filteredData2 = eogFilter2.ProcessSample(Data_2);

                        // 워밍업: 초반 데이터 버림
                        if (stable_count < WARMUP_SAMPLES)
                        {
                            stable_count++;
                            start_flag = 0; data_count = 0;
                            continue;
                        }

                        // 그래프 데이터 시프트
                        for (int i = 0; i < buffsize - 1; i++)
                        {
                            input_Data_1[i] = input_Data_1[i + 1];
                            input_Data_2[i] = input_Data_2[i + 1];
                        }
                        input_Data_1[buffsize - 1] = filteredData;
                        input_Data_2[buffsize - 1] = filteredData2;

                        input_Draw_1 = input_Data_1;
                        input_Draw_2 = input_Data_2;

                        // 캘리브레이션 중 vs 평상시
                        if (currentCalibState != CalibState.None)
                        {
                            lock (calibLock)
                            {
                                tempCalibBuffer_H.Add(filteredData);
                                tempCalibBuffer_V.Add(filteredData2);
                            }
                        }
                        else
                        {
                            ProcessSmartGaze(filteredData, filteredData2);
                        }

                        start_flag = 0;
                        data_count = 0;
                    }
                }
            }
        }

        private void On_timer1(object sender, EventArgs e)
        {
            if (scope1.Channels.Count > 0)
                scope1.Channels[0].Data.SetYData(input_Data_1);

            if (scope2.Channels.Count > 0)
                scope2.Channels[0].Data.SetYData(input_Data_2);
        }

        private void ConnectToPythonServer()
        {
            try
            {
                client = new TcpClient("192.168.165.70", 5000); // IP 확인 필요
                stream = client.GetStream();
                isConnected = true;
                MessageBox.Show("Python 서버와 연결 성공!");
            }
            catch
            {
                isConnected = false;
            }
        }

        // === 캘리브레이션 결과 계산 ===
        private void CalculateConstants()
        {
            double targetAngleH = TARGET_ANGLE_H;

            double deltaR_H = Math.Abs(val_Right - val_Center_H);
            K_Horz_Right = (deltaR_H > 0.001) ? (targetAngleH / deltaR_H) : 1.0;

            double deltaL_H = Math.Abs(val_Left - val_Center_H);
            K_Horz_Left = (deltaL_H > 0.001) ? (targetAngleH / deltaL_H) : 1.0;

            double targetAngleV = TARGET_ANGLE_V;

            double deltaU_V = Math.Abs(val_Up - val_Center_V);
            K_Vert_Up = (deltaU_V > 0.001) ? (targetAngleV / deltaU_V) : 0.1;

            double deltaD_V = Math.Abs(val_Down - val_Center_V);
            K_Vert_down = (deltaD_V > 0.001) ? (targetAngleV / deltaD_V) : 0.1;

            // K 폭주 방지(임시 상한)
            K_Horz_Right = Math.Min(K_Horz_Right, 5.0);
            K_Horz_Left = Math.Min(K_Horz_Left, 5.0);
            K_Vert_Up = Math.Min(K_Vert_Up, 5.0);
            K_Vert_down = Math.Min(K_Vert_down, 5.0);

            // 캘리브레이션 직후 화면 커서를 정중앙으로 리셋
            screen_center_H = 320.0;
            screen_center_V = 240.0;

            holdPixel_H = 320.0;
            holdPixel_V = 240.0;

            ignoreTimer = 0;

            smartBuff_H.Clear();
            smartBuff_V.Clear();

            isCalibrated = true;

            MessageBox.Show(
                "Peak-to-Peak 캘리브레이션 완료!\n\n" +
                $"Center_H: {val_Center_H:F0}\n" +
                $"Center_V: {val_Center_V:F0}\n\n" +
                $"K_H_R: {K_Horz_Right:F3}, K_H_L: {K_Horz_Left:F3}\n" +
                $"K_V_U: {K_Vert_Up:F3}, K_V_D: {K_Vert_down:F3}"
            );
        }

        private double GetK_Horz(double diffH)
        {
            return (diffH > 0) ? K_Horz_Right : K_Horz_Left;
        }

        private double GetK_Vert(double diffV)
        {
            return (diffV > 0) ? K_Vert_Up : K_Vert_down;
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
            double delayedH = smartBuff_H[0];
            double delayedV = smartBuff_V[0];
            double recentH = smartBuff_H[BUFF_SIZE - 1];
            double recentV = smartBuff_V[BUFF_SIZE - 1];

            double diffH = recentH - val_Center_H;
            double diffV = recentV - val_Center_V;

            // 각도 변환 gain
            double K_Horz = GetK_Horz(diffH);
            double K_Vert = GetK_Vert(diffV);

            // 기울기 보정 스케일
            double scaleFactor = 50.0;

            // 1) 피크 찾기
            double maxV = smartBuff_V.Max();
            double minV = smartBuff_V.Min();
            double peakV = (Math.Abs(maxV) > Math.Abs(minV)) ? maxV : minV;

            int peakIdxV = smartBuff_V.LastIndexOf(peakV);
            double stepsV = (BUFF_SIZE - 1) - peakIdxV;
            if (stepsV < 1) stepsV = 1;

            double maxH = smartBuff_H.Max();
            double minH = smartBuff_H.Min();
            double peakH = (Math.Abs(maxH) > Math.Abs(minH)) ? maxH : minH;

            int peakIdxH = smartBuff_H.LastIndexOf(peakH);
            double stepsH = (BUFF_SIZE - 1) - peakIdxH;
            if (stepsH < 1) stepsH = 1;

            // 2) 기울기
            double slopeV_Raw = ((recentV - peakV) / stepsV) * scaleFactor;

            double slopeV_Abs = ((Math.Abs(recentV) - Math.Abs(peakV)) / stepsV) * scaleFactor;
            double slopeH_Abs = ((Math.Abs(recentH) - Math.Abs(peakH)) / stepsH) * scaleFactor;

            // 3) 각도->픽셀 감도 (하드코딩 제거)
            double sensitivity_H = 25; //(SCREEN_W / 2.0) / TARGET_ANGLE_H;  16
            double sensitivity_V = 80;  //(SCREEN_H / 2.0) / TARGET_ANGLE_V; 24

            // [A] 블링크 무시 타이머
            if (ignoreTimer > 0)
            {
                ignoreTimer--;

                if (ignoreTimer == 0)
                {
                    // 저장값을 center로 복귀 (클램프!)
                    screen_center_H = Clamp(holdPixel_H, 0, SCREEN_W);
                    screen_center_V = Clamp(holdPixel_V, 0, SCREEN_H);

                    var tailH = smartBuff_H.Skip(Math.Max(0, BUFF_SIZE - 30)).ToList();
                    var tailV = smartBuff_V.Skip(Math.Max(0, BUFF_SIZE - 30)).ToList();
                    val_Center_H = GetMedian(tailH);
                    val_Center_V = GetMedian(tailV);

                    smartBuff_H.Clear();
                    smartBuff_V.Clear();
                    return;
                }

                SendViaTCP((int)holdPixel_H, (int)holdPixel_V);

                smartBuff_H.RemoveAt(0);
                smartBuff_V.RemoveAt(0);
                return;
            }

            if (Math.Abs(diffV) > THRESH_BLINK_LEVEL)
            {
                // "지금 보고 있던 위치"를 고정하고 싶으면 outX/outY를 쓰는 게 제일 안정적
                // (outInit이 아직 false면 화면 중앙 사용)
                holdPixel_H = outInit ? outX : screen_center_H;
                holdPixel_V = outInit ? outY : screen_center_V;

                holdPixel_H = Clamp(holdPixel_H, 0, SCREEN_W);
                holdPixel_V = Clamp(holdPixel_V, 0, SCREEN_H);

                ignoreTimer = 250; // 기존 유지

                SendViaTCP((int)holdPixel_H, (int)holdPixel_V);

                // 아래 RemoveAt/return까지 해줘야 버퍼가 계속 굴러가면서 ignoreTimer 구간이 정상 동작함
                smartBuff_H.RemoveAt(0);
                smartBuff_V.RemoveAt(0);
                return;
            }

            // [B] 이벤트 감지
            if (Math.Abs(diffV) > THRESH_LEVEL || Math.Abs(diffH) > THRESH_LEVEL)
            {
                // Case 1: Blink
                if (peakV > 0 && slopeV_Raw < THRESH_BLINK_SLOPE)
                {
                    double preBlinkAngleX = (delayedH - val_Center_H) * GetK_Horz(delayedH - val_Center_H);
                    double preBlinkAngleY = (delayedV - val_Center_V) * GetK_Vert(delayedV - val_Center_V);

                    holdPixel_H = screen_center_H + (preBlinkAngleX * sensitivity_H);
                    holdPixel_V = screen_center_V - (preBlinkAngleY * sensitivity_V);

                    // ★ 저장값도 반드시 클램프
                    holdPixel_H = Clamp(holdPixel_H, 0, SCREEN_W);
                    holdPixel_V = Clamp(holdPixel_V, 0, SCREEN_H);

                    // 500Hz 기준 0.5초 = 250샘플
                    ignoreTimer = 250;

                    SendViaTCP((int)holdPixel_H, (int)holdPixel_V);
                }
                // Case 2: Fixation
                else if (Math.Abs(slopeV_Abs) < THRESH_FIXATION_SLOPE &&
                         Math.Abs(slopeH_Abs) < THRESH_FIXATION_SLOPE)
                {
                    double peakAngleX = (recentH - val_Center_H) * GetK_Horz(recentH - val_Center_H);
                    double peakAngleY = (recentV - val_Center_V) * GetK_Vert(recentV - val_Center_V);

                    double currentPeakPixel_H = screen_center_H + (peakAngleX * sensitivity_H);
                    double currentPeakPixel_V = screen_center_V - (peakAngleY * sensitivity_V);

                    // ★ center 갱신도 반드시 클램프
                    screen_center_H = Clamp(currentPeakPixel_H, 0, SCREEN_W);
                    screen_center_V = Clamp(currentPeakPixel_V, 0, SCREEN_H);

                    val_Center_H = recentH;
                    val_Center_V = recentV;

                    SendViaTCP((int)screen_center_H, (int)screen_center_V);

                    smartBuff_H.Clear();
                    smartBuff_V.Clear();
                    return;
                }
                // Case 3: Scan
                else
                {
                    double moveAngleX = (delayedH - val_Center_H) * GetK_Horz(delayedH - val_Center_H);
                    double moveAngleY = (delayedV - val_Center_V) * GetK_Vert(delayedV - val_Center_V);

                    double movePixelX = screen_center_H + (moveAngleX * sensitivity_H);
                    double movePixelY = screen_center_V - (moveAngleY * sensitivity_V);

                    SendViaTCP((int)movePixelX, (int)movePixelY);

                    // center 드리프트 완만하게
                    val_Center_H += (delayedH - val_Center_H) * 0.005;
                    val_Center_V += (delayedV - val_Center_V) * 0.005;
                }
            }
            // [C] 평상시
            else
            {
                double idleAngleX = (delayedH - val_Center_H) * GetK_Horz(delayedH - val_Center_H);
                double idleAngleY = (delayedV - val_Center_V) * GetK_Vert(delayedV - val_Center_V);

                double idlePixelX = screen_center_H + (idleAngleX * sensitivity_H);
                double idlePixelY = screen_center_V - (idleAngleY * sensitivity_V);

                SendViaTCP((int)idlePixelX, (int)idlePixelY);
            }

            smartBuff_H.RemoveAt(0);
            smartBuff_V.RemoveAt(0);
        }

        // [보조] TCP 전송 및 좌표 가두기(Clamping)
        private void SendViaTCP(int x, int y)
        {
            // 1) 먼저 클램프
            x = Math.Max(0, Math.Min(SCREEN_W, x));
            y = Math.Max(0, Math.Min(SCREEN_H, y));

            // 2) 블링크 고정 중에는 "즉시 고정" (원하는 느낌 유지)
            //    - 너 코드에서 ignoreTimer 동안 holdPixel을 계속 보내는데,
            //      여기서 EMA 걸면 고정이 살짝 미끄러질 수 있어서 바로 고정시킴
            if (ignoreTimer > 0)
            {
                outX = x;
                outY = y;
                outInit = true;
            }
            else
            {
                // 3) 첫 프레임은 그대로
                if (!outInit)
                {
                    outX = x;
                    outY = y;
                    outInit = true;
                }
                else
                {
                    // 4) EMA 스무딩: out = out + alpha*(target - out)
                    outX = outX + OUT_ALPHA * (x - outX);
                    outY = outY + OUT_ALPHA * (y - outY);
                }
            }

            int sx = (int)Math.Round(outX);
            int sy = (int)Math.Round(outY);

            // 5) 마지막 클램프
            sx = Math.Max(0, Math.Min(SCREEN_W, sx));
            sy = Math.Max(0, Math.Min(SCREEN_H, sy));

            if (isConnected && stream != null)
            {
                try
                {
                    string msg = $"{sx},{sy}\n";
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
