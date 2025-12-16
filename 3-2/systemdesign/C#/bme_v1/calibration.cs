using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApp1
{
    //public enum CalibState { None, Center, Right, Left, Up, Down, Finish }
    public partial class calibration : Form
    {
        System.Windows.Forms.Timer seqTimer = new System.Windows.Forms.Timer();
        int timeTicks = 0;
        public event Action<CalibState> OnMeasureStart; // "측정 시작해!"
        public event Action OnMeasureStop;              // "측정 멈춰!"


        //화면 크기 계산용
        int cx, cy; // gap: 중앙에서 떨어진 거리 (픽셀)
        int gap_x, gap_y; // 중앙에서 떨어진 거리 (가로, 세로)

        public calibration()
        {
            InitializeComponent();
            this.FormBorderStyle = FormBorderStyle.None;
            this.WindowState = FormWindowState.Maximized; // 전체 화면
            this.BackColor = Color.White;
            this.KeyPreview = true;
            this.KeyDown += new KeyEventHandler(calibration_KeyDown);

            // 점(Dot) 동적 생성 (도구상자에서 Panel 안 만들었으면 이 코드 사용)
            if (this.Controls["dotPanel"] == null)
            {
                Panel dot = new Panel();
                dot.Name = "dotPanel";
                dot.Size = new Size(30, 30);
                dot.BackColor = Color.Red; // 빨간 점
                this.Controls.Add(dot);
            }

            // 타이머 설정 (0.1초 = 100ms 마다 틱)
            seqTimer.Interval = 100;
            seqTimer.Tick += SeqTimer_Tick;

            this.Shown += new EventHandler(calibration_Load);
        }

        // 창이 로드되면(켜지면) 자동으로 실행되는 함수
        private void calibration_Load(object sender, EventArgs e)
        {
            Control dot = this.Controls["dotPanel"];

            if (dot != null)
            {
                System.Drawing.Drawing2D.GraphicsPath path = new System.Drawing.Drawing2D.GraphicsPath();
                path.AddEllipse(0, 0, dot.Width, dot.Height);
                dot.Region = new Region(path);
            }

            // === [2. 화면 크기에 따른 좌표 자동 계산] ===
            cx = this.Width / 2;
            cy = this.Height / 2;

            // gap을 화면 크기에 비례하게 설정 (노트북마다 해상도가 다르므로)
            // 예: 가로 이동폭은 화면 너비의 35%, 세로 이동폭은 화면 높이의 35%
            // 이렇게 하면 어떤 화면에서도 점이 짤리지 않고 적절히 멀리 갑니다.
            gap_x = (int)(this.Width * 0.35);
            gap_y = (int)(this.Height * 0.35);

            // 시작 위치: 중앙
            MoveDot(cx, cy);

            timeTicks = 0;
            seqTimer.Start();
        }
        // calibration.cs 내부 SeqTimer_Tick 함수 수정 제안

        // [핵심 로직 수정] 순서: 중앙 -> 위 -> 아래 -> 우 -> 좌
        private void SeqTimer_Tick(object sender, EventArgs e)
        {
            timeTicks++;
            int t = timeTicks; // 1 = 0.1초
            
            if (t == 1) MoveDot(cx, cy); // 중앙 위치
            if (t == 4) OnMeasureStart?.Invoke(CalibState.Center);
            if (t == 14) OnMeasureStop?.Invoke();

            // =======================================================
            // 2. 위쪽 (Up) - 10도
            // =======================================================
            if (t == 51) MoveDot(cx, cy - gap_y); // 위로 이동
            if (t == 54) OnMeasureStart?.Invoke(CalibState.Up);
            if (t == 64) OnMeasureStop?.Invoke();

            if (t == 71) MoveDot(cx, cy);


            // =======================================================
            // 3. 아래쪽 (Down) - 10도
            // =======================================================
            if (t == 121) MoveDot(cx, cy + gap_y); // 아래로 이동
            if (t == 124) OnMeasureStart?.Invoke(CalibState.Down);
            if (t == 134) OnMeasureStop?.Invoke();

            if (t==141) MoveDot(cx, cy);

            // =======================================================
            // 4. 우측 (Right) - 20도
            // =======================================================
            if (t == 191) MoveDot(cx + gap_x, cy); // 우측 이동
            if (t == 194) OnMeasureStart?.Invoke(CalibState.Right);
            if (t == 204) OnMeasureStop?.Invoke();

            if (t == 211) MoveDot(cx, cy);


            // =======================================================
            // 5. 좌측 (Left) - 20도
            // =======================================================
            if (t == 261) MoveDot(cx - gap_x, cy); // 좌측 이동
            if (t == 264) OnMeasureStart?.Invoke(CalibState.Left);
            if (t == 274) OnMeasureStop?.Invoke();

            // =======================================================
            // 6. 종료 (Finish)
            // =======================================================
            if (t == 281) MoveDot(cx, cy); // 중앙 복귀
            if (t == 291)
            {
                seqTimer.Stop();
                OnMeasureStart?.Invoke(CalibState.Finish);
                this.Close();
            }
        }

        private void calibration_Load_1(object sender, EventArgs e)
        {

        }

        private void calibration_KeyDown(object sender, KeyEventArgs e)
        {
            // ESC 키가 눌렸을 때 실행
            if (e.KeyCode == Keys.Escape)
            {
                seqTimer.Stop(); 
                OnMeasureStop?.Invoke(); 
                this.Close(); 
            }
        }

        private void MoveDot(int x, int y)
        {
            Control dot = this.Controls["dotPanel"];
            if (dot != null)
            {
                // 점의 중심이 해당 좌표에 오도록 조정
                dot.Location = new Point(x - dot.Width / 2, y - dot.Height / 2);
            }
        }
    }
}
