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
    public enum CalibState { None, Center, Right, Left, Up, Down, Finish }
    public partial class calibration : Form
    {
        // -------------------------------------------------------------
        // [1] 여기가 중요합니다! 타이머 변수 선언
        // -------------------------------------------------------------
        System.Windows.Forms.Timer seqTimer = new System.Windows.Forms.Timer();

        // 시간 카운트용 변수
        int timeTicks = 0;
        public event Action<CalibState> OnMeasureStart; // "측정 시작해!"
        public event Action OnMeasureStop;              // "측정 멈춰!"


        //화면 크기 계산용
        int cx, cy; // gap: 중앙에서 떨어진 거리 (픽셀)
        int gap_x, gap_y; // 중앙에서 떨어진 거리 (가로, 세로)

        public calibration()
        {
            InitializeComponent();
            // 폼 디자인 코드 (디자인 탭에서 해도 됨)
            this.FormBorderStyle = FormBorderStyle.None;
            this.WindowState = FormWindowState.Maximized; // 전체 화면
            this.BackColor = Color.White;

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
                // === [핵심] 사각형을 원으로 깎는 마법의 코드 ===
                System.Drawing.Drawing2D.GraphicsPath path = new System.Drawing.Drawing2D.GraphicsPath();
                path.AddEllipse(0, 0, dot.Width, dot.Height);
                dot.Region = new Region(path);
                // ===========================================
            }

            // === [2. 화면 크기에 따른 좌표 자동 계산] ===
            // 현재 꽉 찬 화면(this.Width, Height)을 기준으로 중앙을 잡습니다.
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

        private void SeqTimer_Tick(object sender, EventArgs e)
        {
            timeTicks++;
            int t = timeTicks; // 1 = 0.1초

            // === 시나리오 (총 23초) ===

            // 1. 중앙 (0~5초)
            if (t == 1) MoveDot(cx, cy);
            if (t == 45) OnMeasureStart?.Invoke(CalibState.Center);
            if (t == 50) OnMeasureStop?.Invoke();

            // 2. 우측 (5~7초) -> gap_x 사용
            if (t == 51) MoveDot(cx + gap_x, cy);
            if (t == 65) OnMeasureStart?.Invoke(CalibState.Right);
            if (t == 70) OnMeasureStop?.Invoke();

            // 3. 중앙 (7~9초)
            if (t == 71) MoveDot(cx, cy);

            // 4. 좌측 (9~11초) -> cx - gap_x
            if (t == 91) MoveDot(cx - gap_x, cy);
            if (t == 105) OnMeasureStart?.Invoke(CalibState.Left);
            if (t == 110) OnMeasureStop?.Invoke();

            // 5. 중앙 (11~13초)
            if (t == 111) MoveDot(cx, cy);

            // 6. 위 (13~15초) -> cy - gap_y (화면 위쪽이 y가 작음)
            if (t == 131) MoveDot(cx, cy - gap_y);
            if (t == 145) OnMeasureStart?.Invoke(CalibState.Up);
            if (t == 150) OnMeasureStop?.Invoke();

            // 7. 중앙 (15~17초)
            if (t == 151) MoveDot(cx, cy);

            // 8. 아래 (17~19초) -> cy + gap_y
            if (t == 171) MoveDot(cx, cy + gap_y);
            if (t == 185) OnMeasureStart?.Invoke(CalibState.Down);
            if (t == 190) OnMeasureStop?.Invoke();

            // 9. 종료 (21초)
            if (t == 191) MoveDot(cx, cy);
            if (t == 210)
            {
                seqTimer.Stop();
                OnMeasureStart?.Invoke(CalibState.Finish); // 끝났다고 알림
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
