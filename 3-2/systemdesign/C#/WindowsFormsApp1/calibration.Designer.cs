namespace WindowsFormsApp1
{
    partial class calibration
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.dotPanel = new System.Windows.Forms.Panel();
            this.SuspendLayout();
            // 
            // dotPanel
            // 
            this.dotPanel.BackColor = System.Drawing.Color.Red;
            this.dotPanel.Location = new System.Drawing.Point(495, 278);
            this.dotPanel.Name = "dotPanel";
            this.dotPanel.Size = new System.Drawing.Size(100, 100);
            this.dotPanel.TabIndex = 1;
            // 
            // calibration
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(10F, 18F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ButtonHighlight;
            this.ClientSize = new System.Drawing.Size(1129, 689);
            this.Controls.Add(this.dotPanel);
            this.ForeColor = System.Drawing.Color.Red;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
            this.Name = "calibration";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel dotPanel;
    }
}