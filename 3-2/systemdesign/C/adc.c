// 8-2 Multiple ADC conversion (Final Fixed Version)
// ************************************************************

#include <msp430.h>


// 전역 변수 선언
int adc1, adc2, adc3, adc4, adc5, adc6; 
unsigned char Packet[13]; 

void ReadAdc12(void);   // Read data from internal 12 bits ADC

void main(void)
{
    unsigned int i;

    // Set basic clock and timer
    WDTCTL = WDTPW + WDTHOLD;       // Stop WDT
    BCSCTL1 &= ~XT2OFF;             // XT2 on
    do{
        IFG1 &= ~OFIFG;             // Clear oscillator flag
        for(i=0; i<0xFF; i++);      // Delay for OSC to stabilize
    } while((IFG1 & OFIFG));

    BCSCTL2 = SELM_2;               // MCLK = XT2CLK=6MHz
    BCSCTL2 |= SELS;                // SMCLK=XT2CLK=6MHz

    // Set Port
    P3SEL = BIT4|BIT5;              // P3.4,5 = USART0 TXD/RXD
    
   // P6.0(TP1), P6.1(TP8) 두 핀만 ADC 입력으로 설정
    P6SEL = 0x3f; P6DIR=0x3f; P6OUT=0x00;                    

    // Set UART0
    ME1 |= UTXE0 + URXE0;           // Enable USART0 TXD/RXD
    UCTL0 |= CHAR;                  // 8-bit character
    UTCTL0 |= SSEL0|SSEL1;                // UCLK=SMCLK
    UBR00 = 0x34;                   // 115200 baud rate = 0x34
    UBR10 = 0x00;
    UMCTL0 = 0x00;
    UCTL0 &= ~SWRST;                // Initialize USART state machine

    // Set 12bit Internal ADC
    ADC12CTL0 = ADC12ON | REFON | REF2_5V;  // ADC on, 2.5 V reference on
    ADC12CTL0 |= MSC;
    
    ADC12CTL1=ADC12SSEL_3 | ADC12DIV_7 | CONSEQ_1;
    ADC12CTL1 |= SHP;
  
    // ADC 시퀀스 설정 (TP1 -> TP8)
    ADC12MCTL0 = SREF_0 | INCH_0;               // ADC input channel A0 (TP1)
    ADC12MCTL1 = SREF_0 | INCH_1;
    ADC12MCTL2 = SREF_0 | INCH_2;
    ADC12MCTL3 = SREF_0 | INCH_3;
    ADC12MCTL4 = SREF_0 | INCH_4;
    ADC12MCTL5 = SREF_0 | INCH_5 |EOS;          

    ADC12CTL0 |= ENC;                  // enable conversion

    // Set TimerA
    TACTL = TASSEL_2 + MC_1;    // clock source and mode(UP) select
    TACCTL0 = CCIE;             // Enable interrupt
    TACCR0 = 12000;             // 6M/12000=500Hz -> sample rate

    _BIS_SR(LPM0_bits+GIE);                    // Enter LPM0 w/ interrupt
}

// ************************************************************

// TimerA interrupt
#pragma vector = TIMERA0_VECTOR
__interrupt void TimerA0_interrupt()
{
    ReadAdc12();

    Packet[0] = (unsigned char)0x81;      // Header
    _no_operation();
    // TP1 데이터
    Packet[1] = (unsigned char)(adc1>>7) & 0x7F; 
    Packet[2] = (unsigned char)adc1 & 0x7F;
    
    // TP8 데이터
    Packet[3] = (unsigned char)(adc2>>7) & 0x7F; 
    Packet[4] = (unsigned char)adc2 & 0x7F;
    
    // 나머지 0 채움
    Packet[5] = 0; 
    Packet[6] = 0; 
    
    // [에러 수정] for(int i..) -> for(i..)
    int j;
    for(j=0; j<7; j++){
        while (!(IFG1 & UTXIFG0));  // USART0 TX buffer ready?
        TXBUF0 = Packet[j];        // send data
    }
}

// ************************************************************

// Read ADC12 conversion result
void ReadAdc12(void)
{
  adc1=(int)((long)ADC12MEM0*9000/4096)-4500+7000;
  adc2=(int)((long)ADC12MEM1*9000/4096)-4500+7000;
  adc3=(int)((long)ADC12MEM2*9000/4096)-4500+7000;
  adc4=(int)((long)ADC12MEM3*9000/4096)-4500+7000;
  adc5=(int)((long)ADC12MEM4*9000/4096)-4500+7000;
  adc6=(int)((long)ADC12MEM5*9000/4096)-4500+7000;
  ADC12CTL0 |= ADC12SC;                   // start conversion
}