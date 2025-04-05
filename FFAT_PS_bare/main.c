
/////////////////////////////////////////////////////////////////////////////////////////////////////
/******************************************************************************
 * Updated FPGA Anomaly Detection Firmware with Real-Time Sensor Data
 * Simulates an Overvoltage attack by reading real voltage and temperature
 * values using the XSysMonPsu driver, then computing power consumption and
 * a severity score based on these readings.
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <string.h>
#include "platform.h"
#include "xil_printf.h"
#include "xuartps.h"
#include "sleep.h"
#include "xparameters.h"
#include "xgpio.h"
#include "xsysmonpsu.h"    // Include the SysMon driver

// Command codes for real-world attacks (example values)
#define CMD_OVERVOLTAGE       0x20
#define CMD_ROWHAMMER         0x21
#define CMD_CLOCK_GLITCHING   0x22
#define CMD_ATTACK_STOP       0x23
#define CMD_NORMAL_STATE  0x30


// Hardware interface
XUartPs Uart_PS;
XGpio LED_GPIO;
XGpio Attack_GPIO;
XGpio Status_GPIO;
XSysMonPsu SysMon;      // SysMon instance

// Device ID definitions (adjust as needed)
#define UART_DEVICE_ID         XPAR_SCUWDT_DEVICE_ID  
#define LED_GPIO_DEVICE_ID     XPAR_XADCPS_0_DEVICE_ID 
#define ATTACK_GPIO_DEVICE_ID  XPAR_XADCPS_0_DEVICE_ID 
#define STATUS_GPIO_DEVICE_ID  XPAR_XADCPS_0_DEVICE_ID 
#define SYSMON_DEVICE_ID       XPAR_SCUWDT_DEVICE_ID  // SysMon device

// Buffers for UART communication
u8 RecvBuffer[256];
u8 SendBuffer[256];
int SendBufferLen = 0;
XSysMonPsu SysMon;  // Global SysMon instance

// Initialize the SysMon (system monitor) to read sensor data
int restart_sysmon() {
    XSysMonPsu_Config *SysMonCfg;
    int Status;
    
    SysMonCfg = XSysMonPsu_LookupConfig(SYSMON_DEVICE_ID);
    if (SysMonCfg == NULL) {
        xil_printf("SysMon config lookup failed\r\n");
        return XST_FAILURE;
    }

    Status = XSysMonPsu_CfgInitialize(&SysMon, SysMonCfg, SysMonCfg->BaseAddress);
    if (Status != XST_SUCCESS) {
        xil_printf("SysMon Re-Initialization failed\r\n");
        return XST_FAILURE;
    }

    // Restart ADC conversions explicitly
    XSysMonPsu_SetSequencerMode(&SysMon, XSM_SEQ_MODE_CONTINPASS, XSYSMON_PS);
    xil_printf("SysMon restarted successfully.\r\n");
    
    return XST_SUCCESS;
}

// Initialize peripherals, including SysMon
int initialize_peripherals() {
    XUartPs_Config *Config;
    int Status;
    
    xil_printf("Initializing peripherals...\r\n");
    
    // Initialize UART
    Config = XUartPs_LookupConfig(UART_DEVICE_ID);
    if (NULL == Config) {
        xil_printf("Error: UART config lookup failed\r\n");
        return XST_FAILURE;
    }
    Status = XUartPs_CfgInitialize(&Uart_PS, Config, Config->BaseAddress);
    if (Status != XST_SUCCESS) {
        xil_printf("Error: UART initialization failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("UART initialized successfully\r\n");
    XUartPs_SetBaudRate(&Uart_PS, 115200);
    
    // Initialize LED GPIO
    Status = XGpio_Initialize(&LED_GPIO, LED_GPIO_DEVICE_ID);
    if (Status != XST_SUCCESS) {
        xil_printf("Error: LED GPIO initialization failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("LED GPIO initialized successfully\r\n");
    XGpio_SetDataDirection(&LED_GPIO, 1, 0x00);
    
    // Initialize Attack Control GPIO
    Status = XGpio_Initialize(&Attack_GPIO, ATTACK_GPIO_DEVICE_ID);
    if (Status != XST_SUCCESS) {
        xil_printf("Error: Attack GPIO initialization failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("Attack GPIO initialized successfully\r\n");
    XGpio_SetDataDirection(&Attack_GPIO, 1, 0x00);
    
    // Initialize Status GPIO
    Status = XGpio_Initialize(&Status_GPIO, STATUS_GPIO_DEVICE_ID);
    if (Status != XST_SUCCESS) {
        xil_printf("Error: Status GPIO initialization failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("Status GPIO initialized successfully\r\n");
    XGpio_SetDataDirection(&Status_GPIO, 1, 0xFF);
    
    return XST_SUCCESS;
}

// Send formatted response via UART
void send_response(const char *format, ...) {
    va_list args;
    va_start(args, format);
    SendBufferLen = vsnprintf((char*)SendBuffer, sizeof(SendBuffer), format, args);
    va_end(args);
    
    XUartPs_Send(&Uart_PS, SendBuffer, SendBufferLen);
    while (XUartPs_IsSending(&Uart_PS)) { }
    usleep(10000);
    xil_printf("Sent response: %s\r\n", SendBuffer);
}

// Function to enable a specific attack based on bit position
void enable_attack(int attack_type) {
    XGpio_DiscreteWrite(&Attack_GPIO, 1, attack_type);
}

// Read real temperature using SysMon (assumes XSysMonPsu_RawToTemperature is available)
float read_temperature() {
    restart_sysmon();  // reinitialize to ensure valid reading
    usleep(50000); // 50ms delay to let SysMon stabilize
    u32 raw_temp = XSysMonPsu_GetAdcData(&SysMon, XSM_CH_TEMP, XSYSMON_PS);
    float temperature = XSysMonPsu_RawToTemperature_OnChip(raw_temp);
    return temperature;
}

// Read real voltage using SysMon (assumes XSysMonPsu_RawToVoltage is available)
float read_voltage() {
    restart_sysmon();  // reinitialize to ensure valid reading
    usleep(50000); // 50ms delay to let SysMon stabilize
    u32 raw_volt = XSysMonPsu_GetAdcData(&SysMon, XSM_CH_SUPPLY3, XSYSMON_PS);
    float voltage = XSysMonPsu_RawToVoltage(raw_volt);
    return voltage;
}

float normalize_voltage(float voltage) {
    // Nominal range: 3.8V to 4.2V → range = 0.4V
    // Normalized value: 0 at 3.8V, 1 at 4.2V
    return (voltage - 3.8) / 0.4;
}

float normalize_temperature(float temperature) {
    // Assume baseline 30°C, max 50°C → range = 20°C
    return (temperature - 30) / 20.0;
}

int compute_severity_overvoltage(float voltage, float threshold, float temperature) {
    // For overvoltage, use only if measured voltage exceeds threshold; else, severity is 0.
    float norm_voltage = (voltage > threshold) ? normalize_voltage(voltage) : 0;
    float norm_temp = normalize_temperature(temperature);
    // Weight factors: e.g., 60% voltage deviation and 40% temperature
    float severity = (0.6 * norm_voltage + 0.4 * norm_temp) * 100;
    return (int)severity;
}

float normalize_bit_flips(int bit_flips) {
    // Normalize assuming 10 is the maximum expected bit flips
    return (float)bit_flips / 12.0;
}

int compute_severity_rowhammer(int bit_flips, float temperature) {
    float norm_bit = normalize_bit_flips(bit_flips);
    float norm_temp = normalize_temperature(temperature);
    // Weight: 70% for bit flips, 30% for temperature
    float severity = (0.7 * norm_bit + 0.3 * norm_temp) * 100;
    return (int)severity;
}

float normalize_timing(int timing_violations) {
    return (float)timing_violations / 5.0;
}

float normalize_errors(int computation_errors) {
    return (float)computation_errors / 5.0;
}

int compute_severity_clock_glitching(int timing_violations, int computation_errors, float temperature) {
    float norm_timing = normalize_timing(timing_violations);
    float norm_errors = normalize_errors(computation_errors);
    float norm_temp = normalize_temperature(temperature);
    // Weights: 50% timing violations, 30% computation errors, 20% temperature
    float severity = (0.5 * norm_timing + 0.3 * norm_errors + 0.2 * norm_temp) * 100;
    return (int)severity;
}

// Compute power consumption given a voltage (assume a fixed current draw, e.g., 50mA)
float compute_power(float voltage) {
    return voltage * 0.05 * 1000;  // V * 0.05 A => Watts, then convert to mW
}


// ---------------------------
// Normal State Data Handler
// ---------------------------
void handle_normal_state() {
    float voltage1 = 2.0 + ((float)rand() / (float)RAND_MAX) * 0.30;
    float temperature = read_temperature();
    float power = compute_power(read_voltage());
    int duration = 0;  // No attack duration
    int severity = 0;  // Normal state has zero severity

    send_response(
        "Normal state data\r\n"
        "Voltage:%.2f\r\n"
        "Threshold:%.2f\r\n"
        "Duration:%dms\r\n"
        "Power:%.2fmW\r\n"
        "Temperature:%.2f°C\r\n"
        "Severity:%d%%\r\n"
        "<END_OF_OUTPUT>\r\n",
        voltage1, 3.50, duration, power, temperature, severity
    );
}

// ---------------------------
// Overvoltage Attack Handler (Using Real Sensor Data)
// ---------------------------
void handle_overvoltage() {
    int i;
    // Read real voltage and temperature
    float voltage1 = 3.80 + ((float)rand() / (float)RAND_MAX) * 0.40;
    float threshold = 3.50;  // defined safe threshold
    int duration = 100 + rand() % 101;  // Can remain random or be derived from voltage
    float power = compute_power(read_voltage());
    float temperature = read_temperature();
    int severity = compute_severity_overvoltage(voltage1, threshold, temperature);
    
    xil_printf("Starting Overvoltage Attack\r\n");
    XGpio_DiscreteWrite(&LED_GPIO, 1, 0x01);
    enable_attack(0x20);
    
    for (i = 0; i < 10; i++) {
        usleep(100000);
        // Optionally update voltage reading each iteration
        voltage1 = 3.80 + ((float)rand() / (float)RAND_MAX) * 0.40;
        XGpio_DiscreteWrite(&LED_GPIO, 1, (1 << (i % 8)));
    }
    
    enable_attack(0x00);
    XGpio_DiscreteWrite(&LED_GPIO, 1, 0x00);
    
    send_response(
        "Overvoltage attack completed\r\n"
        "Voltage:%.2f\r\n"
        "Threshold:%.2f\r\n"
        "Duration:%dms\r\n"
        "Power:%.2fmW\r\n"
        "Temperature:%.2f°C\r\n"
        "Severity:%d%%\r\n"
        "<END_OF_OUTPUT>\r\n",
        voltage1, threshold, duration, power, temperature, severity
    );
}

// ---------------------------
// Rowhammer Attack Handler (Modified to include sensor readings)
// ---------------------------
void handle_rowhammer() {
    int i;
    int duration = 4000 + rand() % 1000;
    float power = compute_power(read_voltage());
    float temperature = read_temperature();
    int bit_flips = 1 + rand() % 10;  // 1 to 10 bit flips
    xil_printf("Starting Rowhammer Attack\r\n");
    XGpio_DiscreteWrite(&LED_GPIO, 1, 0x02); // Turn on LED
    enable_attack(0x21);  // Enable rowhammer module
    
    // usleep(1000000); // Simulate 1 second duration

        for (i = 0; i < 20; i++) {
        usleep(200000); // 200ms intervals
        bit_flips = 1 + rand() % 10;
        XGpio_DiscreteWrite(&LED_GPIO, 1, (1 << (i % 8)));
        // status = XGpio_DiscreteRead(&Status_GPIO, 1);
        // xil_printf("Status read: 0x%08x\r\n", status);
    }
    
    enable_attack(0x00);
    XGpio_DiscreteWrite(&LED_GPIO, 1, 0x00);
    
    int severity = compute_severity_rowhammer(bit_flips, temperature);
    
    send_response(
        "Rowhammer attack completed\r\n"
        "Bit flips detected:%d\r\n"
        "Duration:%dms\r\n"
        "Target rows affected:%d\r\n"
        "Power:%.2fmW\r\n"
        "Temperature:%.2f°C\r\n"
        "Severity:%d%%\r\n"
        "<END_OF_OUTPUT>\r\n",
        bit_flips, 
        duration, 
        (bit_flips > 5) ? 3 : 2,  // Now this is the third parameter
        power, 
        temperature, 
        severity
    );
}

// ---------------------------
// Clock Glitching Attack Handler (Modified to include sensor readings)
// ---------------------------
void handle_clock_glitching() {
    int i;
    int timing_violations = 1 + rand() % 5;
    int computation_errors = rand() % 4;
    int duration = 250 + rand() % 100;
    float power = compute_power(read_voltage());
    float temperature = read_temperature();
    int severity = compute_severity_clock_glitching(timing_violations, computation_errors, temperature);
    
    xil_printf("Starting Clock Glitching Attack\r\n");
    XGpio_DiscreteWrite(&LED_GPIO, 1, 0x04);
    enable_attack(0x22);
    
    for (i = 0; i < 5; i++) {
        usleep(50000);
        timing_violations = 1 + rand() % 5;
        computation_errors = rand() % 4;
        XGpio_DiscreteWrite(&LED_GPIO, 1, 0xFF);
        usleep(10000);
        XGpio_DiscreteWrite(&LED_GPIO, 1, 0x00);
        usleep(10000);
    }
    
    enable_attack(0x00);
    XGpio_DiscreteWrite(&LED_GPIO, 1, 0x00);
    
    send_response(
        "Clock glitching attack completed\r\n"
        "Timing violations:%d\r\n"
        "Computation errors:%d\r\n"
        "Duration:%dms\r\n"
        "Power:%.2fmW\r\n"
        "Temperature:%.2f°C\r\n"
        "Severity:%d%%\r\n"
        "Status:%s\r\n"
        "<END_OF_OUTPUT>\r\n",
        timing_violations, computation_errors, duration, power, temperature, severity,
        (computation_errors > 0) ? "Glitch effective" : "No significant effect"
    );
}

void handle_command(u8 command) {
    xil_printf("Received Command: 0x%02X\r\n", command);
    switch(command) {
        case CMD_OVERVOLTAGE:
            xil_printf("Executing Overvoltage Attack\r\n");
            handle_overvoltage();
            break;
        case CMD_ROWHAMMER:
            xil_printf("Executing Rowhammer Attack\r\n");
            handle_rowhammer();
            break;
        case CMD_CLOCK_GLITCHING:
            xil_printf("Executing Clock Glitching Attack\r\n");
            handle_clock_glitching();
            break;
        case CMD_ATTACK_STOP:
            xil_printf("Stopping All Attacks\r\n");
            XGpio_DiscreteWrite(&LED_GPIO, 1, 0x00);
            send_response("Self-healing triggered\r\n<END_OF_OUTPUT>\r\n");
            break;
        case CMD_NORMAL_STATE:
            xil_printf("Providing Normal State Data\r\n");
            handle_normal_state();
            break;
        default:
            xil_printf("Unknown Command: 0x%02X\r\n", command);
            break;
    }
}

int main() {
    int Status;
    int ReceivedCount;
    
    init_platform();
    xil_printf("\n\n*** Hardware Security Attack Framework Starting ***\r\n\n");
    
    srand((unsigned) time(NULL));
    
    Status = initialize_peripherals();
    if (Status != XST_SUCCESS) {
        xil_printf("Peripheral initialization failed\r\n");
        return XST_FAILURE;
    }

    // Initialize system monitor for real sensor readings
    if (restart_sysmon() != XST_SUCCESS) {
        xil_printf("SysMon initialization failed\r\n");
        return XST_FAILURE;
    }
    
    xil_printf("Running LED test pattern...\r\n");
    for (int i = 0; i < 8; i++) {
        xil_printf("Lighting LED %d\r\n", i);
        XGpio_DiscreteWrite(&LED_GPIO, 1, (1 << i));
        usleep(300000);
    }
    
    for (int i = 0; i < 3; i++) {
        XGpio_DiscreteWrite(&LED_GPIO, 1, 0xFF);
        usleep(300000);
        XGpio_DiscreteWrite(&LED_GPIO, 1, 0x00);
        usleep(300000);
    }
    
    xil_printf("Hardware Security Attack Framework Initialized\r\n");
    xil_printf("Ready to receive commands...\r\n");
    
    while (1) {
        ReceivedCount = XUartPs_Recv(&Uart_PS, RecvBuffer, 1);
        if (ReceivedCount > 0) {
            xil_printf("Received command: 0x%02X\r\n", RecvBuffer[0]);
            handle_command(RecvBuffer[0]);
        }
        usleep(10000);
    }
    
    cleanup_platform();
    return 0;
}

int _gettimeofday(struct timeval *tv, void *tz) {
    if (tv) {
        (void)tz;
        tv->tv_sec = 0;
        tv->tv_usec = 0;
    }
    return 0;
}
