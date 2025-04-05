# FFAT: A Semantics-Aware Framework for Generalizable Hardware Fuzzing and Anomaly Detection

FFAT (Formal-Fuzzing Adaptive Tester) is a hardware security framework combining formal verification with runtime fuzzing to discover complex and stealthy vulnerabilities in FPGA-based systems. This hybrid approach enables detection of design-time flaws (using formal tools like SymbiYosys) and runtime anomalies (via feedback-guided fuzzing on real hardware like Xilinx ZCU102).

---

## 📌 Key Features

- 🧠 **Formal Verification Integration**: Uses `SymbiYosys` and `Yosys` to prove design safety and identify vulnerable logic paths.
- ⚡ **Runtime Fuzzing Engine**: Injects dynamic attack patterns (Rowhammer, Overvoltage, Clock Glitching) and observes real-time behavior through UART.
- 🔁 **Adaptive Feedback Loop**: Dynamically adjusts fuzzing strategy based on observed runtime metrics like severity, temperature, and power.
- 💡 **LED-Based Runtime Indicators**: Real-time status shown via onboard LEDs (useful for physical observations or demos).
- 📊 **Live Visualization Dashboard**: Dash + Plotly-based GUI to monitor power, severity, voltage/temperature in real time.
- 📁 **Modular Codebase**: Cleanly separated HDL design, firmware, host scripts, ML models, and attack orchestration.

---

## 📂 Repository Structure
