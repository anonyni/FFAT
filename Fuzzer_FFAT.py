#################################################################################################################################

# Enhanced Python Automation & Visualization for Real-World Hardware Attacks

#####################################################################################################################

import serial
import time
import logging
import numpy as np
import pandas as pd
import re
import threading
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime
import random


# Global flag to control random attack mode
random_mode_active = False

# ----------------------------------------------------------------------------------------
# Critical Adjustments to Mirror Your Working Script
# ----------------------------------------------------------------------------------------
UART_PORT = "COM7"  # Replace with your port
UART_BAUDRATE = 115200
COMMAND_DELAY = 3.0  
RETRY_LIMIT = 3      # <-- We'll retry 3 times if needed

# Attack Commands
ATTACKS = {
    0x20: "Overvoltage",
    0x21: "Rowhammer",
    0x22: "Clock Glitching",
    0x23: "Stop All Attacks",
    0x30: "Normal State"
}

# ----------------------------------------------------------------------------------------
# Data Directories & Structures
# ----------------------------------------------------------------------------------------
data_dir = "attack_data"
os.makedirs(data_dir, exist_ok=True)


normal_state_data = {
    "timestamps": [],
    "voltage": [],
    "threshold": [],
    "duration": [],
    "power": [],
    "temperature": [],
    "severity": []
}


overvoltage_data = {
    "timestamps": [],
    "voltage": [],
    "threshold": [],
    "duration": [],
    "power": [],
    "temperature": [],
    "severity": []
}

rowhammer_data = {
    "timestamps": [],
    "bit_flips": [],
    "durations": [],
    "affected_rows": [],
    "power": [],
    "temperature": [],
    "severity": []
}

clock_glitching_data = {
    "timestamps": [],
    "timing_violations": [],
    "computation_errors": [],
    "durations": [],
    "status": [],
    "power": [],
    "temperature": [],
    "severity": []
}

attack_history = []

# ----------------------------------------------------------------------------------------
# UART Initialization
# ----------------------------------------------------------------------------------------
def initialize_uart():
    """Initialize UART for communication with the FPGA."""
    try:
        uart = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=2)
        logging.info(f"UART connected on {UART_PORT} at {UART_BAUDRATE} baud")
        return uart
    except Exception as e:
        logging.error(f"Failed to initialize UART: {e}")
        return None
    
# ---------------------------
# New function: Random Attack Thread
# ---------------------------
def run_random_attack(uart):
    global random_mode_active
    random_mode_active = True
    attack_codes = [0x20, 0x21, 0x22]  # Overvoltage, Rowhammer, Clock Glitching
    iterations = 0
    MAX_ITERATIONS = 50  # limit iterations

    logging.info("Random attack mode started.")
    while random_mode_active and iterations < 50:
        # Choose a random attack
        code = random.choice(attack_codes)
        logging.info(f"Randomly selected attack code: 0x{code:02X}")
        # Execute the chosen attack
        execute_attack(uart, code)
        # Wait a little before triggering the next attack
        time.sleep(1)  # Adjust delay as needed
    logging.info("Random attack mode stopped.")

# ---------------------------
# Response Parsing
# ---------------------------
def parse_response(response_text):
    """
    Parse structured response from FPGA.
    Returns (attack_type, metrics_dict) or (None, {}).
    """
    attack_type = None
    metrics = {}

    if "Overvoltage attack completed" in response_text:
        attack_type = 0x20
        m_volt    = re.search(r"Voltage:(\d+(\.\d+)?)", response_text)
        m_thresh  = re.search(r"Threshold:(\d+(\.\d+)?)", response_text)
        m_dur     = re.search(r"Duration:(\d+)ms", response_text)
        m_power   = re.search(r"Power:(\d+(\.\d+)?)mW", response_text)
        m_temp    = re.search(r"Temperature:(\d+(\.\d+)?)°C", response_text)
        m_severity= re.search(r"Severity:(\d+)%", response_text)
        if m_volt:
            metrics["voltage"] = float(m_volt.group(1))
        if m_thresh:
            metrics["threshold"] = float(m_thresh.group(1))
        if m_dur:
            metrics["duration"] = int(m_dur.group(1))
        if m_power:
            metrics["power"] = float(m_power.group(1))
        if m_temp:
            metrics["temperature"] = float(m_temp.group(1))
        if m_severity:
            metrics["severity"] = int(m_severity.group(1))

    elif "Rowhammer attack completed" in response_text:
        attack_type = 0x21
        m_bit     = re.search(r"Bit flips detected:(\d+)", response_text)
        m_dur     = re.search(r"Duration:(\d+)ms", response_text)
        m_rows    = re.search(r"Target rows affected:(\d+)", response_text)
        m_power   = re.search(r"Power:(\d+(\.\d+)?)mW", response_text)
        m_temp    = re.search(r"Temperature:(\d+(\.\d+)?)°C", response_text)
        m_severity= re.search(r"Severity:(\d+)%", response_text)
        if m_bit:
            metrics["bit_flips"] = int(m_bit.group(1))
        if m_dur:
            metrics["duration"] = int(m_dur.group(1))
        if m_rows:
            metrics["affected_rows"] = int(m_rows.group(1))
        if m_power:
            metrics["power"] = float(m_power.group(1))
        if m_temp:
            metrics["temperature"] = float(m_temp.group(1))
        if m_severity:
            metrics["severity"] = int(m_severity.group(1))

    elif "Clock glitching attack completed" in response_text:
        attack_type = 0x22
        m_viol    = re.search(r"Timing violations:(\d+)", response_text)
        m_err     = re.search(r"Computation errors:(\d+)", response_text)
        m_dur     = re.search(r"Duration:(\d+)ms", response_text)
        m_power   = re.search(r"Power:(\d+(\.\d+)?)mW", response_text)
        m_temp    = re.search(r"Temperature:(\d+(\.\d+)?)°C", response_text)
        m_severity= re.search(r"Severity:(\d+)%", response_text)
        m_status  = re.search(r"Status:(.+)", response_text)
        if m_viol:
            metrics["timing_violations"] = int(m_viol.group(1))
        if m_err:
            metrics["computation_errors"] = int(m_err.group(1))
        if m_dur:
            metrics["duration"] = int(m_dur.group(1))
        if m_power:
            metrics["power"] = float(m_power.group(1))
        if m_temp:
            metrics["temperature"] = float(m_temp.group(1))
        if m_severity:
            metrics["severity"] = int(m_severity.group(1))
        if m_status:
            metrics["status"] = m_status.group(1)
    elif "Normal state data" in response_text:
        attack_type = 0x30
        m_volt    = re.search(r"Voltage:(\d+(\.\d+)?)", response_text)
        m_thresh  = re.search(r"Threshold:(\d+(\.\d+)?)", response_text)
        m_dur     = re.search(r"Duration:(\d+)ms", response_text)
        m_power   = re.search(r"Power:(\d+(\.\d+)?)mW", response_text)
        m_temp    = re.search(r"Temperature:(\d+(\.\d+)?)°C", response_text)
        m_severity= re.search(r"Severity:(\d+)%", response_text)
        if m_volt:
            metrics["voltage"] = float(m_volt.group(1))
        if m_thresh:
            metrics["threshold"] = float(m_thresh.group(1))
        if m_dur:
            metrics["duration"] = int(m_dur.group(1))
        if m_power:
            metrics["power"] = float(m_power.group(1))
        if m_temp:
            metrics["temperature"] = float(m_temp.group(1))
        if m_severity:
            metrics["severity"] = int(m_severity.group(1))

    
    return attack_type, metrics


# ----------------------------------------------------------------------------------------
# Command Execution with Retry (mirroring your short code)
# ----------------------------------------------------------------------------------------

def read_full_response(uart, timeout=3.0):
    start_time = time.time()
    full_response = ""
    while time.time() - start_time < timeout:
        chunk = uart.read(uart.in_waiting or 1).decode('utf-8', errors='replace')
        if chunk:
            full_response += chunk
            if "<END_OF_OUTPUT>" in full_response:
                break
        else:
            time.sleep(0.999)
    return full_response


def execute_attack(uart, attack_code):
    """
    Sends a single attack command to the FPGA, attempts up to RETRY_LIMIT times,
    parses the complete response, and returns (success_bool, metrics).
    """
    if not uart or not uart.is_open:
        logging.error("UART not available")
        return False, None

    attack_name = ATTACKS.get(attack_code, f"Unknown (0x{attack_code:02X})")
    logging.info(f"Executing attack: {attack_name} (code=0x{attack_code:02X})")

    # Retry loop
    for attempt in range(RETRY_LIMIT):
        uart.reset_input_buffer()  # flush stale data
        
        uart.write(bytes([attack_code]))
        logging.info(f"Sent command byte: 0x{attack_code:02X} (attempt {attempt+1}/{RETRY_LIMIT})")
        
        # Wait a bit (using your COMMAND_DELAY)
        time.sleep(COMMAND_DELAY)
        
        # Accumulate full response over a timeout period:
        response_text = read_full_response(uart, timeout=2.0)
        print(f"[DEBUG RAW] {response_text}")  # Show complete response
        
        parsed_type, metrics = parse_response(response_text)
        logging.info(f"Raw response (first 80 chars): {response_text[:80]}...")
        
        if parsed_type == attack_code:
            logging.info(f"Parsed a valid response for 0x{attack_code:02X}. Attack succeeded.")
            timestamp = time.time()
            store_attack_data(attack_code, metrics, timestamp)
            return True, metrics
        else:
            logging.warning(f"Got response but attack code mismatch or unrecognized. "
                            f"Parsed type={parsed_type}, expected={attack_code}.")
    logging.error(f"Attack command 0x{attack_code:02X} failed after {RETRY_LIMIT} attempts.")
    return False, None

def store_attack_data(attack_code, metrics, timestamp):
    """
    Appends the parsed metrics to the appropriate dictionary 
    and stores a record in the global 'attack_history'.
    """
    attack_name = ATTACKS.get(attack_code, f"Unknown (0x{attack_code:02X})")
    if attack_code == 0x20:
        overvoltage_data["timestamps"].append(timestamp)
        overvoltage_data["voltage"].append(metrics.get("voltage", 0))
        overvoltage_data["threshold"].append(metrics.get("threshold", 0))
        overvoltage_data["duration"].append(metrics.get("duration", 0))
        overvoltage_data["power"].append(metrics.get("power", 0))
        overvoltage_data["temperature"].append(metrics.get("temperature", 0))
        overvoltage_data["severity"].append(metrics.get("severity", 0))
    elif attack_code == 0x21:
        rowhammer_data["timestamps"].append(timestamp)
        rowhammer_data["bit_flips"].append(metrics.get("bit_flips", 0))
        rowhammer_data["durations"].append(metrics.get("duration", 0))
        rowhammer_data["affected_rows"].append(metrics.get("affected_rows", 0))
        rowhammer_data["power"].append(metrics.get("power", 0))
        rowhammer_data["temperature"].append(metrics.get("temperature", 0))
        rowhammer_data["severity"].append(metrics.get("severity", 0))
    elif attack_code == 0x22:
        clock_glitching_data["timestamps"].append(timestamp)
        clock_glitching_data["timing_violations"].append(metrics.get("timing_violations", 0))
        clock_glitching_data["computation_errors"].append(metrics.get("computation_errors", 0))
        clock_glitching_data["durations"].append(metrics.get("duration", 0))
        clock_glitching_data["status"].append(metrics.get("status", "Unknown"))
        clock_glitching_data["power"].append(metrics.get("power", 0))
        clock_glitching_data["temperature"].append(metrics.get("temperature", 0))
        clock_glitching_data["severity"].append(metrics.get("severity", 0))

    elif attack_code == 0x30:
        normal_state_data["timestamps"].append(timestamp)
        normal_state_data["voltage"].append(metrics.get("voltage", 0))
        normal_state_data["threshold"].append(metrics.get("threshold", 0))
        normal_state_data["duration"].append(metrics.get("duration", 0))
        normal_state_data["power"].append(metrics.get("power", 0))
        normal_state_data["temperature"].append(metrics.get("temperature", 0))
        normal_state_data["severity"].append(metrics.get("severity", 0))

    
    # Record in overall history
    attack_history.append({
        "timestamp": timestamp,
        "type": attack_code,
        "name": attack_name,
        "metrics": metrics
    })

    save_attack_data()

# ----------------------------------------------------------------------------------------
# Run a Sequence of Attacks
# ----------------------------------------------------------------------------------------
def run_attack_sequence(uart, attack_code, iterations=10, interval=2.0):
    """
    Run a series of the same attack with specified iterations and interval,
    stopping automatically at the end.
    """
    results = []
    attack_name = ATTACKS.get(attack_code, f"Unknown (0x{attack_code:02X})")
    logging.info(f"Starting {attack_name} attack sequence with {iterations} iterations")

    for i in range(iterations):
        logging.info(f"Attack iteration {i+1}/{iterations}")
        success, metrics = execute_attack(uart, attack_code)
        if success and metrics:
            results.append(metrics)
        else:
            logging.warning("Attack iteration failed or returned no metrics.")
        if i < iterations - 1:
            time.sleep(interval)

    logging.info(f"Attack sequence completed: {attack_name}")
    
    # Finally, send the stop command once after the sequence
    execute_attack(uart, 0x23)
    
    return results

# ----------------------------------------------------------------------------------------
# Save Attack Data
# ----------------------------------------------------------------------------------------

def save_attack_data():
    """
    Save collected attack data to CSV files. 
    Called automatically after each single attack or at user exit.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if overvoltage_data["timestamps"]:
        df = pd.DataFrame({
            "Timestamp": overvoltage_data["timestamps"],
            "Voltage(V)": overvoltage_data["voltage"],
            "Threshold(V)": overvoltage_data["threshold"],
            "Duration(ms)": overvoltage_data["duration"],
            "Power(mW)": overvoltage_data["power"],
            "Temperature(°C)": overvoltage_data["temperature"],
            "Severity(%)": overvoltage_data["severity"]
        })
        df.to_csv(f"{data_dir}/overvoltage_{timestamp}.csv", index=False)
    if rowhammer_data["timestamps"]:
        df = pd.DataFrame({
            "Timestamp": rowhammer_data["timestamps"],
            "Bit Flips": rowhammer_data["bit_flips"],
            "Duration(ms)": rowhammer_data["durations"],
            "Affected Rows": rowhammer_data["affected_rows"],
            "Power(mW)": rowhammer_data["power"],
            "Temperature(°C)": rowhammer_data["temperature"],
            "Severity(%)": rowhammer_data["severity"]
        })
        df.to_csv(f"{data_dir}/rowhammer_{timestamp}.csv", index=False)
    if clock_glitching_data["timestamps"]:
        df = pd.DataFrame({
            "Timestamp": clock_glitching_data["timestamps"],
            "Timing Violations": clock_glitching_data["timing_violations"],
            "Computation Errors": clock_glitching_data["computation_errors"],
            "Duration(ms)": clock_glitching_data["durations"],
            "Status": clock_glitching_data["status"],
            "Power(mW)": clock_glitching_data["power"],
            "Temperature(°C)": clock_glitching_data["temperature"],
            "Severity(%)": clock_glitching_data["severity"]
        })
        df.to_csv(f"{data_dir}/clock_glitching_{timestamp}.csv", index=False)
    if normal_state_data["timestamps"]:
        df = pd.DataFrame({
            "Timestamp": normal_state_data["timestamps"],
            "Voltage(V)": normal_state_data["voltage"],
            "Threshold(V)": normal_state_data["threshold"],
            "Duration(ms)": normal_state_data["duration"],
            "Power(mW)": normal_state_data["power"],
            "Temperature(°C)": normal_state_data["temperature"],
            "Severity(%)": normal_state_data["severity"]
        })
        df.to_csv(f"{data_dir}/normal_state_{timestamp}.csv", index=False)

    # Overall attack history
    if attack_history:
        pd.DataFrame(attack_history).to_csv(
            f"{data_dir}/attack_history_{timestamp}.csv", index=False
        )

# ----------------------------------------------------------------------------------------
# Dash Visualization
# ----------------------------------------------------------------------------------------
def run_dashboard():
    """Create and run a Dash-based visualization dashboard for attack results."""
    app = dash.Dash(__name__, title="Hardware Security Attack Framework")

    tabs_styles = {
        "height": "44px",
        "backgroundColor": "#34495E"
    }
    
    tab_style = {
        "backgroundColor": "#34495E",
        "color": "#BBBBBB",
        "padding": "10px",
        "borderBottom": "1px solid #3498DB"
    }
    
    tab_selected_style = {
        "backgroundColor": "#2C3E50",
        "color": "#FFFFFF",
        "padding": "10px",
        "borderTop": "3px solid #3498DB"
    }
    
    app.layout = html.Div([
        html.Div([
            html.H1("Hardware Security Attack Framework", 
                    style={"textAlign": "center", "color": "#E0E0E0", "marginBottom": "10px"}),
            html.H4("Real-world Attacks on FPGA Hardware",
                    style={"textAlign": "center", "color": "#BBBBBB", "marginTop": "0", "marginBottom": "20px"})
        ], style={"backgroundColor": "#2C3E50", "padding": "20px", "borderRadius": "5px"}),
        
        # Control panel
        html.Div([
            html.H3("Attack Control Panel", style={"color": "#E0E0E0", "marginBottom": "15px"}),
            html.Div([
                html.Button("Overvoltage Attack", id="overvoltage-btn", n_clicks=0,
                            style={"backgroundColor": "#3498DB", "color": "#FFFFFF", "border": "none", 
                                "padding": "10px 20px", "margin": "5px", "borderRadius": "5px"}),
                html.Button("Rowhammer Attack", id="rowhammer-btn", n_clicks=0,
                            style={"backgroundColor": "#E74C3C", "color": "#FFFFFF", "border": "none", 
                                "padding": "10px 20px", "margin": "5px", "borderRadius": "5px"}),
                html.Button("Clock Glitching Attack", id="clock-glitching-btn", n_clicks=0,
                            style={"backgroundColor": "#2ECC71", "color": "#FFFFFF", "border": "none", 
                                "padding": "10px 20px", "margin": "5px", "borderRadius": "5px"}),
                html.Button("Normal Data", id="normal-btn", n_clicks=0,
                            style={"backgroundColor": "#2ECC71", "color": "#FFFFFF", "border": "none", 
                                "padding": "10px 20px", "margin": "5px", "borderRadius": "5px"}),
                # New Random Attack Button:
                html.Button("Random Attack", id="random-attack-btn", n_clicks=0,
                            style={"backgroundColor": "#9B59B6", "color": "#FFFFFF", "border": "none",
                                "padding": "10px 20px", "margin": "5px", "borderRadius": "5px"}),
                html.Button("Stop All Attacks", id="stop-btn", n_clicks=0,
                            style={"backgroundColor": "#95A5A6", "color": "#FFFFFF", "border": "none", 
                                "padding": "10px 20px", "margin": "5px", "borderRadius": "5px"})
            ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"})
        ], style={"backgroundColor": "#34495E", "padding": "20px", "borderRadius": "5px", "marginTop": "15px"}),
        
        # Status panel
        html.Div([
            html.Div(id="attack-status", style={"fontSize": "16px", "color": "#E0E0E0"}),
            html.Div(id="attack-metrics", style={"fontSize": "14px", "color": "#BBBBBB", "marginTop": "10px"})
        ], style={"backgroundColor": "#34495E", "padding": "15px", "borderRadius": "5px", "marginTop": "15px"}),
        
        # Main visualization area with tabs
        html.Div([
            dcc.Tabs([
                # overvoltage Analysis Tab
                dcc.Tab(label="Overvoltage Analysis", children=[
                    html.Div([
                        html.H4("Overvoltage Attack Analysis", 
                               style={"color": "#3498DB", "textAlign": "center", "marginTop": "15px"}),
                        dcc.Graph(id="overvoltage-graph"),
                        html.Div([
                            dcc.Graph(id="overvoltage-hist", style={"width": "50%"}),
                            dcc.Graph(id="overvoltage-heatmap", style={"width": "50%"})
                        ], style={"display": "flex", "flexWrap": "wrap"})
                    ])
                ], style=tab_style, selected_style=tab_selected_style),
                
                # Rowhammer Analysis Tab
                dcc.Tab(label="Rowhammer Analysis", children=[
                    html.Div([
                        html.H4("Rowhammer Attack Analysis", 
                               style={"color": "#E74C3C", "textAlign": "center", "marginTop": "15px"}),
                        dcc.Graph(id="rowhammer-graph"),
                        html.Div([
                            dcc.Graph(id="rowhammer-bitflips-scatter", style={"width": "50%"}),
                            dcc.Graph(id="rowhammer-memory-map", style={"width": "50%"})
                        ], style={"display": "flex", "flexWrap": "wrap"})
                    ])
                ], style=tab_style, selected_style=tab_selected_style),
                
                # Clock Glitching Analysis Tab
                dcc.Tab(label="Clock Glitching Analysis", children=[
                    html.Div([
                        html.H4("Clock Glitching Attack Analysis", 
                               style={"color": "#2ECC71", "textAlign": "center", "marginTop": "15px"}),
                        dcc.Graph(id="clock-glitching-graph"),
                        html.Div([
                            dcc.Graph(id="clock-violations-graph", style={"width": "50%"}),
                            dcc.Graph(id="clock-effects-graph", style={"width": "50%"})
                        ], style={"display": "flex", "flexWrap": "wrap"})
                    ])
                ], style=tab_style, selected_style=tab_selected_style),
                
                # Timeline Analysis
                dcc.Tab(label="Attack Timeline", children=[
                    html.Div([
                        html.H4("Attack Sequence Timeline", 
                               style={"color": "#9B59B6", "textAlign": "center", "marginTop": "15px"}),
                        dcc.Graph(id="attack-timeline-graph"),
                        dcc.Graph(id="attack-comparison-graph")
                    ])
                ], style=tab_style, selected_style=tab_selected_style),
                
                # Advanced Analysis Tab
                dcc.Tab(label="Advanced Analysis", children=[
                    html.Div([
                        html.H4("Advanced Attack Analysis", 
                               style={"color": "#F39C12", "textAlign": "center", "marginTop": "15px"}),
                        dcc.Graph(id="attack-pca-graph"),
                        dcc.Graph(id="attack-correlation-graph")
                    ])
                ], style=tab_style, selected_style=tab_selected_style),
                
            ], style=tabs_styles),
        ], style={"backgroundColor": "#34495E", "padding": "15px", "borderRadius": "5px", "marginTop": "15px"}),
        
        # Update interval
        dcc.Interval(id="update-interval", interval=1000, n_intervals=0)
    ], style={"backgroundColor": "#2C3E50", "padding": "20px", "fontFamily": "Arial"})
    
    # ------------------------------------------------------------------------------------
    # Callbacks for Attack Buttons
    # ------------------------------------------------------------------------------------
    @app.callback(
        [Output("attack-status", "children"),
        Output("attack-metrics", "children")],
        [Input("overvoltage-btn", "n_clicks"),
        Input("rowhammer-btn", "n_clicks"),
        Input("clock-glitching-btn", "n_clicks"),
        Input("normal-btn", "n_clicks"),
        Input("random-attack-btn", "n_clicks"),
        Input("stop-btn", "n_clicks")]
    )
    def handle_attack_buttons(overvoltage_clicks, rowhammer_clicks,
                            clock_glitching_clicks,normal_clicks, random_attack_clicks,
                            stop_clicks):
        global random_mode_active
        ctx = dash.callback_context
        if not ctx.triggered:
            return ["No attack running", ""]
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        uart = initialize_uart()
        if not uart:
            return ["UART connection failed", ""]
        
        try:
            if button_id == "overvoltage-btn" and overvoltage_clicks > 0:
                thread = threading.Thread(target=run_attack_sequence, args=(uart, 0x20, 5, 1.0))
                thread.daemon = True
                thread.start()
                #50
                return ["Running Overvoltage Attack", "Executing iterations..."]
            elif button_id == "rowhammer-btn" and rowhammer_clicks > 0:
                thread = threading.Thread(target=run_attack_sequence, args=(uart, 0x21, 5, 1.5))
                thread.daemon = True
                thread.start()
                #65
                return ["Running Rowhammer Attack", "Executing iterations..."]
            elif button_id == "clock-glitching-btn" and clock_glitching_clicks > 0:
                thread = threading.Thread(target=run_attack_sequence, args=(uart, 0x22, 5, 0.8))
                thread.daemon = True
                thread.start()
                #65
                return ["Running Clock Glitching Attack", "Executing iterations..."]
            elif button_id == "normal-btn" and normal_clicks > 0:
                thread = threading.Thread(target=run_attack_sequence, args=(uart, 0x30, 5 , 1.0))
                thread.daemon = True
                thread.start()
                #32
                return ["Collecting Normal Data", "Executing iterations..."]
            elif button_id == "random-attack-btn" and random_attack_clicks > 0:
                # Start random mode in a separate thread
                thread = threading.Thread(target=run_random_attack, args=(uart, 50, 1.5))
                thread.daemon = True
                thread.start()
                return ["Running Random Attacks", "Random attacks will continue until stopped."]
            elif button_id == "stop-btn" and stop_clicks > 0:
                # Set the flag to stop any ongoing random attacks.
                random_mode_active = False
                execute_attack(uart, 0x23)
                return ["All attacks stopped", ""]
            return ["No attack running", ""]
        finally:
            # Do not close the UART handle here if threads are still using it.
            pass

# ------------------------------------------------------------------------------------
# Overvoltage Visualization Updates (Enhanced with Real-Time Power, Temperature, & Severity)
# ------------------------------------------------------------------------------------
    @app.callback(
        [Output("overvoltage-graph", "figure"),
        Output("overvoltage-hist", "figure"),
        Output("overvoltage-heatmap", "figure")],
        [Input("update-interval", "n_intervals")]
    )
    def update_overvoltage_visuals(n):
        # Convert timestamps to datetime objects
        times = [datetime.fromtimestamp(ts) for ts in overvoltage_data["timestamps"]]

        # Create the time-series graph for Voltage and Threshold
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=overvoltage_data["voltage"],
            mode="lines+markers",
            name="Voltage (V)",
            line=dict(color="#1ABC9C", width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=times,
            y=overvoltage_data["threshold"],
            mode="lines+markers",
            name="Threshold (V)",
            line=dict(color="#2ECC71", dash="dash"),
            marker=dict(size=8)
        ))
        
        # Add a horizontal danger line at 3.9 V (for example)
        if times:
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=times[0],
                x1=times[-1],
                y0=3.9,
                y1=3.9,
                line=dict(color="red", dash="dot")
            )
            fig.add_annotation(
                x=times[-1],
                y=3.9,
                text="Danger Threshold: 3.9 V",
                showarrow=False,
                font=dict(color="red")
            )
        
        # Compute summary statistics for multiple parameters
        if overvoltage_data["voltage"]:
            max_voltage = max(overvoltage_data["voltage"])
            min_voltage = min(overvoltage_data["voltage"])
            avg_voltage = np.mean(overvoltage_data["voltage"])
            avg_power = np.mean(overvoltage_data["power"]) if overvoltage_data["power"] else 0
            avg_temp = np.mean(overvoltage_data["temperature"]) if overvoltage_data["temperature"] else 0
            avg_severity = np.mean(overvoltage_data["severity"]) if overvoltage_data["severity"] else 0
            
            summary_text = (f"Voltage: Max {max_voltage:.2f} V, Min {min_voltage:.2f} V, Avg {avg_voltage:.2f} V; "
                            f"Power: Avg {avg_power:.2f} mW; Temp: Avg {avg_temp:.2f} °C; "
                            f"Severity: Avg {avg_severity:.2f}")
            fig.add_annotation(
                x=times[0],
                y=max_voltage,
                text=summary_text,
                showarrow=False,
                xanchor="left",
                font=dict(color="yellow", size=12)
            )
        
        fig.update_layout(
            title="Overvoltage Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Voltage (V)",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            showlegend=True
        )
        
        # Create a histogram of Voltage readings
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=overvoltage_data["voltage"],
            marker_color="#1ABC9C",
            opacity=0.7,
            nbinsx=10
        ))
        hist_fig.update_layout(
            title="Voltage Distribution",
            xaxis_title="Voltage (V)",
            yaxis_title="Frequency",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0")
        )
        
        # Create a simulated heatmap based on recent Voltage readings
        grid = np.zeros((8, 8))
        recent = overvoltage_data["voltage"][-min(len(overvoltage_data["voltage"]), 10):]
        if recent:
            for i, v in enumerate(recent):
                row = i % 8
                grid[row, :] = v / 5.0  # Scale for visualization
        heat_fig = go.Figure(data=go.Heatmap(
            z=grid,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Voltage Level")
        ))
        heat_fig.update_layout(
            title="Simulated Voltage Heatmap",
            xaxis_title="Sensor Column",
            yaxis_title="Sensor Row",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0")
        )
        
        return fig, hist_fig, heat_fig

# ------------------------------------------------------------------------------------
# Rowhammer Visualization Updates (Enhanced with Extended Metrics)
# ------------------------------------------------------------------------------------
    @app.callback(
        [Output("rowhammer-graph", "figure"),
        Output("rowhammer-bitflips-scatter", "figure"),
        Output("rowhammer-memory-map", "figure")],
        [Input("update-interval", "n_intervals")]
    )
    def update_rowhammer_visuals(n):
        # Convert timestamps to datetime objects for readability
        times = [datetime.fromtimestamp(ts) for ts in rowhammer_data["timestamps"]]
        
        # Create a time-series graph for Bit Flips
        row_series = go.Figure()
        row_series.add_trace(go.Scatter(
            x=times,
            y=rowhammer_data["bit_flips"],
            mode="lines+markers",
            name="Bit Flips",
            line=dict(color="#E74C3C", width=2),
            marker=dict(size=8)
        ))
        
        # Compute summary statistics
        if rowhammer_data["bit_flips"]:
            max_bit = max(rowhammer_data["bit_flips"])
            min_bit = min(rowhammer_data["bit_flips"])
            avg_bit = np.mean(rowhammer_data["bit_flips"])
            avg_power = np.mean(rowhammer_data["power"]) if rowhammer_data["power"] else 0
            avg_temp = np.mean(rowhammer_data["temperature"]) if rowhammer_data["temperature"] else 0
            avg_severity = np.mean(rowhammer_data["severity"]) if rowhammer_data["severity"] else 0
            summary = (f"Bit Flips: Max {max_bit}, Min {min_bit}, Avg {avg_bit:.1f}; "
                    f"Power: {avg_power:.1f} mW; Temp: {avg_temp:.1f} °C; "
                    f"Severity: {avg_severity:.1f}")
            row_series.add_annotation(
                x=times[0],
                y=max_bit,
                text=summary,
                showarrow=False,
                xanchor="left",
                font=dict(color="yellow", size=12)
            )
        
        row_series.update_layout(
            title="Rowhammer Bit Flips Over Time",
            xaxis_title="Time",
            yaxis_title="Number of Bit Flips",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            showlegend=True
        )
        
        # Create a scatter plot: Bit Flips vs. Affected Rows, color-coded by duration
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(
            x=rowhammer_data["bit_flips"],
            y=rowhammer_data["affected_rows"],
            mode="markers",
            marker=dict(
                size=10,
                color=rowhammer_data["durations"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Duration (ms)")
            )
        ))
        scatter.update_layout(
            title="Bit Flips vs. Affected Rows",
            xaxis_title="Number of Bit Flips",
            yaxis_title="Affected Rows",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0")
        )
        
        # Memory map visualization remains similar (simulated)
        rows_val, cols_val = 8, 16
        memory_map = np.zeros((rows_val, cols_val))
        bit_flips = rowhammer_data["bit_flips"]
        if bit_flips:
            target_row = 4
            for i, flips in enumerate(bit_flips[-min(len(bit_flips), 5):]):
                col_offset = (i * 3) % cols_val
                memory_map[target_row, col_offset:col_offset+3] = 0.2
                if flips > 0:
                    upper_flips = flips // 2
                    lower_flips = flips - upper_flips
                    if target_row > 0:
                        for j in range(min(upper_flips, 3)):
                            col = (col_offset + j) % cols_val
                            memory_map[target_row-1, col] = 1.0
                    if target_row < rows_val - 1:
                        for j in range(min(lower_flips, 3)):
                            col = (col_offset + j) % cols_val
                            memory_map[target_row+1, col] = 1.0
        memory_fig = go.Figure(data=go.Heatmap(
            z=memory_map,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Bit Flip Probability")
        ))
        memory_fig.update_layout(
            title="Memory Bit Flip Pattern",
            xaxis_title="Memory Column",
            yaxis_title="Memory Row",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            annotations=[dict(
                x=cols_val // 2,
                y=4,
                text="Hammered Row",
                showarrow=False,
                font=dict(color="#FFFFFF")
            )]
        )
        
        return row_series, scatter, memory_fig

# ------------------------------------------------------------------------------------
# Clock Glitching Visualization Updates (Enhanced with Extended Metrics)
# ------------------------------------------------------------------------------------
    @app.callback(
        [Output("clock-glitching-graph", "figure"),
        Output("clock-violations-graph", "figure"),
        Output("clock-effects-graph", "figure")],
        [Input("update-interval", "n_intervals")]
    )
    def update_clock_glitching_visuals(n):
        # Convert timestamps to datetime objects
        times = [datetime.fromtimestamp(ts) for ts in clock_glitching_data["timestamps"]]
        
        # Create a combined time-series for Timing Violations and Computation Errors
        clk_series = go.Figure()
        clk_series.add_trace(go.Scatter(
            x=times,
            y=clock_glitching_data["timing_violations"],
            mode="lines+markers",
            name="Timing Violations",
            line=dict(color="#2ECC71", width=2),
            marker=dict(size=8)
        ))
        clk_series.add_trace(go.Scatter(
            x=times,
            y=clock_glitching_data["computation_errors"],
            mode="lines+markers",
            name="Computation Errors",
            line=dict(color="#F39C12", width=2),
            marker=dict(size=8)
        ))
        
        # Compute summary statistics for clock glitching
        if clock_glitching_data["timing_violations"]:
            avg_viol = np.mean(clock_glitching_data["timing_violations"])
            avg_err = np.mean(clock_glitching_data["computation_errors"])
            avg_power = np.mean(clock_glitching_data["power"]) if clock_glitching_data["power"] else 0
            avg_temp = np.mean(clock_glitching_data["temperature"]) if clock_glitching_data["temperature"] else 0
            avg_severity = np.mean(clock_glitching_data["severity"]) if clock_glitching_data["severity"] else 0
            summary_text = (f"Avg Violations: {avg_viol:.1f}, Avg Errors: {avg_err:.1f}; "
                            f"Power: {avg_power:.1f} mW, Temp: {avg_temp:.1f} °C, Severity: {avg_severity:.1f}")
            clk_series.add_annotation(
                x=times[0],
                y=max(clock_glitching_data["timing_violations"]),
                text=summary_text,
                showarrow=False,
                xanchor="left",
                font=dict(color="yellow", size=12)
            )
        
        clk_series.update_layout(
            title="Clock Glitching Effects Over Time",
            xaxis_title="Time",
            yaxis_title="Count",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            showlegend=True
        )
        
        # Violations waveform visualization (similar to previous)
        violations_fig = go.Figure()
        if clock_glitching_data["timing_violations"]:
            t_normal = np.linspace(0, 10, 1000)
            normal_clock = np.sin(t_normal * 2 * np.pi)
            t_glitched = np.linspace(0, 10, 1000)
            glitched_clock = np.sin(t_glitched * 2 * np.pi)
            for i in range(1, 10, 2):
                idx_start = int(i / 10 * 1000)
                idx_end = int((i + 0.2) / 10 * 1000)
                t_segment = np.linspace(0, 4, idx_end - idx_start)
                glitched_clock[idx_start:idx_end] = np.sin(t_segment * 2 * np.pi)
            violations_fig.add_trace(go.Scatter(
                x=t_normal,
                y=normal_clock,
                mode="lines",
                name="Normal Clock",
                line=dict(color="#3498DB", width=2)
            ))
            violations_fig.add_trace(go.Scatter(
                x=t_glitched,
                y=glitched_clock,
                mode="lines",
                name="Glitched Clock",
                line=dict(color="#E74C3C", width=2)
            ))
            for i in range(1, 10, 2):
                violations_fig.add_trace(go.Scatter(
                    x=[(i / 10) * 10],
                    y=[0],
                    mode="markers",
                    marker=dict(size=10, color="#F39C12", symbol="x"),
                    name="Violation" if i == 1 else "",
                    showlegend=(i == 1)
                ))
        violations_fig.update_layout(
            title="Clock Glitching Waveform Visualization",
            xaxis_title="Time",
            yaxis_title="Clock Signal",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            showlegend=True
        )
        
        # Effects visualization: Expected vs. Glitched Results
        effects_fig = go.Figure()
        if clock_glitching_data["computation_errors"]:
            x_values = np.arange(10)
            expected_values = x_values * 2
            error_values = expected_values.copy()
            error_count = min(sum(clock_glitching_data["computation_errors"]), 5)
            if error_count > 0:
                error_positions = np.random.choice(10, error_count, replace=False)
                for pos in error_positions:
                    error_values[pos] += np.random.randint(-3, 4)
            effects_fig.add_trace(go.Scatter(
                x=x_values,
                y=expected_values,
                mode="lines+markers",
                name="Expected Result",
                line=dict(color="#3498DB", width=2),
                marker=dict(size=8)
            ))
            effects_fig.add_trace(go.Scatter(
                x=x_values,
                y=error_values,
                mode="lines+markers",
                name="Glitched Result",
                line=dict(color="#E74C3C", width=2),
                marker=dict(size=8)
            ))
        effects_fig.update_layout(
            title="Impact of Clock Glitching on Computation",
            xaxis_title="Input Value",
            yaxis_title="Computation Result",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            showlegend=True
        )
        
        return clk_series, violations_fig, effects_fig

    
    # ------------------------------------------------------------------------------------
    # Timeline & Comparison Visualization
    # ------------------------------------------------------------------------------------
    @app.callback(
        [Output("attack-timeline-graph", "figure"),
         Output("attack-comparison-graph", "figure")],
        [Input("update-interval", "n_intervals")]
    )
    def update_timeline_visuals(n):
        timeline_fig = go.Figure()
        if attack_history:
            timestamps = [entry["timestamp"] for entry in attack_history]
            attack_types = [entry["type"] for entry in attack_history]
            attack_names = [entry["name"] for entry in attack_history]
            
            colors = {
                0x20: "#3498DB",  # overvoltage
                0x21: "#E74C3C",  # Rowhammer
                0x22: "#2ECC71",  # Clock Glitching
                0x23: "#95A5A6"   # Stop
            }
            
            color_list = [colors.get(a_type, "#FFFFFF") for a_type in attack_types]
            timeline_fig.add_trace(go.Scatter(
                x=timestamps,
                y=attack_types,
                mode="markers",
                marker=dict(
                    size=12,
                    color=color_list,
                    symbol="circle"
                ),
                text=attack_names,
                hoverinfo="text+x"
            ))
        
        timeline_fig.update_layout(
            title="Attack Timeline",
            xaxis_title="Time",
            yaxis_title="Attack Type",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            yaxis=dict(
                tickvals=[0x20, 0x21, 0x22, 0x23],
                ticktext=["Overvoltage", "Rowhammer", "Clock Glitching", "Stop"]
            )
        )
        
        comparison_fig = go.Figure()
        attack_counts = {
            "Overvoltage": len(overvoltage_data["timestamps"]),
            "Rowhammer": len(rowhammer_data["timestamps"]),
            "Clock Glitching": len(clock_glitching_data["timestamps"])
        }
        
        avg_metrics = {
            "Overvoltage": np.mean(overvoltage_data["voltage"]) if overvoltage_data["voltage"] else 0,
            "Rowhammer": np.mean(rowhammer_data["bit_flips"]) if rowhammer_data["bit_flips"] else 0,
            "Clock Glitching": np.mean(clock_glitching_data["computation_errors"]) if clock_glitching_data["computation_errors"] else 0
        }
        
        comparison_fig.add_trace(go.Bar(
            x=list(attack_counts.keys()),
            y=list(attack_counts.values()),
            name="Attack Count",
            marker_color="#3498DB"
        ))
        comparison_fig.add_trace(go.Bar(
            x=list(avg_metrics.keys()),
            y=list(avg_metrics.values()),
            name="Avg. Impact",
            marker_color="#E74C3C"
        ))
        
        comparison_fig.update_layout(
            title="Attack Comparison",
            xaxis_title="Attack Type",
            yaxis_title="Count / Impact",
            barmode="group",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            showlegend=True
        )
        
        return timeline_fig, comparison_fig
    
    # ------------------------------------------------------------------------------------
    # Advanced Analysis Visualization
    # ------------------------------------------------------------------------------------
    @app.callback(
        [Output("attack-pca-graph", "figure"),
         Output("attack-correlation-graph", "figure")],
        [Input("update-interval", "n_intervals")]
    )
    def update_advanced_analysis(n):
        pca_fig = go.Figure()
        
        # Gather data for PCA
        data_points = []
        labels = []
        
        # Overvoltage
        for i in range(len(overvoltage_data["timestamps"])):
            data_points.append([
                overvoltage_data["voltage"][i],
                overvoltage_data["duration"][i],
                0,  # dummy for bit_flips
                0   # dummy for timing_violations
                    ])
            labels.append("Overvoltage")
        # Rowhammer
        for i in range(len(rowhammer_data["timestamps"])):
            data_points.append([
                0,  # corruption
                rowhammer_data["durations"][i],
                rowhammer_data["bit_flips"][i],
                0   # timing_violations
            ])
            labels.append("Rowhammer")
        
        # Clock Glitching
        for i in range(len(clock_glitching_data["timestamps"])):
            data_points.append([
                0,  # corruption
                clock_glitching_data["durations"][i],
                0,  # bit_flips
                clock_glitching_data["timing_violations"][i]
            ])
            labels.append("Clock Glitching")
        
        if data_points:
            if len(data_points) >= 3:
                X = np.array(data_points)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                for attack_type in set(labels):
                    indices = [i for i, label in enumerate(labels) if label == attack_type]
                    color = "#3498DB" if attack_type == "Overvoltage" else \
                            "#E74C3C" if attack_type == "Rowhammer" else "#2ECC71"
                    
                    pca_fig.add_trace(go.Scatter(
                        x=X_pca[indices, 0],
                        y=X_pca[indices, 1],
                        mode="markers",
                        name=attack_type,
                        marker=dict(size=10, color=color)
                    ))
        
        pca_fig.update_layout(
            title="Principal Component Analysis of Attack Patterns",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0"),
            showlegend=True
        )
        
        corr_fig = go.Figure()
        if data_points:
            feature_names = ["Voltage", "Duration", "Bit Flips", "Timing Violations"]
            X = np.array(data_points)
            corr_matrix = np.corrcoef(X.T)
            corr_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=feature_names,
                y=feature_names,
                colorscale="Viridis",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation")
            ))
        
        corr_fig.update_layout(
            title="Correlation Between Attack Metrics",
            plot_bgcolor="#2C3E50",
            paper_bgcolor="#34495E",
            font=dict(color="#E0E0E0")
        )
        
        return pca_fig, corr_fig
    
    # ------------------------------------------------------------------------------------
    # Run Dashboard
    # ------------------------------------------------------------------------------------
    app.run_server(debug=False)

# ----------------------------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------------------------
def main():
    """Main entry point for the attack framework."""
    logging.info("Initializing Hardware Security Attack Framework")
    
    # Start dashboard in a separate thread
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    logging.info("Attack visualization dashboard started at http://127.0.0.1:8050")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        logging.info("Attack framework terminated by user")
        save_attack_data()  # Save final data
    
    logging.info("Hardware Security Attack Framework terminated")


if __name__ == "__main__":
    main()
