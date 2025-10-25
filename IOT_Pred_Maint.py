import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

# --- Configuration ---
NUM_PUMPS = 5
SIMULATION_INTERVAL_SECONDS = 2
ANOMALY_THRESHOLD_VIBRATION = 1.5  # Multiplier above baseline for anomaly
ANOMALY_THRESHOLD_TEMP = 5       # Degrees C above baseline for anomaly
TEMP_CRITICAL_THRESHOLD = 85     # Critical temperature for high failure risk
PROB_INCREASE_PER_ANOMALY = 0.15 # How much probability increases per detected anomaly

# --- Synthetic Data Generation Functions ---
def generate_baseline_data():
    """Generates a realistic baseline for each pump."""
    baselines = {}
    for i in range(1, NUM_PUMPS + 1):
        baselines[f'Pump {i}'] = {
            'vibration': np.random.uniform(0.5, 2.0), # Units (e.g., mm/s)
            'temperature': np.random.uniform(30.0, 50.0), # Degrees C
            'current': np.random.uniform(10.0, 25.0) # Amps
        }
    return baselines

def generate_sensor_data(pump_id, baselines, current_state):
    """Generates new sensor data based on baseline and current state."""
    baseline = baselines[pump_id]
    data = {
        'timestamp': pd.Timestamp.now(),
        'vibration': np.random.normal(baseline['vibration'], 0.1),
        'temperature': np.random.normal(baseline['temperature'], 0.5),
        'current': np.random.normal(baseline['current'], 0.2)
    }

    # Introduce anomalies periodically for demonstration
    if current_state.get(pump_id, {}).get('inject_anomaly', False):
        if np.random.rand() < 0.6: # 60% chance to increase anomaly
            data['vibration'] *= np.random.uniform(ANOMALY_THRESHOLD_VIBRATION, ANOMALY_THRESHOLD_VIBRATION + 0.5)
            data['temperature'] += np.random.uniform(ANOMALY_THRESHOLD_TEMP, ANOMALY_THRESHOLD_TEMP + 3)
            data['current'] *= np.random.uniform(1.2, 1.5)
    else: # 5% chance to inject a new anomaly
        if np.random.rand() < 0.05:
            st.session_state.pump_states[pump_id]['inject_anomaly'] = True
            st.toast(f"Injecting anomaly into {pump_id}!", icon="⚠️")

    return data

# --- Anomaly Detection and Prediction Logic ---
def detect_anomalies(pump_id, current_data, baselines):
    """Simple anomaly detection based on thresholds."""
    baseline = baselines[pump_id]
    anomalies = []

    if current_data['vibration'] > baseline['vibration'] * ANOMALY_THRESHOLD_VIBRATION:
        anomalies.append('High Vibration')
    if current_data['temperature'] > baseline['temperature'] + ANOMALY_THRESHOLD_TEMP:
        anomalies.append('High Temperature')
    if current_data['current'] > baseline['current'] * 1.2: # Simple current anomaly
        anomalies.append('High Current Draw')

    return anomalies

def predict_failure_likelihood(pump_id, anomalies, current_data, pump_state):
    """
    Predicts failure likelihood based on anomalies and current state.
    This is a very simplified model for simulation purposes.
    """
    prob_of_failure = pump_state.get('prob_of_failure', 0.0)
    time_to_failure = pump_state.get('time_to_failure', 'N/A')
    recommendation = "Normal Operation"

    if anomalies:
        # Increase probability for each anomaly detected
        prob_of_failure = min(1.0, prob_of_failure + PROB_INCREASE_PER_ANOMALY * len(anomalies))
        recommendation = "Investigate Anomalies"

    if current_data['temperature'] > TEMP_CRITICAL_THRESHOLD:
        prob_of_failure = min(1.0, prob_of_failure + 0.2) # Further increase for critical temp
        recommendation = "URGENT: Inspect Cooling System"
        time_to_failure = "Immediate"
        st.session_state.pump_states[pump_id]['inject_anomaly'] = True # Keep anomaly active

    # Simplified time to failure based on probability
    if prob_of_failure > 0.8:
        time_to_failure = "Within 24 Hours"
        recommendation = "URGENT: Schedule Immediate Maintenance"
    elif prob_of_failure > 0.5:
        time_to_failure = "Within 3 Days"
        if recommendation == "Normal Operation": # Don't overwrite more specific reco
            recommendation = "Schedule Proactive Maintenance"
    elif prob_of_failure > 0.2:
        time_to_failure = "Within 7 Days"
        if recommendation == "Normal Operation":
            recommendation = "Monitor Closely"

    # Reset inject_anomaly flag if probability drops significantly (e.g., 'maintenance' applied)
    if prob_of_failure < 0.1 and pump_state.get('inject_anomaly', False):
        st.session_state.pump_states[pump_id]['inject_anomaly'] = False
        st.toast(f"{pump_id} returning to normal operation.", icon="✅")

    return prob_of_failure, time_to_failure, recommendation

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Industrial AI Predictive Maintenance (Opal Sim)")

st.title("🏭 Industrial AI Predictive Maintenance (Google Opal Simulation)")
st.markdown("""
This simulation demonstrates a simplified version of a Google Opal-like AI mini-app for industrial predictive maintenance.
It generates synthetic sensor data, detects anomalies, predicts failure likelihood, and suggests maintenance actions.
""")

# Initialize session state for pump data and statuses
if 'pump_data' not in st.session_state:
    st.session_state.pump_data = {f'Pump {i}': pd.DataFrame(columns=['timestamp', 'vibration', 'temperature', 'current']) for i in range(1, NUM_PUMPS + 1)}
    st.session_state.pump_baselines = generate_baseline_data()
    st.session_state.pump_states = {f'Pump {i}': {'prob_of_failure': 0.0, 'time_to_failure': 'N/A', 'recommendation': 'Normal Operation', 'inject_anomaly': False} for i in range(1, NUM_PUMPS + 1)}
    st.session_state.stop_simulation = False

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Simulation Controls")
    # Use a placeholder for the start button to allow continuous updates
    start_button_placeholder = st.empty()
    if start_button_placeholder.button("Start Simulation", type="primary", disabled=not st.session_state.stop_simulation):
        st.session_state.stop_simulation = False
        st.rerun() # Trigger rerun to start the loop
    if st.button("Stop Simulation", disabled=st.session_state.stop_simulation):
        st.session_state.stop_simulation = True
        st.toast("Simulation stopped.", icon="🛑")
    if st.button("Reset Simulation"):
        st.session_state.pump_data = {f'Pump {i}': pd.DataFrame(columns=['timestamp', 'vibration', 'temperature', 'current']) for i in range(1, NUM_PUMPS + 1)}
        st.session_state.pump_baselines = generate_baseline_data()
        st.session_state.pump_states = {f'Pump {i}': {'prob_of_failure': 0.0, 'time_to_failure': 'N/A', 'recommendation': 'Normal Operation', 'inject_anomaly': False} for i in range(1, NUM_PUMPS + 1)}
        st.session_state.stop_simulation = False
        st.toast("Simulation reset!", icon="🔄")
        st.rerun() # Trigger rerun to apply reset

with col2:
    st.subheader("Current Pump Status Overview")
    pump_status_df = pd.DataFrame(st.session_state.pump_states).T
    pump_status_df.index.name = 'Pump ID'
    # Apply color coding for probability
    def color_prob(val):
        color = 'lightgreen'
        if val > 0.8: color = 'salmon'
        elif val > 0.5: color = 'orange'
        elif val > 0.2: color = 'yellow'
        return f'background-color: {color}'

    # Use a placeholder for dynamic updates to the dataframe
    status_df_placeholder = st.empty()
    status_df_placeholder.dataframe(pump_status_df.drop(columns=['inject_anomaly']).style.applymap(color_prob, subset=['prob_of_failure']), use_container_width=True)


st.subheader("Detailed Pump Insights & Sensor Data")

# Use placeholders for tab content to allow dynamic updates
tab_placeholders = {f'Pump {i}': st.empty() for i in range(1, NUM_PUMPS + 1)}

# Create tabs for each pump
tabs = st.tabs([f'Pump {i}' for i in range(1, NUM_PUMPS + 1)])

for i, tab in enumerate(tabs):
    pump_id = f'Pump {i+1}'
    with tab:
        st.write(f"### {pump_id} Health Status")
        current_pump_state = st.session_state.pump_states[pump_id]
        st.metric("Probability of Failure", f"{current_pump_state['prob_of_failure']:.1%}")
        st.metric("Estimated Time to Failure", current_pump_state['time_to_failure'])
        st.metric("Recommended Action", current_pump_state['recommendation'])

        st.write("---")
        st.write(f"#### {pump_id} Sensor Data Trends")
        if not st.session_state.pump_data[pump_id].empty:
            df_pump = st.session_state.pump_data[pump_id]

            # Plot Vibration
            fig_vibration = px.line(df_pump, x='timestamp', y='vibration', title=f'{pump_id} Vibration')
            fig_vibration.add_hline(y=st.session_state.pump_baselines[pump_id]['vibration'] * ANOMALY_THRESHOLD_VIBRATION,
                                    line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            st.plotly_chart(fig_vibration, use_container_width=True)

            # Plot Temperature
            fig_temp = px.line(df_pump, x='timestamp', y='temperature', title=f'{pump_id} Temperature')
            fig_temp.add_hline(y=st.session_state.pump_baselines[pump_id]['temperature'] + ANOMALY_THRESHOLD_TEMP,
                                line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            fig_temp.add_hline(y=TEMP_CRITICAL_THRESHOLD, line_dash="dot", line_color="purple", annotation_text="Critical Temp")
            st.plotly_chart(fig_temp, use_container_width=True)

            # Plot Current
            fig_current = px.line(df_pump, x='timestamp', y='current', title=f'{pump_id} Current Draw')
            fig_current.add_hline(y=st.session_state.pump_baselines[pump_id]['current'] * 1.2,
                                  line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            st.plotly_chart(fig_current, use_container_width=True)

            st.write("Raw Data (Last 10 entries):")
            st.dataframe(df_pump.tail(10))
        else:
            st.info("No data yet. Start the simulation!")

        # Manual intervention for simulation (mimics a technician fixing a pump)
        if st.button(f"Simulate Maintenance for {pump_id}", key=f'maintain_{pump_id}'):
            st.session_state.pump_states[pump_id]['prob_of_failure'] = 0.0
            st.session_state.pump_states[pump_id]['time_to_failure'] = 'N/A'
            st.session_state.pump_states[pump_id]['recommendation'] = 'Maintenance Performed - Monitoring'
            st.session_state.pump_states[pump_id]['inject_anomaly'] = False # Stop injecting anomalies
            # Reset baseline slightly to simulate repair
            st.session_state.pump_baselines[pump_id] = {
                'vibration': np.random.uniform(0.5, 2.0),
                'temperature': np.random.uniform(30.0, 50.0),
                'current': np.random.uniform(10.0, 25.0)
            }
            st.toast(f"Maintenance simulated for {pump_id}. State reset!", icon="🛠️")
            st.rerun() # Trigger rerun to show updated state

# --- Simulation Loop ---
# This loop will continuously run as long as stop_simulation is False
# and will trigger reruns to update the UI
if not st.session_state.stop_simulation:
    # Update data for all pumps
    for pump_id in st.session_state.pump_data.keys():
        new_data = generate_sensor_data(pump_id, st.session_state.pump_baselines, st.session_state.pump_states)
        current_df = st.session_state.pump_data[pump_id]
        updated_df = pd.concat([current_df, pd.DataFrame([new_data])], ignore_index=True)
        # Keep only last X points for performance
        st.session_state.pump_data[pump_id] = updated_df.tail(100) # Keep 100 data points

        # Perform anomaly detection and prediction
        anomalies = detect_anomalies(pump_id, new_data, st.session_state.pump_baselines)
        prob, ttf, reco = predict_failure_likelihood(pump_id, anomalies, new_data, st.session_state.pump_states[pump_id])

        st.session_state.pump_states[pump_id]['anomalies'] = anomalies
        st.session_state.pump_states[pump_id]['prob_of_failure'] = prob
        st.session_state.pump_states[pump_id]['time_to_failure'] = ttf
        st.session_state.pump_states[pump_id]['recommendation'] = reco

    # This sleep combined with st.rerun() creates the continuous update effect
    time.sleep(SIMULATION_INTERVAL_SECONDS)
    st.rerun() # Use st.rerun() to trigger the next iteration