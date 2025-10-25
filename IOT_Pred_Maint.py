import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

# --- Game Configuration ---
NUM_PUMPS = 5
SIMULATION_INTERVAL_SECONDS = 0.1 # Small delay for "Next Round" click

# --- Financial Rules ---
STARTING_CASH = 50000
REVENUE_PER_PUMP_ROUND = 1000
COST_PROACTIVE_MAINT = 5000
COST_CATASTROPHE = 25000

# --- Gameplay Rules ---
DOWNTIME_PROACTIVE_MAINT = 1 # Rounds
DOWNTIME_CATASTROPHE = 3     # Rounds
CATASTROPHE_THRESHOLD = 0.9  # 90% probability

# --- Sensor Simulation Configuration ---
ANOMALY_THRESHOLD_VIBRATION = 1.5
ANOMALY_THRESHOLD_TEMP = 5
TEMP_CRITICAL_THRESHOLD = 85
PROB_INCREASE_PER_ANOMALY = 0.15

# --- Synthetic Data Generation Functions ---
def generate_baseline_data():
    """Generates a realistic baseline for each pump."""
    baselines = {}
    for i in range(1, NUM_PUMPS + 1):
        baselines[f'Pump {i}'] = {
            'vibration': np.random.uniform(0.5, 2.0),
            'temperature': np.random.uniform(30.0, 50.0),
            'current': np.random.uniform(10.0, 25.0)
        }
    return baselines

def generate_sensor_data(pump_id, baselines, current_pump_state):
    """Generates new sensor data based on baseline and current state."""
    baseline = baselines[pump_id]
    data = {
        'timestamp': pd.Timestamp.now(),
        'vibration': np.random.normal(baseline['vibration'], 0.1),
        'temperature': np.random.normal(baseline['temperature'], 0.5),
        'current': np.random.normal(baseline['current'], 0.2)
    }

    # Introduce anomalies
    if current_pump_state.get('inject_anomaly', False):
        if np.random.rand() < 0.6:
            data['vibration'] *= np.random.uniform(ANOMALY_THRESHOLD_VIBRATION, ANOMALY_THRESHOLD_VIBRATION + 0.5)
            data['temperature'] += np.random.uniform(ANOMALY_THRESHOLD_TEMP, ANOMALY_THRESHOLD_TEMP + 3)
            data['current'] *= np.random.uniform(1.2, 1.5)
    else: 
        if np.random.rand() < 0.05: # 5% chance to inject a new anomaly
            current_pump_state['inject_anomaly'] = True
            st.toast(f"Anomaly detected in {pump_id}!", icon="⚠️")

    return data, current_pump_state

# --- Anomaly Detection and Prediction Logic ---
def detect_anomalies(pump_id, current_data, baselines):
    """Simple anomaly detection based on thresholds."""
    baseline = baselines[pump_id]
    anomalies = []
    if current_data['vibration'] > baseline['vibration'] * ANOMALY_THRESHOLD_VIBRATION:
        anomalies.append('High Vibration')
    if current_data['temperature'] > baseline['temperature'] + ANOMALY_THRESHOLD_TEMP:
        anomalies.append('High Temperature')
    if current_data['current'] > baseline['current'] * 1.2:
        anomalies.append('High Current Draw')
    return anomalies

def predict_failure_likelihood(pump_id, anomalies, current_data, pump_state):
    """Predicts failure likelihood based on anomalies and current state."""
    prob_of_failure = pump_state.get('prob_of_failure', 0.0)
    time_to_failure = pump_state.get('time_to_failure', 'N/A')
    recommendation = "Normal Operation"

    if anomalies:
        prob_of_failure = min(1.0, prob_of_failure + PROB_INCREASE_PER_ANOMALY * len(anomalies))
        recommendation = "Investigate Anomalies"
    
    if current_data['temperature'] > TEMP_CRITICAL_THRESHOLD:
        prob_of_failure = min(1.0, prob_of_failure + 0.2)
        recommendation = "URGENT: Inspect Cooling System"
        time_to_failure = "Immediate"
        pump_state['inject_anomaly'] = True

    if prob_of_failure > 0.8:
        time_to_failure = "Within 24 Hours"
        recommendation = "URGENT: Schedule Immediate Maintenance"
    elif prob_of_failure > 0.5:
        time_to_failure = "Within 3 Days"
        if recommendation == "Normal Operation": 
            recommendation = "Schedule Proactive Maintenance"
    elif prob_of_failure > 0.2:
        time_to_failure = "Within 7 Days"
        if recommendation == "Normal Operation":
            recommendation = "Monitor Closely"

    if prob_of_failure < 0.1 and pump_state.get('inject_anomaly', False):
        pump_state['inject_anomaly'] = False
        st.toast(f"{pump_id} returning to normal operation.", icon="✅")

    # Update the pump_state dictionary directly
    pump_state['prob_of_failure'] = prob_of_failure
    pump_state['time_to_failure'] = time_to_failure
    pump_state['recommendation'] = recommendation
    return pump_state

# --- New Game Logic Functions ---

def initialize_game():
    """Sets up the session state for the single player."""
    st.session_state.game_state = {'round': 0, 'running': False}
    
    pump_states = {}
    pump_data_history = {}
    pump_downtime = {}
    for i in range(1, NUM_PUMPS + 1):
        pump_id = f'Pump {i}'
        pump_states[pump_id] = {
            'prob_of_failure': 0.0, 
            'time_to_failure': 'N/A', 
            'recommendation': 'Normal Operation', 
            'inject_anomaly': False
        }
        pump_data_history[pump_id] = pd.DataFrame(columns=['timestamp', 'vibration', 'temperature', 'current'])
        pump_downtime[pump_id] = 0 # Rounds of downtime remaining
        
    st.session_state.player_data = {
        'net_profit': STARTING_CASH,
        'revenue': 0,
        'maint_costs': 0,
        'failure_costs': 0,
        'pump_states': pump_states,
        'pump_baselines': generate_baseline_data(),
        'pump_data_history': pump_data_history,
        'pump_downtime': pump_downtime
    }
    st.toast("Game Initialized! Good luck!", icon="🎉")

def reset_pump_state(pump_id):
    """Resets a single pump's state after maintenance."""
    player_data = st.session_state.player_data
    player_data['pump_states'][pump_id] = {
        'prob_of_failure': 0.0, 
        'time_to_failure': 'N/A', 
        'recommendation': 'Maintenance Performed - Monitoring', 
        'inject_anomaly': False
    }
    # Reset baseline slightly to simulate repair
    player_data['pump_baselines'][pump_id] = {
        'vibration': np.random.uniform(0.5, 2.0),
        'temperature': np.random.uniform(30.0, 50.0),
        'current': np.random.uniform(10.0, 25.0)
    }

def apply_proactive_maintenance(pump_id):
    """Player action to apply proactive maintenance."""
    player_data = st.session_state.player_data
    
    # Apply cost
    player_data['maint_costs'] += COST_PROACTIVE_MAINT
    
    # Set downtime
    player_data['pump_downtime'][pump_id] = DOWNTIME_PROACTIVE_MAINT
    
    # Reset the pump's state
    reset_pump_state(pump_id)
    
    st.toast(f"Proactive Maint. for {pump_id}! 1 Round Downtime.", icon="🛠️")
    st.rerun()

def run_simulation_round():
    """Runs one round of the game for the player."""
    st.session_state.game_state['round'] += 1
    player_data = st.session_state.player_data
    
    for i in range(1, NUM_PUMPS + 1):
        pump_id = f'Pump {i}'
        
        # 1. Check Downtime
        if player_data['pump_downtime'][pump_id] > 0:
            player_data['pump_downtime'][pump_id] -= 1
            if player_data['pump_downtime'][pump_id] == 0:
                st.toast(f"{pump_id} is back online!", icon="✅")
            else:
                st.toast(f"{pump_id} is in maintenance ({player_data['pump_downtime'][pump_id]} rounds left).", icon="⏳")
            continue # Skip this pump for this round
        
        # 2. Earn Revenue
        player_data['revenue'] += REVENUE_PER_PUMP_ROUND
        
        # 3. Simulate Sensor Data
        current_pump_state = player_data['pump_states'][pump_id]
        baselines = player_data['pump_baselines']
        
        new_data, current_pump_state = generate_sensor_data(pump_id, baselines, current_pump_state)
        
        # 4. Update Data History
        current_df = player_data['pump_data_history'][pump_id]
        updated_df = pd.concat([current_df, pd.DataFrame([new_data])], ignore_index=True)
        player_data['pump_data_history'][pump_id] = updated_df.tail(100)
        
        # 5. Detect & Predict
        anomalies = detect_anomalies(pump_id, new_data, baselines)
        current_pump_state = predict_failure_likelihood(pump_id, anomalies, new_data, current_pump_state)
        
        # 6. Check for Catastrophic Failure
        if current_pump_state['prob_of_failure'] > CATASTROPHE_THRESHOLD:
            st.error(f"CATASTROPHE! {pump_id} has failed!")
            st.balloons()
            
            # Apply cost
            player_data['failure_costs'] += COST_CATASTROPHE
            
            # Set downtime
            player_data['pump_downtime'][pump_id] = DOWNTIME_CATASTROPHE
            
            # Reset pump state
            reset_pump_state(pump_id)
        
        # Save the updated state
        player_data['pump_states'][pump_id] = current_pump_state

    # 7. Recalculate Net Profit
    player_data['net_profit'] = STARTING_CASH + player_data['revenue'] - player_data['maint_costs'] - player_data['failure_costs']

    time.sleep(SIMULATION_INTERVAL_SECONDS) # Add a small delay for dramatic effect
    st.rerun()

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Factory Ops Challenge")

st.title("🏭 Factory Ops Challenge (Single Player)")
st.markdown("Your goal is to run the factory for as many rounds as possible and **maximize your Net Profit**. Decide when to perform proactive maintenance to avoid costly catastrophic failures!")

# Initialize game state
if 'player_data' not in st.session_state:
    initialize_game()

# --- Player Controls Sidebar ---
st.sidebar.title("Player Controls")
st.sidebar.metric("Current Round", st.session_state.game_state['round'])

if not st.session_state.game_state['running']:
    if st.sidebar.button("Start Game", type="primary"):
        st.session_state.game_state['running'] = True
        st.rerun()
else:
    if st.sidebar.button("Next Round (Run 1 Round)", type="primary"):
        run_simulation_round()
    
    if st.sidebar.button("Reset Game"):
        initialize_game()
        st.rerun()

# --- Main Scoreboard ---
st.subheader("My Factory Scoreboard 📈")
player_data = st.session_state.player_data

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Net Profit", f"${player_data['net_profit']:,.0f}")
col2.metric("💵 Revenue", f"${player_data['revenue']:,.0f}")
col3.metric("🛠️ Maint. Costs", f"${player_data['maint_costs']:,.0f}")
col4.metric("🔥 Failure Costs", f"${player_data['failure_costs']:,.0f}")


st.write("---")

# --- Player Dashboard ---
st.subheader("My Operations Dashboard")

# Display pump status overview
pump_status_df = pd.DataFrame(player_data['pump_states']).T
pump_status_df['Downtime'] = [f"{player_data['pump_downtime'][p]} Rounds" for p in pump_status_df.index]

# Apply color coding for probability
def color_prob(val):
    if isinstance(val, float):
        color = 'lightgreen'
        if val > CATASTROPHE_THRESHOLD: color = 'red'
        elif val > 0.8: color = 'salmon'
        elif val > 0.5: color = 'orange'
        elif val > 0.2: color = 'yellow'
        return f'background-color: {color}'
    return ''
    
st.dataframe(
    pump_status_df.drop(columns=['inject_anomaly']).style.applymap(color_prob, subset=['prob_of_failure']),
    use_container_width=True
)

# Detailed pump data
st.write("---")
st.subheader("Detailed Pump Insights & Sensor Data")
pump_detail_tabs = st.tabs([f'Pump {i}' for i in range(1, NUM_PUMPS + 1)])

for i, pump_tab in enumerate(pump_detail_tabs):
    pump_id = f'Pump {i+1}'
    with pump_tab:
        current_pump_state = player_data['pump_states'][pump_id]
        
        # --- PLAYER ACTION BUTTON ---
        if st.button(f"Simulate Proactive Maintenance for {pump_id}", key=f'maintain_{pump_id}'):
            apply_proactive_maintenance(pump_id)

        st.write(f"#### {pump_id} Health Status")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Probability of Failure", f"{current_pump_state['prob_of_failure']:.1%}")
        col2.metric("Est. Time to Failure", current_pump_state['time_to_failure'])
        col3.metric("Recommended Action", current_pump_state['recommendation'])
        col4.metric("Downtime Left", f"{player_data['pump_downtime'][pump_id]} Rounds")

        st.write("---")
        st.write(f"#### {pump_id} Sensor Data Trends")
        
        df_pump = player_data['pump_data_history'][pump_id]
        if not df_pump.empty:
            # Plot Vibration
            fig_vibration = px.line(df_pump, x='timestamp', y='vibration', title=f'{pump_id} Vibration')
            fig_vibration.add_hline(y=player_data['pump_baselines'][pump_id]['vibration'] * ANOMALY_THRESHOLD_VIBRATION,
                                    line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            st.plotly_chart(fig_vibration, use_container_width=True)

            # Plot Temperature
            fig_temp = px.line(df_pump, x='timestamp', y='temperature', title=f'{pump_id} Temperature')
            fig_temp.add_hline(y=player_data['pump_baselines'][pump_id]['temperature'] + ANOMALY_THRESHOLD_TEMP,
                                line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            fig_temp.add_hline(y=TEMP_CRITICAL_THRESHOLD, line_dash="dot", line_color="purple", annotation_text="Critical Temp")
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Plot Current
            fig_current = px.line(df_pump, x='timestamp', y='current', title=f'{pump_id} Current Draw')
            fig_current.add_hline(y=player_data['pump_baselines'][pump_id]['current'] * 1.2,
                                    line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            st.plotly_chart(fig_current, use_container_width=True)
        else:
            st.info("No data yet. Start the game and click 'Next Round'!")
