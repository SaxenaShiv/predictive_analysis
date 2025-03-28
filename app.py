
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from groq import Groq
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
import traceback
import io

load_dotenv()

class ManufacturingIntelligenceDashboard:
    def __init__(self, equipment_type, groq_api_key):
        self.equipment_type = equipment_type
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Enhanced equipment types with more nuanced parameter characteristics
        self.equipment_types = {
            "CNC Machine": {
                "parameters": ["Spindle Speed", "Cutting Force", "Tool Wear", "Temperature"],
                "ranges": {
                    "Spindle Speed": (0, 10000),
                    "Cutting Force": (0, 5000),
                    "Tool Wear": (0, 100),
                    "Temperature": (20, 120)
                },
                "initial_values": {
                    "Spindle Speed": 5000,
                    "Cutting Force": 2500,
                    "Tool Wear": 50,
                    "Temperature": 70
                },
                "drift_scales": {
                    "Spindle Speed": 0.02,
                    "Cutting Force": 0.015,
                    "Tool Wear": 0.01,
                    "Temperature": 0.01
                },
                "units": {
                    "Spindle Speed": "RPM",
                    "Cutting Force": "N",
                    "Tool Wear": "%",
                    "Temperature": "°C"
                }
            },
            "Injection Molding Machine": {
                "parameters": ["Injection Pressure", "Barrel Temperature", "Cooling Time", "Cycle Time"],
                "ranges": {
                    "Injection Pressure": (0, 2000),
                    "Barrel Temperature": (100, 350),
                    "Cooling Time": (5, 60),  # Minimum cooling time realistic for most processes
                    "Cycle Time": (10, 120)  # Minimum cycle time more realistic
                },
                "initial_values": {
                    "Injection Pressure": 1000,
                    "Barrel Temperature": 225,
                    "Cooling Time": 30,
                    "Cycle Time": 60
                },
                "drift_scales": {
                    "Injection Pressure": 0.03,
                    "Barrel Temperature": 0.02,
                    "Cooling Time": 0.015,
                    "Cycle Time": 0.01
                },
                "units": {
                    "Injection Pressure": "bar",
                    "Barrel Temperature": "°C",
                    "Cooling Time": "sec",
                    "Cycle Time": "sec"
                },
                "anomaly_thresholds": {
                    "Injection Pressure": 200,  # Significant deviation
                    "Barrel Temperature": 25,   # Large temperature variation
                    "Cooling Time": 10,         # Substantial time difference
                    "Cycle Time": 15            # Noteworthy cycle time change
                }
            },
            "Robotic Assembly Line": {
                "parameters": ["Positioning Accuracy", "Cycle Speed", "Power Consumption", "Vibration"],
                "ranges": {
                    "Positioning Accuracy": (0, 0.1),
                    "Cycle Speed": (0, 50),
                    "Power Consumption": (0, 10),
                    "Vibration": (0, 5)
                },
                "initial_values": {
                    "Positioning Accuracy": 0.05,
                    "Cycle Speed": 25,
                    "Power Consumption": 5,
                    "Vibration": 2.5
                },
                "drift_scales": {
                    "Positioning Accuracy": 0.005,
                    "Cycle Speed": 0.02,
                    "Power Consumption": 0.015,
                    "Vibration": 0.01
                },
                "units": {
                    "Positioning Accuracy": "mm",
                    "Cycle Speed": "cycles/min",
                    "Power Consumption": "kW",
                    "Vibration": "mm/s"
                },
                "anomaly_thresholds": {
                    "Positioning Accuracy": 0.02,  # Significant positioning deviation
                    "Cycle Speed": 5,              # Notable speed change
                    "Power Consumption": 1,        # Meaningful power variation
                    "Vibration": 1                 # Substantial vibration increase
                }
            }
        }
        
        # Initialize predictive analyzer
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # 10% potential anomalies
            random_state=42
        )
    
    def generate_realistic_sensor_value(self, param, current_value=None):
        """Generate a realistic sensor value with sophisticated drift and noise"""
        equipment_config = self.equipment_types[self.equipment_type]
        ranges = equipment_config["ranges"][param]
        drift_scales = equipment_config["drift_scales"][param]
        initial_values = equipment_config["initial_values"][param]
        
        # If no current value, use initial value
        if current_value is None:
            current_value = initial_values
        
        # More nuanced drift with equipment-specific characteristics
        drift = np.random.normal(0, (ranges[1] - ranges[0]) * drift_scales)
        noise = np.random.normal(0, (ranges[1] - ranges[0]) * 0.05)
        
        # Add cyclical/sinusoidal variation to make data more realistic
        cycle_factor = np.sin(datetime.now().timestamp() / 100) * (ranges[1] - ranges[0]) * 0.05
        
        # Combine all factors
        new_value = float(current_value) + drift + noise + cycle_factor
        
        # Ensure value stays within range
        return float(np.clip(new_value, ranges[0], ranges[1]))
    
    def generate_live_sensor_data(self, existing_data=None):
        """Generate next data point for live streaming"""
        params = self.equipment_types[self.equipment_type]["parameters"]
        
        # If no existing data, initialize with initial values
        if existing_data is None or len(existing_data) == 0:
            initial_values = self.equipment_types[self.equipment_type]["initial_values"]
            data = {param: initial_values[param] for param in params}
            data['Timestamp'] = datetime.now()
            return pd.DataFrame([data]).set_index('Timestamp')
        
        # Generate next data point based on existing data
        new_data = {}
        for param in params:
            # Find the correct column name (handle potential case sensitivity or slight differences)
            matching_columns = [col for col in existing_data.columns if param.lower() in col.lower()]
            
            if matching_columns:
                # Use the first matching column
                column = matching_columns[0]
                current_value = existing_data[column].iloc[-1] if len(existing_data) > 0 else None
            else:
                # If no matching column found, use the initial value
                current_value = self.equipment_types[self.equipment_type]["initial_values"][param]
            
            new_data[param] = self.generate_realistic_sensor_value(param, current_value)
        
        new_data['Timestamp'] = datetime.now()  
    
        # Create new DataFrame and concatenate
        new_df = pd.DataFrame([new_data]).set_index('Timestamp')
        return pd.concat([existing_data, new_df])
    
    def detect_anomalies(self, data):
        """Detect anomalies in sensor data with NaN handling"""
        if len(data) < 2:
            return {
                'total_samples': 0,
                'anomalous_samples': 0,
                'anomaly_percentage': 0,
                'anomaly_details': pd.DataFrame()
            }
        
        # Select numeric columns and handle NaN values
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Drop rows with NaN values to prevent issues with anomaly detection
        numeric_data_cleaned = numeric_data.dropna()
        
        # If no data remains after dropping NaNs, return empty results
        if len(numeric_data_cleaned) < 2:
            return {
                'total_samples': len(numeric_data),
                'anomalous_samples': 0,
                'anomaly_percentage': 0,
                'anomaly_details': pd.DataFrame()
            }
        
        # Scale data
        try:
            scaled_data = self.scaler.fit_transform(numeric_data_cleaned)
        except ValueError:
            # Fallback if scaling fails
            return {
                'total_samples': len(numeric_data),
                'anomalous_samples': 0,
                'anomaly_percentage': 0,
                'anomaly_details': pd.DataFrame()
            }
        
        # Detect anomalies with a more robust method
        try:
            anomaly_detector = IsolationForest(
                contamination=0.1,  # 10% potential anomalies
                random_state=42,
                max_samples='auto',  # Automatically determine max_samples
                bootstrap=False      # Avoid potential sampling issues
            )
            
            anomaly_labels = anomaly_detector.fit_predict(scaled_data)
            
            # Custom anomaly detection for specific equipment types
            if self.equipment_type in ["Injection Molding Machine", "Robotic Assembly Line"]:
                anomaly_thresholds = self.equipment_types[self.equipment_type].get("anomaly_thresholds", {})
                
                for param, threshold in anomaly_thresholds.items():
                    if param in numeric_data_cleaned.columns:
                        deviation = abs(numeric_data_cleaned[param] - numeric_data_cleaned[param].mean())
                        param_anomalies = deviation > threshold
                        anomaly_labels[param_anomalies.values] = -1
            
            # Calculate anomaly metrics
            total_samples = len(numeric_data)
            anomalous_samples = np.sum(anomaly_labels == -1)
            anomaly_percentage = (anomalous_samples / total_samples) * 100
            
            return {
                'total_samples': total_samples,
                'anomalous_samples': anomalous_samples,
                'anomaly_percentage': anomaly_percentage,
                'anomaly_details': numeric_data_cleaned[anomaly_labels == -1]
            }
        
        except Exception as e:
            # Fallback error handling
            print(f"Anomaly detection error: {e}")
            return {
                'total_samples': len(numeric_data),
                'anomalous_samples': 0,
                'anomaly_percentage': 0,
                'anomaly_details': pd.DataFrame()
            }
    
    def create_real_time_visualization(self, data, anomalies):
        """Create real-time interactive visualization"""
        # Retrieve units for the current equipment type
        units = self.equipment_types[self.equipment_type]["units"]
        
        # Create trace for each parameter
        traces = []
        for column in data.columns:
            if column != 'Timestamp':
                traces.append(go.Scatter(
                    x=data.index, 
                    y=data[column], 
                    mode='lines+markers', 
                    name=f"{column} ({units.get(column, '')})",
                    line=dict(width=3)
                ))
        
        # Add anomaly markers if exists
        anomaly_data = anomalies['anomaly_details']
        if not anomaly_data.empty:
            traces.append(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data.iloc[:, 0],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
        
        # Create layout
        layout = go.Layout(
            title=f'{self.equipment_type} Performance',
            xaxis_title='Time',
            yaxis_title='Sensor Values',
            height=500
        )
        
        return go.Figure(data=traces, layout=layout)
    
    def export_data_to_csv(self, data):
        """
        Convert DataFrame to CSV for download
        
        Args:
            data (pd.DataFrame): DataFrame to convert to CSV
        
        Returns:
            str: CSV content as a string
        """
        # Convert index to a column if it's a datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            data_copy = data.reset_index()
        else:
            data_copy = data.copy()
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        data_copy.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

# Rest of the code remains the same
def main():
    # Streamlit app configuration
    st.set_page_config(
        page_title="Real-time Manufacturing Intelligence", 
        page_icon="🏭", 
        layout="wide"
    )
    
    # Retrieve API Key from environment variable
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    # Optional: Add a fallback or warning if API key is not set
    if not groq_api_key:
        st.error("Groq API Key not found. Please set GROQ_API_KEY in your .env file.")
        return
    
    # Sidebar configuration
    st.sidebar.header("Dashboard Controls")
    equipment_type = st.sidebar.selectbox(
        "Select Equipment Type", 
        ["CNC Machine", "Injection Molding Machine", "Robotic Assembly Line"]
    )
    
    # Pause/Resume toggle
    pause_toggle = st.sidebar.toggle("Pause Real-time Updates", value=False)
    
    # Initialize dashboard
    try:
        dashboard = ManufacturingIntelligenceDashboard(equipment_type, groq_api_key)
    except Exception as e:
        st.error(f"Error initializing dashboard: {str(e)}")
        return
    
    # Initialize session state for data and pause state
    if 'live_data' not in st.session_state:
        st.session_state.live_data = dashboard.generate_live_sensor_data()
    
    # CSV Export section
    st.sidebar.header("Data Export")
    export_duration = st.sidebar.selectbox(
        "Export Data Duration", 
        [
            "Last 5 Minutes", 
            "Last 15 Minutes", 
            "Last 30 Minutes", 
            "Last Hour", 
            "All Data"
        ]
    )
    
    # Duration mapping
    duration_map = {
        "Last 5 Minutes": timedelta(minutes=5),
        "Last 15 Minutes": timedelta(minutes=15),
        "Last 30 Minutes": timedelta(minutes=30),
        "Last Hour": timedelta(hours=1),
        "All Data": None
    }
    
    # Export button
    if st.sidebar.button("Download CSV"):
        # Determine data to export based on selected duration
        if duration_map[export_duration] is None:
            export_data = st.session_state.live_data
        else:
            cutoff_time = datetime.now() - duration_map[export_duration]
            export_data = st.session_state.live_data[st.session_state.live_data.index >= cutoff_time]
        
        # Convert to CSV
        csv_data = dashboard.export_data_to_csv(export_data)
        
        # Create download button
        st.sidebar.download_button(
            label="Click to Download CSV",
            data=csv_data,
            file_name=f"{equipment_type}_sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Real-time performance visualization
    performance_placeholder = st.empty()
    metrics_placeholder = st.empty()
    error_placeholder = st.empty()
    
    # Number of data points to keep
    max_points = 50
    
    # Update interval control
    update_interval = st.sidebar.slider("Update Interval (seconds)", min_value=1, max_value=10, value=2)
    
    # Main data update and visualization loop
    while True:
        # Clear any previous errors
        error_placeholder.empty()
        
        # Check if paused
        if pause_toggle:
            st.sidebar.warning("Real-time updates are paused")
            time.sleep(1)
            continue
        
        try:
            # Generate new data point
            st.session_state.live_data = dashboard.generate_live_sensor_data(st.session_state.live_data)
            
            # Trim data to max points
            if len(st.session_state.live_data) > max_points:
                st.session_state.live_data = st.session_state.live_data.tail(max_points)
            
            # Detect anomalies
            anomalies = dashboard.detect_anomalies(st.session_state.live_data)
            
            # Create visualization
            performance_chart = dashboard.create_real_time_visualization(st.session_state.live_data, anomalies)
            
            # Update chart
            performance_placeholder.plotly_chart(performance_chart, use_container_width=True)
            
            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Performance", f"{st.session_state.live_data.mean().mean():.2f}")
                with col2:
                    st.metric("Performance Variance", f"{st.session_state.live_data.std().mean():.4f}")
                with col3:
                    st.metric("Peak Deviation", f"{(st.session_state.live_data.max() - st.session_state.live_data.min()).mean():.2f}")
            
            # Wait before next update
            time.sleep(update_interval)
        
        except Exception as e:
            # Detailed error logging
            error_details = traceback.format_exc()
            error_placeholder.error(f"Error in real-time update: {str(e)}")
            st.sidebar.error("An error occurred. Check the error message.")
            
            # Print full traceback for debugging
            print(error_details)
            
            # Wait before retrying
            time.sleep(1)

if __name__ == "__main__":
    main()