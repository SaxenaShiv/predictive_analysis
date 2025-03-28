import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ollama

class PredictiveAnalyzer:
    def __init__(self, equipment_type):
        """
        Initialize predictive analyzer for specific equipment type
        
        Args:
            equipment_type (str): Type of manufacturing equipment
        """
        self.equipment_type = equipment_type
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # 10% potential anomalies
            random_state=42
        )
    
    def detect_anomalies(self, data):
        """
        Detect anomalies in sensor data
        
        Args:
            data (pd.DataFrame): Sensor data
        
        Returns:
            dict: Anomaly detection results
        """
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Scale data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        # Detect anomalies
        anomaly_labels = self.anomaly_detector.fit_predict(scaled_data)
        
        # Calculate anomaly metrics
        total_samples = len(data)
        anomalous_samples = np.sum(anomaly_labels == -1)
        anomaly_percentage = (anomalous_samples / total_samples) * 100
        
        return {
            'total_samples': total_samples,
            'anomalous_samples': anomalous_samples,
            'anomaly_percentage': anomaly_percentage,
            'anomaly_details': numeric_data[anomaly_labels == -1]
        }
    
    def generate_predictive_insights(self, data, anomalies):
        """
        Generate predictive insights using Ollama Qwen model
        
        Args:
            data (pd.DataFrame): Original sensor data
            anomalies (dict): Anomaly detection results
        
        Returns:
            str: Predictive insights from Ollama
        """
        try:
            # Prepare detailed prompt for predictive insights
            prompt = f"""
            Provide predictive maintenance insights for {self.equipment_type}:
            
            Data Overview:
            - Total Samples: {anomalies['total_samples']}
            - Anomalous Samples: {anomalies['anomalous_samples']}
            - Anomaly Percentage: {anomalies['anomaly_percentage']:.2f}%
            
            Anomaly Details:
            {anomalies['anomaly_details'].to_string()}
            
            Key Analysis Requests:
            1. Predict potential equipment failure risks
            2. Recommend preventive maintenance actions
            3. Suggest optimization strategies
            4. Estimate remaining useful life
            """
            
            # Generate response using Ollama Qwen model
            response = ollama.chat(
                model='qwen2.5:1.5b ',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"Error in generating predictive insights: {str(e)}"