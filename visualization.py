import plotly.express as px
import plotly.graph_objs as go

def create_equipment_charts(data, parameter, equipment_type):
    """
    Create interactive charts for equipment parameters
    """
    # Time series line chart
    fig = px.line(
        data, 
        x='Timestamp', 
        y=parameter,
        title=f"{parameter} for {equipment_type}",
        labels={'value': parameter}
    )
    
    # Customize chart appearance
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Time',
        yaxis_title=parameter,
        title_x=0.5
    )
    
    # Add statistical reference lines
    mean = data[parameter].mean()
    std = data[parameter].std()
    
    fig.add_shape(
        type='line',
        x0=data['Timestamp'].min(),
        x1=data['Timestamp'].max(),
        y0=mean,
        y1=mean,
        line=dict(color='green', width=2, dash='dash')
    )
    
    # Add standard deviation bands
    fig.add_shape(
        type='rect',
        x0=data['Timestamp'].min(),
        x1=data['Timestamp'].max(),
        y0=mean - std,
        y1=mean + std,
        fillcolor='lightgreen',
        opacity=0.2,
        layer='below',
        line_width=0
    )
    
    return fig

def create_anomaly_heatmap(anomalies):
    """
    Create a heatmap of parameter anomalies
    """
    # Implementation of anomaly heatmap visualization
    pass