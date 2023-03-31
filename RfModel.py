import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from joblib import dump, load


class RfModel:
    def __init__(self):
        self.degree = None
        self.models = None
        self.freq_points = None
        self.model_type = None

    def save_model(self, file_name):
        dump(self, file_name)

    @classmethod
    def load_model(cls, file_name):
        return load(file_name)   

    def fit(self, X, y, regression_type="poly", degree=3):
        self.degree = degree
        self.freq_points = np.array(X.columns[2:], dtype=float)
        self.models = []
        for freq_label in X.columns[2:]:
            s21_magnitudes = y[freq_label].values
            if regression_type == "poly":                
                model = make_pipeline(PolynomialFeatures(self.degree), LinearRegression())
            elif regression_type == "linear":
                model = LinearRegression()
            model.fit(X["temperature"].values.reshape(-1, 1), s21_magnitudes)
            self.models.append(model)
        self.model_type = regression_type
                
    def predict(self, X):        
        y_pred = np.zeros((len(X), len(self.freq_points)))
        for i, model in enumerate(self.models):
            y_pred[:, i] = model.predict(X["temperature"].values.reshape(-1, 1)).flatten()        
        return y_pred
    
    def calculate_error(self, X, y):
        y_pred = self.predict(X)
        error = y.iloc[:, 2:].values - y_pred
        return error
    
    def compute_sensitivity(self, X):
        # Calculate the sensitivity (derivative) of the predicted S21 magnitude with respect to temperature
        y_pred = self.predict(X)
        sensitivity = np.zeros((len(X) - 1, len(self.freq_points)))
        for j in range(len(X) - 1):
            temp1 = X.iloc[j, 1]
            temp2 = X.iloc[j + 1, 1]
            s21_pred1 = y_pred[j, :]
            s21_pred2 = y_pred[j + 1, :]
            sensitivity[j, :] = (s21_pred2 - s21_pred1) / (temp2 - temp1)
        return sensitivity

    def plot(self, X, y):
        # Predict the S21 magnitude using the trained models
        y_pred = self.predict(X)
        error = self.calculate_error(X, y)
        sensitivity = self.compute_sensitivity(X)

        # Define colors for the temperature lines
        colors = ["red", "blue", "green", "purple", "orange"]

        # Plot the S21 magnitude data vs. frequency and error for each temperature
        fig_s21 = go.Figure()
        fig_error = go.Figure()
        fig_sensitivity = go.Figure()
        for i in range(len(X) - 1):
            temp = X.iloc[i, 1]
            s21_data = y.iloc[i, 2:].values
            s21_pred = y_pred[i, :]
            error_data = error[i, :]
            sensitivity_data = sensitivity[i, :]
            color = colors[i % len(colors)]

            # Create S21 Magnitude plot            
            fig_s21.add_trace(go.Scatter(x=self.freq_points, y=s21_data, mode='markers', name=f'Actual - Temp: {temp}°C', marker=dict(color=color)))
            fig_s21.add_trace(go.Scatter(x=self.freq_points, y=s21_pred, mode='lines', name=f'Predicted - Temp: {temp}°C', line=dict(color=color)))
        
            # Create Prediction Error plot            
            fig_error.add_trace(go.Scatter(x=self.freq_points, y=error_data, mode='lines', name=f'Prediction Error - Temp: {temp}°C', line=dict(color=color, width=2)))          

            # Create Sensitivity plot
            if i < len(X) - 1:
                sensitivity_data = sensitivity[i, :]
                # Create Sensitivity plot
                fig_sensitivity.add_trace(go.Scatter(x=self.freq_points, y=sensitivity_data, mode='lines', name=f'Temp: {temp}°C', line=dict(color=color, width=2)))
                
        fig_s21.update_layout(
            height=600,
            width=1200,
            title=f'S21 Magnitude vs. Frequency',
            xaxis_title='Frequency (GHz)',
            yaxis_title='S21 Magnitude (dB)',
            legend=dict(x=1.1, y=1.05, traceorder='normal')
        )      
        fig_error.update_layout(
            height=600,
            width=1200,
            title=f'Prediction Error vs. Frequency',
            xaxis_title='Frequency (GHz)',
            yaxis_title='Prediction Error (dB)',
            legend=dict(x=1.1, y=1.05, traceorder='normal')
        ) 
        fig_sensitivity.update_layout(
            height=600,
            width=1200,
            title=f'Sensitivity vs. Frequency',
            xaxis_title='Frequency (GHz)',
            yaxis_title='Sensitivity (dB/°C)',
            legend=dict(x=1.1, y=1.05, traceorder='normal')
        )

        fig_s21.show()
        fig_error.show()
        fig_sensitivity.show()

