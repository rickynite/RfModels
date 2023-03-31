import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class RfModelData:
    def __init__(self, num_samples=5, num_freq_points=25):
        self.num_samples = num_samples
        self.num_freq_points = num_freq_points

    def generate_data(self, filename=None):
        np.random.seed(42)

        # Generate temperature data
        temperatures = np.linspace(0, 100, self.num_samples)

        # Generate frequency points
        freq_points = np.linspace(1, 5, self.num_freq_points) * 1e9

        # Define a bandpass filter-like S21 magnitude function
        def s21_magnitude_func(temp, freq, a, b, c, d, e):
            return a * temp + b * (1 / freq) + c + d * np.exp(-e * (freq - 3e9)**2)

        # Generate S21 magnitude data with higher sensitivity at low temperatures and high frequencies
        a_values = np.linspace(0.1, 0.55, self.num_freq_points)
        b_values = np.linspace(1e-9, 5.5e-9, self.num_freq_points)
        c_values = np.full(self.num_freq_points, -20)
        d_values = np.full(self.num_freq_points, 15)
        e_values = np.full(self.num_freq_points, 1e-18)

        s21_magnitudes = np.zeros((self.num_samples, len(freq_points)))

        for i, temp in enumerate(temperatures):
            for j, freq in enumerate(freq_points):
                s21_magnitudes[i, j] = s21_magnitude_func(temp, freq, a_values[j], b_values[j], c_values[j], d_values[j], e_values[j])

        # Clip S21 magnitude data between -40 dB and -9 dB
        s21_magnitudes = np.clip(s21_magnitudes, -40, -9)

        # Add measurement noise
        noise_std = 0.5
        s21_magnitudes += np.random.normal(0, noise_std, s21_magnitudes.shape)

        # Create a pandas DataFrame with timestamp, temperature, and S21 magnitude data
        timestamps = pd.date_range("2021-01-01", periods=self.num_samples, freq="1H")
        data = pd.DataFrame({"timestamp": timestamps, "temperature": temperatures})
        for i, freq in enumerate(freq_points):
            data[f"{freq/1e9:.3f}"] = s21_magnitudes[:, i]

        # Save the dummy data as a CSV file
        if filename:
            data.to_csv(filename, index=False)
        
        return data

    def plot_data(self, csvfile):
        data = pd.read_csv(csvfile)
        # Extract temperature and S21 magnitude data
        temperatures = data["temperature"].values
        freq_points = np.linspace(1, 5, self.num_freq_points)
        s21_magnitudes = data[[f"{freq:.3f}" for freq in freq_points]].values

        # Create the plot
        fig = go.Figure()

        # Plot the S21 magnitude data vs. frequency for each temperature
        for i in range(len(temperatures)):
            temp = temperatures[i]
            s21_data = s21_magnitudes[i, :]
            fig.add_trace(go.Scatter(x=freq_points, y=s21_data, mode='lines', name=f'Temperature: {temp}°C'))

        # Set labels and legend
        fig.update_layout(
            height=600,
            width=1200,
            title="S21 Magnitude Data vs. Frequency for Different Temperatures",
            xaxis_title="Frequency (GHz)",
            yaxis_title="S21 Magnitude (dB)",
            legend=dict(
                x=1,
                y=1,
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                    color='black'
                ),
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(255, 255, 255, 0)',
                borderwidth=1
            )
        )

        # Set y-axis range
        #fig.update_yaxes(range=[-40, -9])

        # Display the plot
        fig.show()
                                             
