# Real-Time Data Stream Anomaly Detection

![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This project implements a **Real-Time Anomaly Detection System** designed to identify unusual patterns in continuous data streams. Whether applied to financial transactions, system metrics, or other types of sequential data, the system uses multiple algorithms to flag outliers and deviations from expected behavior.

The system offers real-time anomaly detection using:
- **Z-Score Method**: A statistical approach to detecting outliers based on standard deviations.
- **Isolation Forest Algorithm**: A machine learning model for detecting anomalies, specifically suited for high-dimensional data and unsupervised learning.

The entire project is implemented in Python with minimal external dependencies, making it lightweight, easy to deploy, and adaptable for various use cases.

## Features

- **Real-Time Stream Simulation**: Generates continuous data streams simulating seasonal patterns and random noise.
- **Multiple Anomaly Detection Algorithms**: Choose between Z-Score or Isolation Forest to detect outliers.
- **Dynamic Visualizations**: Real-time plots of the data stream with flagged anomalies.
- **Web Interface**: A user-friendly interface built with Streamlit for controlling parameters and monitoring the detection process.
- **Modular Design**: Clean, extendable architecture with well-organized code structure.
- **Test-Driven Development (TDD)**: Extensive unit tests ensure reliability and correctness.

## Project Structure

```
Stream-Data-Anomaly-Detection
├── src
│   ├── __init__.py
│   ├── anomaly_detection.py     # Anomaly detection algorithms (Z-Score & Isolation Forest)
│   ├── stream_simulation.py      # Data stream simulation
├── tests
│   ├── __init__.py
│   ├── test_anomaly_detection.py # Unit tests for anomaly detection
│   ├── test_stream_simulation.py # Unit tests for stream simulation
├── web_app.py                    # Streamlit app for real-time data stream and anomaly visualization
├── LICENSE                       
├── README.md                     
├── requirements.txt              
```

## Installation

To install and run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/AITYOUB-Abdelmoughit/Stream-Data-Anomaly-Detection.git
   cd Stream-Data-Anomaly-Detection
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv env  # Or Python3 in Unix-like OS
   `env\Scripts\activate`   # On Unix-like OS use "source env/bin/activate"
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit web app:
   ```bash
   streamlit run web_app.py
   ```

## Usage

Once the web app is running, you can simulate real-time data streams and detect anomalies. The app includes controls for:

- **Stream Length**: Set the length of the data stream to simulate.
- **Noise Level**: Adjust the randomness in the data.
- **Algorithm**: Choose between the Z-Score or Isolation Forest algorithm for anomaly detection.
- **Contamination Rate** (for Isolation Forest): Set the proportion of anomalies in the data.

The results are displayed in real-time, with a live plot showing the data stream and any anomalies detected.

## Anomaly Detection Algorithms

### 1. **Z-Score Method**
The Z-Score method is a simple yet effective statistical approach to detect outliers. It calculates the Z-score for each incoming data point, which measures how many standard deviations the data point is from the mean of the data.

**How it Works:**
- For each data point `x`, calculate the Z-score:
  ```python
  z_score = (x - mean) / standard_deviation
  ```
- If the Z-score is greater than 3 or less than -3 (i.e., the data point is more than 3 standard deviations away from the mean), it is flagged as an anomaly.

This method is computationally efficient and works well when the data is normally distributed.

### 2. **Isolation Forest Algorithm**
Isolation Forest is an ensemble learning method that excels in unsupervised anomaly detection. It isolates observations by randomly selecting a feature and splitting it into random intervals. Anomalies are isolated quickly because they are few and differ significantly from normal data. [Click Here For More](https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/#:~:text=The%20Isolation%20Forest%20algorithm,%20introduced%20by%20Fei%20Tony%20Liu%20and)

**How it Works:**
- The algorithm randomly partitions data using decision trees.
- Anomalies require fewer splits to isolate, whereas normal data requires more splits.
- The contamination rate defines the proportion of anomalies expected in the dataset.
  
Unlike the Z-Score, Isolation Forest does not assume a specific distribution of the data and works well in cases where the data might be high-dimensional or non-Gaussian.

## Testing

The project follows a **Test-Driven Development (TDD)** approach. To run the unit tests:

1. Ensure you are in the root directory of the project.
2. Run the following command to execute the tests:
   ```bash
   python -m unittest discover tests
   ```

Unit tests cover both the anomaly detection algorithms and the stream simulation logic, ensuring robustness and reliability.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

