from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    """
    Anomaly detection class using Isolation Forest and Z-Score.
    This class is designed to detect anomalies in real-time data streams.
    """
    
    def __init__(self, contamination=0.01, n_estimators=100):
        """
        Initializes the anomaly detection models.

        Args:
        - contamination (float): The proportion of anomalies in the data.
        - n_estimators (int): The number of trees in the Isolation Forest.
        """
        self.isolation_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=91)
        self.fitted_if = False
        self.data_points = []
        self.mean = None
        self.std_dev = None

    def fit(self, data):
        """
        Fits the Isolation Forest model and calculates mean and std_dev for Z-Score.

        Args:
        - data (array-like): The initial data to fit the models.
        """
        data = np.array(data).reshape(-1, 1)

        # Fit Isolation Forest
        self.isolation_forest.fit(data)
        self.fitted_if = True

        # Calculate mean and std for Z-Score
        self.mean = np.mean(data)
        self.std_dev = np.std(data)
        self.data_points = data.flatten().tolist()

    def detect_isolation_forest(self, new_data):
        """
        Detects anomalies using Isolation Forest.

        Args:
        - new_data (float): The new data point to evaluate.

        Returns:
        - bool: True if anomaly, False otherwise.
        """
        if not self.fitted_if:
            raise RuntimeError("Isolation Forest model must be fitted before calling detect.")
        
        new_data = np.array([new_data]).reshape(1, -1)
        prediction = self.isolation_forest.predict(new_data)
        return prediction[0] == -1  # Return True if anomaly

    def detect_z_score(self, new_data):
        """
        Detects anomalies using Z-Score method.

        Args:
        - new_data (float): The new data point to evaluate.

        Returns:
        - bool: True if anomaly, False otherwise.
        """
        if self.mean is None or self.std_dev is None:
            raise RuntimeError("Mean and standard deviation must be calculated before using Z-Score.")

        if self.std_dev == 0:
            return False  # No variance, no anomalies possible

        z_score = (new_data - self.mean) / self.std_dev
        return abs(z_score) > 3  # Return True if z-score > 3 (anomaly)
